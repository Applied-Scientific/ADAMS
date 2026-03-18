"""
Strict grompp warning parsing and policy/persistence helpers.
"""

import json
import os
import re
import subprocess
from typing import Optional

from ....logger_utils import get_logger
from ....path_config import get_subdirectory
from ....utils import run_cmd

_GROMPP_WARNING_BLOCK_RE = re.compile(
    r"WARNING\s+\d+\s*\[([^\]]*)\]\s*[:\-]?\s*\n(.*?)(?=\nWARNING\s+\d+\s*\[|\nThere\s+w(?:as|ere)\s+\d+\s+warning|\Z)",
    re.DOTALL,
)
_GROMPP_WARNING_COUNT_RE = re.compile(r"There\s+w(?:as|ere)\s+(\d+)\s+warning")
_GROMPP_TOO_MANY_WARNINGS_RE = re.compile(r"Too many warnings \((\d+)\)")


class GromppWarning:
    """A single parsed grompp warning."""

    __slots__ = ("source", "message", "fingerprint")

    def __init__(self, source: str, message: str):
        self.source = source.strip()
        self.message = message.strip()
        self.fingerprint = self._make_fingerprint()

    def _make_fingerprint(self) -> str:
        text = re.sub(r"file\s+\S+", "file <FILE>", self.source)
        text += " " + re.sub(r"\s+", " ", self.message)
        text = re.sub(r"\b\d+\b", "N", text)
        return text.strip()

    def __str__(self):
        return f"[{self.source}] {self.message}"

    def __repr__(self):
        return f"GromppWarning(source={self.source!r}, fingerprint={self.fingerprint!r})"


class GromppWarningError(RuntimeError):
    """Raised when grompp produces warnings that require user review."""

    def __init__(
        self,
        warnings: list,
        grompp_output: str,
        cmd: list,
    ):
        self.warnings = warnings
        self.grompp_output = grompp_output
        self.cmd = cmd
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        header = (
            f"grompp produced {len(self.warnings)} unapproved warning(s). "
            "Each must be reviewed before proceeding."
        )
        lines = [header, ""]
        for i, w in enumerate(self.warnings, 1):
            lines.append(f"  Warning {i}: [{w.source}]")
            for wline in w.message.splitlines():
                lines.append(f"    {wline.strip()}")
            lines.append(f"    (fingerprint: {w.fingerprint!r})")
            lines.append("")
        lines.append(
            "To classify as approved for future review continuity, pass "
            "approved_grompp_warnings={<fingerprints>} or call "
            "warning_policy.approve(fingerprint) on the GromppWarningPolicy instance."
        )
        return "\n".join(lines)


class GromppWarningPolicy:
    """
    Shared, session-scoped policy for approving/rejecting grompp warnings.
    """

    def __init__(self, approved: set = None, descriptions: dict = None):
        self._approved: set = set(approved) if approved else set()
        self._descriptions: dict = dict(descriptions) if descriptions else {}
        self._seen: list = []

    def approve(self, fingerprint: str) -> None:
        self._approved.add(fingerprint)

    def approve_all(self, warnings: list) -> None:
        for w in warnings:
            fp = w.fingerprint if isinstance(w, GromppWarning) else w
            self._approved.add(fp)

    def is_approved(self, warning: "GromppWarning") -> bool:
        return warning.fingerprint in self._approved

    def filter_unapproved(self, warnings: list) -> list:
        return [w for w in warnings if not self.is_approved(w)]

    @property
    def approved_fingerprints(self) -> set:
        return set(self._approved)

    @property
    def seen_warnings(self) -> list:
        return list(self._seen)

    def _record_seen(self, warnings: list) -> None:
        self._seen.extend(warnings)

    def __repr__(self):
        return (
            f"GromppWarningPolicy(approved={len(self._approved)}, "
            f"seen={len(self._seen)})"
        )

    def get_approved_with_descriptions(self) -> list:
        return [
            {
                "fingerprint": fp,
                "description": self._descriptions.get(fp, "(no description stored)"),
            }
            for fp in sorted(self._approved)
        ]


APPROVED_GROMPP_WARNINGS_FILENAME = "approved_grompp_warnings.json"


def get_approved_grompp_warnings_path() -> Optional[str]:
    """Path to persisted warning approvals in agent_data/memory. Returns None if agent data path not configured."""
    try:
        memory_dir = get_subdirectory("memory")
    except RuntimeError:
        return None
    memory_dir = os.path.abspath(str(memory_dir))
    return os.path.join(memory_dir, APPROVED_GROMPP_WARNINGS_FILENAME)


def load_approved_grompp_warnings(path: str) -> dict:
    """Load approved warning fingerprints/descriptions from JSON."""
    out = {"approved_fingerprints": set(), "descriptions": {}}
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out["approved_fingerprints"] = set(data.get("approved_fingerprints", []))
        out["descriptions"] = dict(data.get("descriptions", {}))
    except (OSError, json.JSONDecodeError):
        return out
    return out


def save_approved_grompp_warnings(
    path: str,
    approved_fingerprints: set,
    descriptions: dict,
) -> None:
    """Merge and save approved warning fingerprints/descriptions to JSON."""
    if not path:
        return
    existing = load_approved_grompp_warnings(path)
    merged_fp = existing["approved_fingerprints"] | approved_fingerprints
    merged_desc = {**existing["descriptions"]}
    for fp in approved_fingerprints:
        if fp in descriptions:
            merged_desc[fp] = descriptions[fp]
    merged_desc = {k: v for k, v in merged_desc.items() if k in merged_fp}
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "approved_fingerprints": sorted(merged_fp),
                    "descriptions": merged_desc,
                },
                f,
                indent=2,
            )
    except OSError:
        pass


def parse_grompp_warnings(output: str) -> list:
    """Parse grompp stdout+stderr into a list of GromppWarning objects."""
    warnings = []
    for match in _GROMPP_WARNING_BLOCK_RE.finditer(output):
        source = match.group(1).strip()
        message = match.group(2).strip()
        if message:
            warnings.append(GromppWarning(source, message))

    if not warnings:
        count_match = _GROMPP_WARNING_COUNT_RE.search(output)
        if not count_match:
            count_match = _GROMPP_TOO_MANY_WARNINGS_RE.search(output)
        if count_match and int(count_match.group(1)) > 0:
            warnings.append(
                GromppWarning(
                    "grompp",
                    f"grompp reported {count_match.group(1)} warning(s) but they could not "
                    f"be parsed individually. Relevant output:\n{output[-1500:]}",
                )
            )
    return warnings


def _truncate_description(msg: str, max_len: int = 300) -> str:
    line = msg.split("\n")[0].strip() if msg else ""
    return line[:max_len] + ("..." if len(line) > max_len else "")


# Upper bound for -maxwarn when pre-approved fingerprints exist.  The actual
# safety gate is per-warning policy validation, not this number; it only lets
# grompp complete so we can parse its full output in a single pass.
_MAXWARN_PARSE_BOUND = 100


def _strip_maxwarn(cmd: list) -> list:
    """Return a copy of *cmd* with any ``-maxwarn <value>`` pair removed."""
    out = []
    skip_next = False
    for token in cmd:
        if skip_next:
            skip_next = False
            continue
        if token == "-maxwarn":
            skip_next = True
            continue
        out.append(token)
    return out


def run_grompp(
    cmd: list,
    warning_policy: "GromppWarningPolicy" = None,
    cwd: str = None,
    input_str: str = None,
) -> "subprocess.CompletedProcess":
    """
    Run a grompp command with user-reviewed warning handling.

    Behavior:
    - No pre-approved fingerprints: run strictly without ``-maxwarn`` so every
      warning stops execution for user review.
    - Pre-approved fingerprints exist: run with a high ``-maxwarn`` bound so
      grompp can complete in a single pass; every emitted warning is still
      validated against the policy.  Unapproved warnings raise
      ``GromppWarningError``; duplicates of an approved fingerprint are
      inherently covered and need no special counting.
    """
    logger = get_logger()

    if warning_policy is None:
        warning_policy = GromppWarningPolicy()
    persist_approved_path = get_approved_grompp_warnings_path()

    def _persist_approved() -> None:
        if persist_approved_path and warning_policy._approved:
            desc = {
                w.fingerprint: _truncate_description(w.message)
                for w in warning_policy._seen
                if w.fingerprint in warning_policy._approved
            }
            save_approved_grompp_warnings(
                persist_approved_path,
                warning_policy._approved,
                desc,
            )

    clean_cmd = _strip_maxwarn(cmd)

    if warning_policy.approved_fingerprints:
        # Let grompp complete so every warning can be parsed and validated.
        # Duplicates of an approved fingerprint are inherently covered.
        run_cmd_args = [*clean_cmd, "-maxwarn", str(_MAXWARN_PARSE_BOUND)]
    else:
        # Strict: no -maxwarn, any warning stops execution for review.
        run_cmd_args = clean_cmd

    result = run_cmd(run_cmd_args, check=False, cwd=cwd, input_str=input_str)
    combined = (result.stdout or "") + (result.stderr or "")
    warnings = parse_grompp_warnings(combined)

    if warnings:
        warning_policy._record_seen(warnings)
        unapproved = warning_policy.filter_unapproved(warnings)
        if unapproved:
            logger.warning(
                "grompp produced %d warning(s), %d unapproved. "
                "Raising GromppWarningError for review.",
                len(warnings),
                len(unapproved),
            )
            raise GromppWarningError(unapproved, combined, run_cmd_args)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, run_cmd_args, result.stdout, result.stderr
        )

    _persist_approved()
    return result

