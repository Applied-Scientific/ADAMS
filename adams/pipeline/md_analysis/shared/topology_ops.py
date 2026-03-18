"""Topology include-graph and position-restraint patching helpers."""
import os
import re
from pathlib import Path

from .constants import RESTRAINT_FC_VALUES

_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)
_POSRES_BLOCK_RE = re.compile(r"#ifdef\s+POSRES\b(.*?)#endif", re.DOTALL)


def generate_posre_with_fc(input_posre_path, output_posre_path, fc):
    """Generate a position restraint ITP file with a specified force constant."""
    with open(input_posre_path, "r") as fh:
        lines = fh.readlines()
    out_lines = []
    in_restraints = False
    for line in lines:
        stripped = line.split(";", 1)[0].strip()
        if stripped.startswith("["):
            in_restraints = "position_restraints" in stripped.lower()
            out_lines.append(line)
            continue
        if in_restraints and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            if len(parts) >= 5:
                out_lines.append(
                    f"{parts[0]:>6s}  {parts[1]:>4s}  {fc:>8.1f}  {fc:>8.1f}  {fc:>8.1f}\n"
                )
                continue
        out_lines.append(line)
    with open(output_posre_path, "w") as fh:
        fh.writelines(out_lines)


def generate_staged_restraint_variants(posre_files, fc_values=RESTRAINT_FC_VALUES):
    """Generate FC-variant posre files for each base posre and each force constant."""
    for base_posre in posre_files:
        for fc in fc_values:
            variant_path = posre_variant_path(base_posre, fc)
            if not os.path.exists(variant_path):
                generate_posre_with_fc(base_posre, variant_path, fc)


def _resolve_include_path(base_file: Path, include_target: str):
    candidate = (base_file.parent / include_target).resolve()
    return candidate if candidate.exists() else None


def _collect_topology_include_graph(top_path):
    start = Path(top_path).resolve()
    seen = set()
    ordered = []
    unresolved = []

    def _walk(cur: Path):
        if cur in seen or not cur.exists():
            return
        seen.add(cur)
        ordered.append(cur)
        text = cur.read_text(encoding="utf-8", errors="ignore")
        for inc in _INCLUDE_RE.findall(text):
            resolved = _resolve_include_path(cur, inc)
            if resolved is None:
                unresolved.append((cur, inc))
                continue
            _walk(resolved)

    _walk(start)
    return ordered, unresolved


def validate_topology_includes(top_path):
    _, unresolved = _collect_topology_include_graph(top_path)
    return [f"{str(src)} -> {inc}" for src, inc in unresolved]


def _raise_on_unresolved_includes(unresolved, context):
    if not unresolved:
        return
    missing = "\n".join(f"  {src} -> {inc}" for src, inc in unresolved)
    raise FileNotFoundError(f"{context}:\n{missing}")


def collect_resolved_topology_files(top_path, strict=True):
    files, unresolved = _collect_topology_include_graph(top_path)
    if unresolved and strict:
        _raise_on_unresolved_includes(unresolved, "Unresolved #include references in topology graph")
    return [str(path) for path in files]


def collect_posre_includes(top_path, strict=True):
    files, unresolved = _collect_topology_include_graph(top_path)
    if unresolved and strict:
        _raise_on_unresolved_includes(unresolved, "Unresolved #include references in topology graph")
    posre_files = []
    seen = set()
    for fpath in files:
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        for block in _POSRES_BLOCK_RE.findall(text):
            for inc in _INCLUDE_RE.findall(block):
                resolved = _resolve_include_path(fpath, inc)
                if resolved is None or resolved.suffix.lower() != ".itp":
                    continue
                if resolved not in seen:
                    seen.add(resolved)
                    posre_files.append(str(resolved))
    return posre_files


def _dedupe_existing_paths(candidates):
    checked_paths = []
    existing_paths = []
    seen = set()
    for path in candidates:
        if not path:
            continue
        resolved = os.path.abspath(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        checked_paths.append(resolved)
        if os.path.exists(resolved):
            existing_paths.append(resolved)
    return existing_paths, checked_paths


def resolve_protein_topology_assets(
    protein_top=None, explicit_posre=None, protein_dir=None, root_path=None, logger=None
):
    local_include_files = []
    posre_candidates = []
    if explicit_posre:
        posre_candidates.append(explicit_posre)
    if protein_top:
        try:
            all_files = collect_resolved_topology_files(protein_top, strict=False)
            protein_top_abs = os.path.abspath(protein_top)
            local_include_files = [os.path.abspath(p) for p in all_files if os.path.abspath(p) != protein_top_abs]
        except FileNotFoundError:
            if logger is not None:
                logger.debug("Could not fully resolve topology include graph.", exc_info=True)
            local_include_files = []
        try:
            posre_candidates.extend(collect_posre_includes(protein_top, strict=False))
        except FileNotFoundError:
            if logger is not None:
                logger.debug("Could not collect POSRES includes.", exc_info=True)
        posre_candidates.append(os.path.join(os.path.dirname(protein_top), "posre.itp"))
    if protein_dir:
        posre_candidates.append(os.path.join(protein_dir, "posre.itp"))
    if root_path:
        posre_candidates.append(os.path.join(root_path, "posre.itp"))
    existing_posre_paths, checked_posre_paths = _dedupe_existing_paths(posre_candidates)
    return {
        "local_include_files": local_include_files,
        "posre_files": existing_posre_paths,
        "checked_posre_paths": checked_posre_paths,
    }


def posre_variant_path(posre_path, fc):
    path = Path(posre_path)
    return str(path.with_name(f"{path.stem}_fc{int(fc)}{path.suffix}"))


def patch_topology_staged_restraints(top_path, strict=True):
    files, unresolved = _collect_topology_include_graph(top_path)
    if unresolved and strict:
        missing = "\n".join(f"  {src} -> {inc}" for src, inc in unresolved)
        raise FileNotFoundError("Cannot patch staged restraints due to unresolved includes:\n" + missing)
    patched_any = False
    for fpath in files:
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        original = content

        def _replace_block(match):
            block = match.group(0)
            include_targets = _INCLUDE_RE.findall(block)
            if not include_targets:
                return block
            insert_lines = []
            need_fc500 = "#ifdef POSRES_FC500" not in block
            need_fc200 = "#ifdef POSRES_FC200" not in block
            for inc in include_targets:
                if not inc.lower().endswith(".itp"):
                    continue
                base_name = os.path.basename(inc)
                base_stem, base_ext = os.path.splitext(base_name)
                base_dir = os.path.dirname(inc)
                fc500_name = f"{base_dir}/{base_stem}_fc500{base_ext}" if base_dir else f"{base_stem}_fc500{base_ext}"
                fc200_name = f"{base_dir}/{base_stem}_fc200{base_ext}" if base_dir else f"{base_stem}_fc200{base_ext}"
                if need_fc500:
                    insert_lines.extend(["#ifdef POSRES_FC500", f'#include "{fc500_name}"', "#endif", ""])
                if need_fc200:
                    insert_lines.extend(["#ifdef POSRES_FC200", f'#include "{fc200_name}"', "#endif", ""])
            if not insert_lines:
                return block
            return block.rstrip() + "\n\n" + "\n".join(insert_lines).rstrip() + "\n"

        content = _POSRES_BLOCK_RE.sub(_replace_block, content)
        if content != original:
            fpath.write_text(content, encoding="utf-8")
            patched_any = True
    if not patched_any:
        raise ValueError(
            "No #ifdef POSRES block found in topology include graph. "
            "Staged restraint workflow requires a topology that already defines POSRES includes."
        )
