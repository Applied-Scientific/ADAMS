"""ACPYPE runner: locate executable and run topology generation for one ligand."""

import os
import shutil
import subprocess

from ....logger_utils import get_logger
from ....utils import run_cmd
from .ligand_ops import formal_charge


def find_acpype():
    """Locate the acpype executable."""
    acpype_cmd = shutil.which("acpype")
    if acpype_cmd is None:
        amber_home = os.environ.get("AMBERHOME")
        if amber_home:
            acpype_cmd = shutil.which("acpype", path=os.path.join(amber_home, "bin"))
        if acpype_cmd is None:
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                potential_path = os.path.join(conda_prefix, "bin", "acpype")
                if os.path.exists(potential_path):
                    acpype_cmd = potential_path
            if acpype_cmd is None and conda_prefix:
                acpype_cmd = shutil.which(
                    "acpype", path=os.path.join(conda_prefix, "bin")
                )
    if acpype_cmd is None:
        raise FileNotFoundError(
            "acpype executable not found. Please ensure:\n"
            "1. acpype is installed (pip install acpype or conda install acpype)\n"
            "2. acpype is in your PATH, or\n"
            "3. AMBERHOME environment variable is set correctly"
        )
    return acpype_cmd


def run_acpype(
    mol2_file,
    resname,
    charge_type="bcc",
    atom_type="gaff2",
    net_charge_override=None,
    retry_with_gas_on_failure=False,
):
    """Run ACPYPE to generate GROMACS-compatible ligand topology."""
    logger = get_logger()

    if net_charge_override is not None:
        net_charge = int(net_charge_override)
        logger.debug(
            "Using net charge from SMILES for ACPYPE: %s (file: %s)",
            net_charge,
            mol2_file,
        )
    else:
        net_charge = formal_charge(mol2_file)
        logger.debug("The net charge from file %s is %s", mol2_file, net_charge)

    workdir = os.path.dirname(mol2_file)
    acpype_cmd = find_acpype()

    cmd_bcc = [
        acpype_cmd, "-i", mol2_file, "-b", resname,
        "-a", atom_type, "-c", charge_type, "-n", str(net_charge),
    ]

    try:
        run_cmd(cmd_bcc, cwd=workdir, check=True)
    except subprocess.CalledProcessError as exc:
        auto_retry_on_odd_electrons = _should_retry_with_gas_after_failure(
            exc,
            charge_type=charge_type,
        )
        should_retry_with_gas = (
            charge_type.strip().lower() != "gas"
            and (retry_with_gas_on_failure or auto_retry_on_odd_electrons)
        )
        if not should_retry_with_gas:
            raise
        if auto_retry_on_odd_electrons and not retry_with_gas_on_failure:
            logger.warning(
                "ACPYPE %s failed for %s due to odd-electron AM1-BCC markers; retrying with gas.",
                charge_type,
                mol2_file,
            )
        else:
            logger.warning(
                "ACPYPE %s failed for %s, retrying with gas.",
                charge_type,
                mol2_file,
            )
        cmd_gas = [
            acpype_cmd, "-i", mol2_file, "-b", resname,
            "-a", atom_type, "-c", "gas", "-n", str(net_charge),
        ]
        run_cmd(cmd_gas, cwd=workdir, check=True)
        logger.info("ACPYPE gas succeeded for %s", mol2_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"acpype executable not found at {acpype_cmd}. "
            "Please check your installation and PATH configuration."
        ) from e


def _should_retry_with_gas_after_failure(
    exc: subprocess.CalledProcessError,
    *,
    charge_type: str,
) -> bool:
    """Detect ACPYPE failures that should narrow-retry with gas."""
    if (charge_type or "").strip().lower() == "gas":
        return False
    text = "\n".join(
        part for part in (exc.stdout, exc.stderr) if isinstance(part, str) and part
    ).lower()
    odd_electron_markers = (
        "number of electrons is odd",
        "the number of electrons is odd",
        "odd number of electrons",
        "spin multiplicity",
    )
    return any(marker in text for marker in odd_electron_markers)
