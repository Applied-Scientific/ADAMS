"""
Error handling utilities for the adams pipeline.

This module provides:
- Error type hierarchy for classifying failures
- SIGINT (Ctrl+C) handling infrastructure
- Utility functions and decorators for robust error handling
- Wrappers for external library calls (RDKit, Vina, Meeko)

Design principles:
1. Per-ligand errors should skip the ligand and continue
2. Fatal errors should stop the pipeline immediately
3. SIGINT should cause immediate clean exit without saving
4. All errors should be logged appropriately
"""

import signal
import subprocess
from functools import wraps
from typing import Any, Callable, Optional

from .logger_utils import get_logger
from .utils import run_cmd

# ============================================================================
# Error Type Hierarchy
# ============================================================================


class PipelineError(Exception):
    """Base class for all pipeline errors."""

    pass


class PerLigandError(PipelineError):
    """
    Errors that affect a single ligand only.

    When raised, the pipeline should skip the affected ligand and continue
    processing remaining ligands. These are typically caused by:
    - Invalid SMILES strings
    - Conformer generation failures
    - Ligand preparation issues
    - Individual docking failures
    """

    def __init__(self, ligand_id: str, message: str, cause: Exception = None):
        self.ligand_id = ligand_id
        self.cause = cause
        super().__init__(f"Ligand {ligand_id}: {message}")


class RDKitMoleculeError(PerLigandError):
    """Error parsing SMILES or generating conformer with RDKit."""

    pass


class MeekoPreparationError(PerLigandError):
    """Error preparing ligand PDBQT file with Meeko."""

    pass


class VinaExecutionError(PerLigandError):
    """Error executing Vina docking (including sys.exit() calls)."""

    pass


class GPUExecutionError(PerLigandError):
    """Error executing GPU docking subprocess."""

    pass


class PerPoseError(PerLigandError):
    """
    Error affecting a single MD pose (inherits per-ligand semantics).

    MD poses are treated like ligands - when a pose fails, the pipeline
    should skip it and continue with remaining poses.
    """

    pass


class GROMACSExecutionError(PerPoseError):
    """Error executing GROMACS subprocess (grompp, mdrun, etc.)."""

    pass


class FatalError(PipelineError):
    """
    Fatal errors that should stop the entire pipeline.

    These are typically caused by:
    - Missing required input files (receptor, ligand CSV)
    - Invalid configuration parameters
    - Missing external executables
    - System resource exhaustion
    """

    pass


# ============================================================================
# SIGINT (Ctrl+C) Handling Infrastructure
# ============================================================================

_sigint_received = False
_original_sigint_handler = None


def setup_sigint_handler():
    """
    Install SIGINT (Ctrl+C) handler for clean shutdown.

    This should be called once in the main process before spawning workers.
    When SIGINT is received, workers will check is_sigint_pending() and
    exit gracefully without saving partial results.

    Example:
        >>> from adams.error_handling import setup_sigint_handler, is_sigint_pending
        >>> setup_sigint_handler()
        >>> # In worker loop:
        >>> for item in items:
        ...     if is_sigint_pending():
        ...         logger.info("SIGINT detected, exiting")
        ...         return
        ...     process(item)
    """
    global _sigint_received, _original_sigint_handler

    def handler(signum, frame):
        global _sigint_received
        _sigint_received = True
        logger = get_logger()
        logger.info("SIGINT (Ctrl+C) received - shutting down gracefully...")
        # Raise KeyboardInterrupt to break out of blocking operations (e.g., pool.join())
        raise KeyboardInterrupt("SIGINT received")

    # Save original handler in case we need to restore it
    _original_sigint_handler = signal.signal(signal.SIGINT, handler)


def is_sigint_pending() -> bool:
    """
    Check if SIGINT (Ctrl+C) was received.

    Workers should call this periodically (e.g., at the start of each
    iteration) and exit gracefully if True.

    Returns:
        bool: True if SIGINT was received, False otherwise

    Example:
        >>> if is_sigint_pending():
        ...     logger.info("Worker exiting due to SIGINT")
        ...     return
    """
    return _sigint_received


def reset_sigint_handler():
    """
    Reset SIGINT handler to original state.

    This is mainly useful for testing or cleanup.
    """
    global _sigint_received, _original_sigint_handler
    _sigint_received = False
    if _original_sigint_handler is not None:
        signal.signal(signal.SIGINT, _original_sigint_handler)
        _original_sigint_handler = None


# ============================================================================
# Error Handling Decorators and Utilities
# ============================================================================


def handle_per_ligand_error(operation_name: str = "operation"):
    """
    Decorator for functions that process a single ligand.

    Converts exceptions into PerLigandError with appropriate logging.
    The decorated function should have 'ligand_id' as a parameter.

    Args:
        operation_name: Name of the operation for error messages

    Example:
        >>> @handle_per_ligand_error("conformer generation")
        ... def generate_conformer(ligand_id, mol):
        ...     # If this raises, it becomes a PerLigandError
        ...     AllChem.EmbedMolecule(mol)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract ligand_id from kwargs or args
            ligand_id = kwargs.get("ligand_id", None)
            if ligand_id is None and len(args) > 0:
                # Try to find ligand_id in args by parameter name
                import inspect

                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if "ligand_id" in param_names:
                    idx = param_names.index("ligand_id")
                    if idx < len(args):
                        ligand_id = args[idx]

            if ligand_id is None:
                ligand_id = "unknown"

            try:
                return func(*args, **kwargs)
            except PerLigandError:
                # Already a PerLigandError, re-raise as-is
                raise
            except Exception as e:
                logger = get_logger()
                logger.warning(f"Error in {operation_name} for ligand {ligand_id}: {e}")
                raise PerLigandError(ligand_id, f"{operation_name} failed", cause=e)

        return wrapper

    return decorator


def log_and_skip_on_error(func: Callable) -> Callable:
    """
    Decorator that logs PerLigandError and returns None instead of raising.

    Useful for batch processing where you want to collect failures but
    continue processing.

    Example:
        >>> @log_and_skip_on_error
        ... def process_ligand(ligand_id, smiles):
        ...     mol = safe_mol_from_smiles(smiles, ligand_id)
        ...     if mol is None:
        ...         raise PerLigandError(ligand_id, "Invalid SMILES")
        ...     return mol
        ...
        >>> result = process_ligand("lig_001", "invalid_smiles")
        >>> # Returns None, logs warning
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PerLigandError as e:
            logger = get_logger()
            logger.warning(f"Skipping: {e}")
            return None
        except Exception as e:
            logger = get_logger()
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return None

    return wrapper


# ============================================================================
# RDKit Error Handling Wrappers
# ============================================================================


def safe_mol_from_smiles(
    smiles: str, ligand_id: str, add_hs: bool = True
) -> Optional[Any]:
    """
    Parse SMILES string with standardized error handling.

    Args:
        smiles: SMILES string to parse
        ligand_id: Identifier for the ligand (for error messages)
        add_hs: Whether to add hydrogens (default: True)

    Returns:
        RDKit Mol object or None if parsing failed

    Raises:
        RDKitMoleculeError: If SMILES parsing fails

    Example:
        >>> mol = safe_mol_from_smiles("CCO", "ethanol")
        >>> if mol is None:
        ...     print("Failed to parse")
    """
    from rdkit.Chem import AddHs, MolFromSmiles

    logger = get_logger()

    try:
        mol = MolFromSmiles(smiles)
        if mol is None:
            raise RDKitMoleculeError(
                ligand_id, f"RDKit could not parse SMILES: {smiles}", cause=None
            )

        if add_hs:
            mol = AddHs(mol)

        return mol

    except RDKitMoleculeError:
        # Re-raise our custom error
        raise
    except Exception as e:
        logger.warning(f"Unexpected error parsing SMILES for {ligand_id}: {e}")
        raise RDKitMoleculeError(
            ligand_id, f"Failed to parse SMILES: {str(e)}", cause=e
        )


def safe_generate_conformer(mol: Any, ligand_id: str, max_attempts: int = 50) -> bool:
    """
    Generate 3D conformer with standardized error handling.

    Uses ETKDGv2 algorithm with MMFF94s optimization. Falls back to
    random coordinates if embedding fails.

    Args:
        mol: RDKit Mol object
        ligand_id: Identifier for the ligand (for error messages)
        max_attempts: Maximum embedding attempts (default: 50)

    Returns:
        bool: True if conformer generated successfully, False otherwise

    Raises:
        RDKitMoleculeError: If conformer generation fails completely

    Example:
        >>> mol = MolFromSmiles("CCO")
        >>> success = safe_generate_conformer(mol, "ethanol")
        >>> if success:
        ...     print("Conformer generated")
    """
    from rdkit.Chem import AllChem

    logger = get_logger()

    try:
        ps = AllChem.ETKDGv2()
        rid = -1

        # Try to embed molecule up to max_attempts
        for _ in range(max_attempts):
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == 0:
                break

        # If embedding failed, try using random coordinates
        if rid != 0:
            logger.debug(
                f"Ligand {ligand_id}: Using random coords for conformer generation"
            )
            ps.useRandomCoords = True
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid != 0:
                raise RDKitMoleculeError(
                    ligand_id, "Failed to generate 3D conformer even with random coords"
                )

        # Optimize geometry if conformer was generated
        if mol.GetNumConformers() > 0:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=500)
            return True
        else:
            raise RDKitMoleculeError(ligand_id, "No conformer available to optimize")

    except RDKitMoleculeError:
        # Re-raise our custom error
        raise
    except Exception as e:
        logger.warning(f"Unexpected error generating conformer for {ligand_id}: {e}")
        raise RDKitMoleculeError(
            ligand_id, f"Conformer generation failed: {str(e)}", cause=e
        )


def safe_read_and_generate_conformer(smiles: str, ligand_id: str) -> Optional[Any]:
    """
    Combined SMILES parsing and conformer generation with error handling.

    This is a convenience function that combines safe_mol_from_smiles
    and safe_generate_conformer into a single operation.

    Args:
        smiles: SMILES string
        ligand_id: Identifier for the ligand

    Returns:
        RDKit Mol object with 3D conformer, or None if failed

    Example:
        >>> mol = safe_read_and_generate_conformer("CCO", "ethanol")
        >>> if mol is not None:
        ...     print("Ready for docking")
    """
    try:
        mol = safe_mol_from_smiles(smiles, ligand_id, add_hs=True)
        if mol is None:
            return None

        success = safe_generate_conformer(mol, ligand_id)
        if not success:
            return None

        return mol

    except (RDKitMoleculeError, PerLigandError):
        # Error already logged by the wrapper functions
        return None


# ============================================================================
# Vina Error Handling Wrappers
# ============================================================================


def safe_vina_operation(operation_name: str = "Vina operation"):
    """
    Decorator for Vina operations that may call sys.exit().

    Vina library sometimes calls sys.exit() instead of raising exceptions,
    particularly for missing affinity maps. This decorator catches SystemExit
    and converts it to VinaExecutionError.

    Args:
        operation_name: Name of the operation for error messages

    Example:
        >>> @safe_vina_operation("docking")
        ... def dock_ligand(v, ligand_id, ligand_pdbqt):
        ...     v.set_ligand_from_string(ligand_pdbqt)
        ...     v.dock(exhaustiveness=32, n_poses=5)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract ligand_id from kwargs or args
            ligand_id = kwargs.get("ligand_id", None)
            if ligand_id is None and len(args) > 0:
                import inspect

                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if "ligand_id" in param_names:
                    idx = param_names.index("ligand_id")
                    if idx < len(args):
                        ligand_id = args[idx]

            if ligand_id is None:
                ligand_id = "unknown"

            logger = get_logger()

            try:
                return func(*args, **kwargs)
            except SystemExit as e:
                # Vina called sys.exit() - usually missing affinity map
                exit_code = e.code if e.code is not None else 1
                error_msg = f"Vina called sys.exit({exit_code}) - likely missing affinity map for atom type"
                logger.warning(f"Ligand {ligand_id}: {error_msg}")
                raise VinaExecutionError(ligand_id, error_msg, cause=e)
            except VinaExecutionError:
                # Already our error type, re-raise
                raise
            except Exception as e:
                logger.warning(
                    f"Vina {operation_name} failed for ligand {ligand_id}: {e}"
                )
                raise VinaExecutionError(
                    ligand_id, f"{operation_name} failed: {str(e)}", cause=e
                )

        return wrapper

    return decorator


# ============================================================================
# Meeko Error Handling Wrappers
# ============================================================================


def safe_meeko_preparation(mol: Any, ligand_id: str) -> Optional[tuple]:
    """
    Prepare ligand PDBQT with Meeko with error handling.

    Args:
        mol: RDKit Mol object
        ligand_id: Identifier for the ligand

    Returns:
        Tuple of (pdbqt_string, is_ok) or None if preparation failed

    Raises:
        MeekoPreparationError: If PDBQT preparation fails

    Example:
        >>> result = safe_meeko_preparation(mol, "lig_001")
        >>> if result:
        ...     pdbqt_string, is_ok = result
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    logger = get_logger()

    try:
        ligprep = MoleculePreparation()
        lig_setup = ligprep.prepare(mol)

        lig_string = None
        is_ok = False
        error_msg = None

        for setup in lig_setup:
            lig_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                break
            else:
                logger.debug(f"Ligand {ligand_id} setup had issues: {error_msg}")

        if lig_string is None or not is_ok:
            raise MeekoPreparationError(
                ligand_id, f"PDBQT preparation failed: {error_msg or 'Unknown error'}"
            )

        return (lig_string, is_ok)

    except MeekoPreparationError:
        # Re-raise our custom error
        raise
    except Exception as e:
        logger.warning(f"Unexpected error in Meeko preparation for {ligand_id}: {e}")
        raise MeekoPreparationError(
            ligand_id, f"PDBQT preparation failed: {str(e)}", cause=e
        )


# ============================================================================
# Subprocess Error Handling Wrappers
# ============================================================================


def safe_subprocess_call(
    cmd, ligand_id: str = None, operation_name: str = "subprocess"
) -> Any:
    """
    Execute subprocess with error handling.

    Wraps subprocess.CalledProcessError into appropriate error type.
    For GPU docking, converts to GPUExecutionError (per-ligand).

    Args:
        cmd: Command to execute (list or string)
        ligand_id: Optional ligand identifier (for per-ligand operations)
        operation_name: Name of the operation for error messages

    Returns:
        subprocess.CompletedProcess result

    Raises:
        GPUExecutionError: If ligand_id is provided and subprocess fails
        FatalError: If ligand_id is None and subprocess fails (not per-ligand)

    Example:
        >>> try:
        ...     result = safe_subprocess_call(["ls", "-l"], operation_name="file listing")
        ... except Exception as e:
        ...     print(f"Command failed: {e}")
    """
    logger = get_logger()

    try:
        return run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        if ligand_id is not None:
            # Per-ligand operation (e.g., GPU docking)
            error_msg = f"{operation_name} failed with exit code {e.returncode}"
            logger.warning(f"Ligand {ligand_id}: {error_msg}")
            raise GPUExecutionError(ligand_id, error_msg, cause=e)
        else:
            # Not per-ligand, this is a fatal error
            error_msg = f"{operation_name} failed with exit code {e.returncode}"
            logger.error(error_msg)
            raise FatalError(error_msg)
