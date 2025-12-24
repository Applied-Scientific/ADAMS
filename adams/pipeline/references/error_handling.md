# Error Handling Patterns

## Overview
This document describes the error handling infrastructure in the pipeline. The system is designed to be robust to:
1. **SIGINT (Ctrl+C)**: User interrupts cause immediate clean exit
2. **External code failures**: Per-ligand recovery for RDKit, Vina, Meeko, GPU executable failures

## Error Type Hierarchy

The pipeline uses a structured error hierarchy defined in `adams/error_handling.py`:

```
PipelineError (base class)
├── PerLigandError (skip ligand, continue)
│   ├── RDKitMoleculeError (SMILES parsing, conformer generation)
│   ├── MeekoPreparationError (PDBQT conversion)
│   ├── VinaExecutionError (docking failure, SystemExit)
│   ├── GPUExecutionError (GPU subprocess failure)
│   └── PerPoseError (MD-specific, skip pose, continue)
│       └── GROMACSExecutionError (GROMACS subprocess failure)
└── FatalError (stop entire pipeline)
    ├── FileNotFoundError (missing receptor/input)
    └── InvalidConfigurationError (bad parameters)
```

### PerLigandError
Errors that affect a single ligand (or pose in MD) only. When raised, the pipeline skips the affected ligand and continues processing remaining ligands. These are typically caused by:
- Invalid SMILES strings
- Conformer generation failures
- Ligand preparation issues
- Individual docking failures
- MD simulation failures (via PerPoseError subclass)

**Handling**: Log warning, skip ligand/pose, continue with next ligand/pose.

#### PerPoseError (MD-specific)
A subclass of PerLigandError for MD simulations. MD poses are treated like ligands - when a pose fails, the pipeline skips it and continues with remaining poses. Common causes:
- GROMACS equilibration failures (NVT, NPT)
- Production MD simulation crashes
- Topology errors for specific poses

**Handling**: Log warning, skip pose, continue with other poses.

### FatalError
Fatal errors that should stop the entire pipeline. These are typically caused by:
- Missing required input files (receptor, ligand CSV)
- Invalid configuration parameters
- Missing external executables
- System resource exhaustion

**Handling**: Log error, stop pipeline, report to user.

## SIGINT (Ctrl+C) Handling

The pipeline uses a global SIGINT handler that provides immediate, clean shutdown and returns control to the user.

### How It Works

1. **Signal Handler**: When Ctrl+C is pressed, the handler:
   - Sets a global `_sigint_received` flag
   - Logs "SIGINT (Ctrl+C) received - shutting down gracefully..."
   - Raises `KeyboardInterrupt` to break out of blocking operations

2. **Main Process**: Catches `KeyboardInterrupt`, terminates workers, and returns control to user

3. **Workers**: Check `is_sigint_pending()` at iteration boundaries and exit early if detected

### MD-Specific SIGINT Behavior
MD simulations have multiple phases (NVT, NPT, production MD). The pipeline checks for SIGINT:
- Before starting each pose
- Before each MD phase (NVT, NPT, production)
- This allows quick shutdown even during long-running simulations

### Setup
```python
from adams.error_handling import setup_sigint_handler

# In main process __init__:
setup_sigint_handler()
```

### Main Process Handling
```python
try:
    pool = Pool(processes=num_workers)
    pool.starmap(worker_function, tasks)
    pool.close()
    pool.join()
except KeyboardInterrupt:
    logger.info("Process interrupted by user (Ctrl+C)")
    if pool is not None:
        pool.terminate()
        pool.join()
    logger.info("Workers terminated, returning control to user")
    return  # Return to allow user to continue or restart
```

### Usage in Workers
```python
from adams.error_handling import is_sigint_pending

# At the start of each iteration:
for item in items:
    if is_sigint_pending():
        logger.info("SIGINT detected, exiting")
        return
    process(item)
```

### Behavior
- **Immediate exit**: Workers check for SIGINT at the start of each iteration and exit immediately
- **No partial results saved**: Workers return without saving partial work
- **Clean shutdown**: All workers exit gracefully with code 0

## Safe Wrapper Functions

The `error_handling.py` module provides safe wrappers for external library calls:

### RDKit Wrappers

#### safe_mol_from_smiles
```python
from adams.error_handling import safe_mol_from_smiles

mol = safe_mol_from_smiles("CCO", "ethanol", add_hs=True)
# Returns: RDKit Mol object or raises RDKitMoleculeError
```

#### safe_generate_conformer
```python
from adams.error_handling import safe_generate_conformer

success = safe_generate_conformer(mol, "ethanol", max_attempts=50)
# Returns: True if successful, raises RDKitMoleculeError if failed
```

#### safe_read_and_generate_conformer
```python
from adams.error_handling import safe_read_and_generate_conformer

mol = safe_read_and_generate_conformer("CCO", "ethanol")
# Returns: RDKit Mol with 3D conformer or None if failed
```

### Vina Wrappers

#### safe_vina_operation (decorator)
```python
from adams.error_handling import safe_vina_operation

@safe_vina_operation("docking")
def dock_ligand(v, ligand_id, ligand_pdbqt):
    v.set_ligand_from_string(ligand_pdbqt)
    v.dock(exhaustiveness=32, n_poses=5)
    # If Vina calls sys.exit(), it's converted to VinaExecutionError
```

### Meeko Wrappers

#### safe_meeko_preparation
```python
from adams.error_handling import safe_meeko_preparation

result = safe_meeko_preparation(mol, "lig_001")
# Returns: (pdbqt_string, is_ok) or raises MeekoPreparationError
```

### Subprocess Wrappers

#### safe_subprocess_call
```python
from adams.error_handling import safe_subprocess_call

result = safe_subprocess_call(
    ["ls", "-l"],
    ligand_id="lig_001",  # Optional: for per-ligand operations
    operation_name="file listing"
)
# Raises: GPUExecutionError if ligand_id provided and subprocess fails
#         FatalError if ligand_id is None and subprocess fails
```

## Worker Error Handling Patterns

### CPU Docking Worker Pattern

```python
def _autodock_worker(self, batch, worker_id, log_queue):
    # Configure logging
    configure_worker_logging(log_queue)
    logger = get_logger()
    
    # Initialize resources
    v = Vina(...)
    ligprep = MoleculePreparation()
    
    # Process each ligand
    for idx, (lig_idx, grid_idx) in enumerate(batch):
        # Check for SIGINT at start of each iteration
        if is_sigint_pending():
            logger.info(f"Worker {worker_id}: SIGINT detected, exiting")
            return
        
        # Per-ligand error handling
        try:
            self._dock_vina(v, ligprep, lig_idx, lig, grid_idx, center)
            completed += 1
        except (VinaExecutionError, PerLigandError) as e:
            logger.warning(f"Worker {worker_id}: Skipping ligand {lig_idx}: {e}")
            failed += 1
            continue
        except SystemExit as e:
            # Vina called exit() - convert to per-ligand error
            logger.error(f"Worker {worker_id}: Vina exit({e.code}) for ligand {lig_idx}")
            failed += 1
            continue
        except Exception as e:
            # Unexpected error - log and skip
            logger.error(f"Worker {worker_id}: Unexpected error for ligand {lig_idx}: {e}")
            failed += 1
            continue
```

### GPU Docking Worker Pattern

```python
def _dock_pocket_chunk_worker(self, prepared_ligand_dir, ligand_indices, site_id, chunk_id, center, gpu_id, log_queue):
    # Configure logging
    configure_worker_logging(log_queue)
    logger = get_logger()
    
    # Check for SIGINT before starting work
    if is_sigint_pending():
        logger.info(f"GPU worker (site {site_id}, chunk {chunk_id}): SIGINT detected, exiting")
        return
    
    try:
        # Run GPU docking
        self._run_gpu_docking(ligand_dir, output_dir, center, box_size, pocket_idx, gpu_id)
    except subprocess.CalledProcessError as e:
        logger.error(f"GPU docking failed for pocket {pocket_idx} with return code {e.returncode}")
        # For batch processing, log error but don't crash worker
        raise
```

### MD Worker Pattern

```python
def _gro_run(self, pose_name):
    """Run MD simulation for a single pose with error recovery."""
    # Configure logging
    configure_worker_logging(log_queue)
    logger = get_logger()
    
    # Check for SIGINT before starting work
    if is_sigint_pending():
        logger.info(f"MD worker: SIGINT detected for pose {pose_name}, exiting")
        return False
    
    try:
        # Check before each phase
        if is_sigint_pending():
            return False
        
        # NVT equilibration
        _launch_gro(gmx_binary, ...)
        
        if is_sigint_pending():
            return False
        
        # NPT equilibration
        _launch_gro(gmx_binary, ...)
        
        if is_sigint_pending():
            return False
        
        # Production MD
        _launch_gro(gmx_binary, ...)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"MD failed for pose {pose_name} (exit code {e.returncode})")
        return False
    except Exception as e:
        logger.error(f"Unexpected error for pose {pose_name}: {e}")
        return False
```

## Error Handling Best Practices

### 1. Handle Errors at the Lowest Appropriate Level

**Good**: Per-ligand errors handled in the docking loop
```python
for ligand in ligands:
    try:
        dock(ligand)
    except PerLigandError as e:
        logger.warning(f"Skipping ligand: {e}")
        continue
```

**Bad**: Wrapping entire worker in try-except
```python
try:
    for ligand in ligands:
        dock(ligand)
except Exception as e:
    logger.error(f"Worker failed: {e}")
    return  # Loses all work
```

### 2. Use Specific Error Types

**Good**: Specific error types for different failures
```python
try:
    mol = safe_mol_from_smiles(smiles, ligand_id)
except RDKitMoleculeError as e:
    logger.warning(f"SMILES parsing failed: {e}")
    return None
```

**Bad**: Catching generic exceptions
```python
try:
    mol = MolFromSmiles(smiles)
except Exception as e:
    logger.warning(f"Something failed: {e}")
    return None
```

### 3. Check for SIGINT Regularly

**Good**: Check at start of each iteration
```python
for item in items:
    if is_sigint_pending():
        logger.info("SIGINT detected, exiting")
        return
    process(item)
```

**Bad**: No SIGINT checks (worker hangs on Ctrl+C)
```python
for item in items:
    process(item)
```

### 4. Log Appropriately

**Good**: Different log levels for different error types
```python
try:
    dock(ligand)
except PerLigandError as e:
    logger.warning(f"Skipping ligand: {e}")  # Expected, use warning
except Exception as e:
    logger.error(f"Unexpected error: {e}")  # Unexpected, use error
```

**Bad**: Everything logged as error
```python
try:
    dock(ligand)
except Exception as e:
    logger.error(f"Error: {e}")  # Too noisy
```

## Wrapping New External Calls

When adding new external library calls, follow these patterns:

### 1. Identify Error Type
- **Per-ligand operation?** → Use PerLigandError subclass
- **Fatal operation?** → Use FatalError subclass

### 2. Create Wrapper Function
```python
def safe_external_operation(input_data, ligand_id: str):
    """
    Safe wrapper for external library call.
    
    Args:
        input_data: Input for the operation
        ligand_id: Identifier for the ligand
        
    Returns:
        Result or None
        
    Raises:
        PerLigandError: If operation fails for this ligand
    """
    logger = get_logger()
    
    try:
        result = external_library.operation(input_data)
        return result
    except ExternalLibraryError as e:
        logger.warning(f"External operation failed for {ligand_id}: {e}")
        raise PerLigandError(ligand_id, f"Operation failed: {str(e)}", cause=e)
```

### 3. Use Wrapper in Workers
```python
for ligand in ligands:
    try:
        result = safe_external_operation(ligand.data, ligand.id)
        completed += 1
    except PerLigandError as e:
        logger.warning(f"Skipping ligand: {e}")
        failed += 1
        continue
```

## Error Report Format

### Required Information
Always include these fields in error reports:

1. **Output folder**: The exact path being used (e.g., "Output folder: /path/to/outputs/run_xxx")
2. **Steps completed**: Which pipeline steps finished successfully before the error
3. **Step that failed**: Which step encountered the error
4. **Error details**: The actual error message and likely cause
5. **Entry point for resume**: Which entry point to use if resuming

### Error Report Template
```
ERROR OCCURRED during [step_name]

Run Context:
- Output folder: /full/path/to/output_folder
- Log file: /full/path/to/adams_pipeline.log
- Steps completed: preprocessing, docking
- Step failed: md_analysis

Error Details:
[error message]

Likely Cause:
[brief explanation]

To Resume:
- Entry point: md_protein_topology (or appropriate entry point)
- Required files: [list paths that will be needed]
```

## Common Error Scenarios

### Scenario 1: Invalid SMILES
**Error Type**: RDKitMoleculeError (PerLigandError)
**Handling**: Skip ligand, continue with next
**Example**:
```
Worker 0: Skipping ligand 42 (ID: CHEMBL123): RDKit could not parse SMILES: invalid_smiles
```

### Scenario 2: Vina Missing Affinity Map
**Error Type**: VinaExecutionError (PerLigandError)
**Handling**: Skip ligand, continue with next
**Example**:
```
Worker 0: Vina exit(1) for ligand 42 (ID: CHEMBL123), grid 0. Likely missing affinity map for unusual atom type.
```

### Scenario 3: GPU Out of Memory
**Error Type**: GPUExecutionError (PerLigandError) or FatalError
**Handling**: Depends on context - may skip batch or stop pipeline
**Example**:
```
GPU docking failed for pocket 0 with return code 1
GPU stderr: CUDA out of memory
```

### Scenario 4: Missing Receptor File
**Error Type**: FatalError
**Handling**: Stop pipeline, report to user
**Example**:
```
ERROR: File not found during docking
Error Type: FileNotFoundError
Error Message: [Errno 2] No such file or directory: '/path/to/receptor.pdb'
```

### Scenario 5: User Interrupt (Ctrl+C)
**Error Type**: SIGINT
**Handling**: Immediate clean exit, no partial results saved
**Example**:
```
Worker 0: SIGINT detected, exiting immediately
SIGINT (Ctrl+C) received - shutting down gracefully...
```

## Testing Error Handling

After implementing error handling, test these scenarios:

1. **Normal execution**: Ensure no regressions
2. **SIGINT (Ctrl+C)**: Press Ctrl+C during docking, verify immediate exit
3. **Invalid SMILES**: Include invalid SMILES in input, verify per-ligand recovery
4. **Missing files**: Remove receptor file, verify fatal error and clear message
5. **Verify logs**: Check that appropriate warnings/errors are logged

## Success Criteria

✅ Single source of truth for error classification  
✅ No redundant try-except blocks  
✅ SIGINT causes immediate exit in all workers  
✅ External code failures skip only affected ligand  
✅ Consistent error logging across modules  
✅ Workers never crash - always exit gracefully with code 0 or log error and continue
