# Docking backends

Backends plug into the unified pipeline via `DockingPipeline(backend="vina"|"vina_gpu"|"unidock", **kwargs)` and are registered in this package.

## Architecture

All backends share a single parallelism model defined in `BaseDockingPipeline`:

1. `_build_docking_jobs()` — create a list of docking jobs.
2. `_run_docking_jobs_parallel(jobs)` — dispatch jobs to `_get_worker_count()` worker processes, each pulling from a shared queue.
3. `_execute_docking_job(job, worker_id)` — process one docking job in a worker.

**GPU backends** inherit from `BaseGPUBackend` (in `base_gpu.py`), which implements these three methods for the common GPU flow (chunking, symlinks, box sizing, output organization, cleanup). GPU backends only implement `_run_gpu_docking(...)` to provide their specific executable and CLI arguments.

```
BaseDockingPipeline          (docking.py)
├── VinaBackend              (vina.py)        — CPU, per-ligand Vina API
└── BaseGPUBackend           (base_gpu.py)    — shared GPU chunking + job flow
    ├── VinaGPUBackend       (vina_gpu.py)    — AutoDock-Vina-GPU-2-1 binary
    └── UniDockBackend       (unidock.py)     — UniDock binary
```

## Adding a new backend

### CPU backend

1. **Subclass `BaseDockingPipeline`** (from `..docking`).
2. **Constructor:** call `super().__init__(...)` with the common parameters (`input_data`, `receptor`, `complex`, `mode`, `num_pockets`, `num_poses`, `docking_centers`, `docking_centers_file`, `minimized_dock`, `search_gridsize`, `search_margin`, `out_folder`, `pH`). Add backend-specific parameters.
3. **Implement the docking job API:**
   - `_get_worker_count()` — max concurrent workers (e.g. CPU core count).
   - `_build_docking_jobs()` — return a list of docking jobs. Each job is passed to one `_execute_docking_job` call.
   - `_execute_docking_job(job, worker_id)` — run one docking job in a worker process. Logging is already configured.
4. **Pose output:** write PDBQT files to `self.dir_structure["poses"]` with naming:
   - Search mode: `ligand_{lig_id}_grid_{grid_id}_docked.pdbqt`
   - Production mode: `ligand_{lig_id}_pocket_{pocket_id}_docked.pdbqt`

### GPU backend

1. **Subclass `BaseGPUBackend`** (from `.base_gpu`).
2. **Constructor:** call `super().__init__(...)` with common parameters (`input_data`, `receptor`, `complex`, `mode`, `num_pockets`, `num_poses`, `docking_centers`, `docking_centers_file`, `minimized_dock`, `search_gridsize`, `search_margin`, `out_folder`, `pH`) + `num_gpus`, `gpu_ids`. Add backend-specific parameters (executable path, CLI options, etc.).
3. **Implement `_run_gpu_docking(ligand_dir, output_dir, center, box_size, pocket_idx, gpu_id)`:**
   - Build the CLI command and environment (`CUDA_VISIBLE_DEVICES`).
   - Call `self._run_gpu_subprocess(cmd, env, cwd, context_str)` for standardized execution and error handling.
4. `_build_docking_jobs`, `_execute_docking_job`, `_get_worker_count`, `_cleanup_after_run`, and output organization are all inherited from `BaseGPUBackend`.

### Register and document

- In `backends/__init__.py`: import the class, add to `BACKEND_REGISTRY` and `__all__`.
- Update `DockingPipeline` docstring in `docking.py` and the "Choose from: ..." error message.
- In `docking_agent.py`: add parameters to `run_docking` signature, kwargs dict, and docstring.
- In `docking_agent_prompt.md`: add the backend to the engines table and parameter sections.

## Reference implementations

- **CPU:** `vina.py` — per-ligand Python Vina API.
- **GPU:** `vina_gpu.py` (AutoDock-Vina-GPU-2-1 binary) and `unidock.py` (UniDock binary), both via `BaseGPUBackend`.
