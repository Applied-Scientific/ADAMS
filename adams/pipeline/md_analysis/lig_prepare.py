"""
Ligand Preparation Module - MD Analysis Pipeline Step 2

This module processes docking results, selects top ligands, and prepares them
for MD simulation by combining with protein, solvating, adding ions, and
performing energy minimization.

POSITION IN PIPELINE:
    Step 2 of 4: run_lig_prepare
    - Requires outputs from run_protein_topology (protein_gro, protein_top)
    - Must be executed before run_gro
    - Can be skipped if prepared poses already exist

INPUTS (from file_paths dictionary):
    - docking_csv: Path to docking results CSV file
    - ligand_input: Ligand structure input (SMILES string, CSV, SDF, MOL2, or directory)
    - protein_gro: Path to protein GRO file (from run_protein_topology)
    - protein_top: Path to protein topology file (from run_protein_topology)
    - poses_dir: Directory to store prepared poses
    - gromacs_path: Path to GROMACS installation
    - ambertools_path: Path to AmberTools installation

OUTPUTS (added to file_paths dictionary):
    - poses_dir: Updated with prepared pose subdirectories (each containing min.gro)

KEY FUNCTIONALITY:
    - Selects top N ligands per grid from docking results
    - Generates ligand topology using ACPYPE (from MOL2 files)
    - Combines protein and ligand into complex
    - Solvates system with water molecules
    - Adds ions to neutralize and set ionic strength
    - Performs energy minimization
    - Creates index files for restraints

EXTERNAL COMMANDS:
    - acpype: Generate GROMACS-compatible ligand topology from MOL2
    - gmx editconf: Create simulation box around complex
    - gmx solvate: Add water molecules to the system
    - gmx grompp: Prepare TPR files for ion addition and minimization
    - gmx genion: Add ions to neutralize and set ionic strength
    - gmx make_ndx: Create index groups for restraints
    - gmx genrestr: Generate position restraints for ligand
    - gmx mdrun: Run energy minimization
"""

import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from ...common_utils import get_cpu_count, get_gpu_count
from ...error_handling import is_sigint_pending, setup_sigint_handler
from ...logger_utils import (
    get_logger,
    log_step_execution,
    setup_multiprocessing_logging,
)
from ...utils import run_cmd
from ...utils.multiprocessing_utils import (
    Process,
    cleanup_process,
    configure_worker_logging,
)
from .ligand_resolver import LigandResolver
from .utils import (
    add_ligand_topology_with_atomtypes,
    clean_gro_file,
    combine_gro,
    extract_pose_from_pdbqt,
    formal_charge,
    get_gromacs_binary,
    get_mdp_dir,
    restore_from_pdbqt,
)


class LigPrepare:
    def __init__(
        self,
        file_paths,
        tops: int = 50,
        charge_type: str = "bcc",
        num_cores: int = None,
        num_gpus: int = 1,
        water_margin: float = 1.0,
        ion_conc: float = 0.15,
        pname: str = "K",
        nname: str = "CL",
    ):
        r"""
        Args:
            file_paths: dict: File paths dictionary - single source of truth (required).
                Must include:
                - docking_csv: Path to docking results CSV file
                - ligand_input: Ligand structure input (SMILES string, CSV, SDF, MOL2, or directory)
                - protein_gro: Path to protein GRO file
                - protein_top: Path to protein topology file
                - poses_dir: Directory to store prepared poses
                - gromacs_path: Path to GROMACS installation
                - ambertools_path: Path to AmberTools installation
                - gromacs_binary_type: Type of GROMACS binary (from discover_paths)
            tops: int: Number of top ligands per grid (default: 50).
            charge_type: str: Charge type of Antechamber partial charge (default: bcc) fallback to 'gas'.
            num_cores: int: Number of CPU cores (None uses all-1).
            num_gpus: int: Number of GPUs for energy minimization (default: 1).
            water_margin: float: Water box margin in nm (default: 1.0 nm).
            ion_conc: float: Ion concentration in mol/L (default: 0.15 M).
            pname: str: Cation name (default: K).
            nname: str: Anion name (default: CL).
        """
        self.logger = get_logger()

        # Set up multiprocessing logging queue (spawn-safe, eliminates fork deadlocks)
        # Note: Store queue for Process() calls (inherited, not pickled)
        self.log_queue = setup_multiprocessing_logging()

        # Set up SIGINT handler for clean shutdown on Ctrl+C
        setup_sigint_handler()

        if file_paths is None:
            raise ValueError(
                "file_paths dictionary is required. Use build_file_paths() and discover_paths() first."
            )
        self.file_paths = file_paths

        self.validate_files()

        self.docking_csv = file_paths["docking_csv"]
        self.gromacs_path = file_paths["gromacs_path"]
        self.ambertools_path = file_paths["ambertools_path"]
        self.gromacs_binary_type = file_paths.get("gromacs_binary_type", "standard")
        self.gmx_binary = get_gromacs_binary(
            self.gromacs_path, binary_type=self.gromacs_binary_type, require_mpi=False
        )

        self.md_workdir = file_paths.get("md_root", ".")

        # Resolve ligand structures using LigandResolver
        self.ligand_resolver = LigandResolver()
        ligand_output_dir = os.path.join(self.md_workdir, "ligand_resolution")

        resolution_result = self.ligand_resolver.resolve_ligand_structures(
            ligand_input=file_paths["ligand_input"],
            docking_csv=file_paths["docking_csv"],
            output_dir=ligand_output_dir,
        )

        # Store resolved SMILES CSV path (for restore_from_pdbqt)
        self.smiles_file = resolution_result["smiles_csv_path"]
        self.logger.info(
            f"Resolved ligand structures from {resolution_result['source']}: "
            f"{self.smiles_file}"
        )

        self.tops = tops
        self.charge_type = charge_type
        self.num_cores = num_cores if num_cores is not None else get_cpu_count()

        if self.gromacs_binary_type == "cuda":
            if num_gpus == 0:
                self.num_gpus = get_gpu_count()
                self.logger.info(
                    f"Auto-detected {self.num_gpus} GPUs for energy minimization."
                )
            else:
                self.num_gpus = num_gpus

            if self.num_gpus == 0:
                self.logger.warning(
                    "gromacs_binary_type is 'cuda' but no GPUs detected. Energy minimization will run on CPU."
                )
                self.gromacs_binary_type = "standard"  # Fallback to standard
        else:
            self.num_gpus = 0
        self.water_margin = water_margin
        self.ion_conc = ion_conc
        self.pname = pname
        self.nname = nname

        self.root_path = os.getcwd()

    def validate_files(self):
        """
        Validate required keys exist in file_paths.

        Checks that:
        - Required keys exist in file_paths (docking_csv, ligand_input, protein_gro, protein_top, poses_dir, gromacs_path, ambertools_path)

        Raises:
            ValueError: If required paths are missing from file_paths
        """
        required_keys = [
            "docking_csv",
            "ligand_input",
            "protein_gro",
            "protein_top",
            "poses_dir",
            "gromacs_path",
            "ambertools_path",
        ]
        missing = [k for k in required_keys if not self.file_paths.get(k)]

        if missing:
            raise ValueError(
                f"Required paths missing from file_paths: {missing}\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Ensure previous pipeline steps have run or provide explicit paths."
            )

        poses_dir = self.file_paths["poses_dir"]
        os.makedirs(poses_dir, exist_ok=True)

    def run(self) -> dict:
        """
        Run ligand preparation workflow.

        External commands called:
            - acpype: Generate GROMACS-compatible ligand topology from MOL2
            - gmx editconf: Create simulation box around complex
            - gmx solvate: Add water molecules to the system
            - gmx grompp: Prepare TPR files for ion addition and minimization
            - gmx genion: Add ions to neutralize and set ionic strength
            - gmx make_ndx: Create index groups for restraints
            - gmx genrestr: Generate position restraints for ligand
            - gmx mdrun: Run energy minimization

        Returns:
            dict: Updated file_paths dictionary with prepared_poses list
        """
        step_logger = log_step_execution("Ligand Preparation", self.logger)
        with step_logger:
            with step_logger.timing("prepwork"):
                self._prepwork()
            with step_logger.timing("lig_prepare_batch"):
                self._run_lig_prepare_batch()

            # poses_dir is already in file_paths, no need to update
            return self.file_paths

    def _prepwork(self):
        """
        Prepare ligand poses from docking results.

        Reads docking CSV directly from file_paths["docking_csv"] - no file discovery.
        PDBQT paths are resolved relative to the CSV location.
        """

        # Get paths from file_paths dict
        docking_csv = self.file_paths["docking_csv"]
        # smiles_file is now set in __init__ via ligand_resolver
        output_poses_dir = self.file_paths["poses_dir"]  # Where to write prepared poses

        # Create top by grid directory in the MD root
        md_root = self.file_paths.get("md_root", os.path.dirname(output_poses_dir))
        top_by_grid_dir = os.path.join(md_root, f"top{self.tops}_by_grid")
        os.makedirs(top_by_grid_dir, exist_ok=True)
        self.logger.info(f"Top by grid directory: {top_by_grid_dir}")
        self.logger.info(f"Using docking CSV from file_paths: {docking_csv}")

        # Read docking results directly from provided CSV path
        df = pd.read_csv(docking_csv)
        required_cols = [
            "ligand_id",
            "ID",
            "grid_id",
            "pose_id",
            "affinity",
            "COM_x",
            "COM_y",
            "COM_z",
            "MolWt",
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Docking CSV missing required columns: {missing_cols}")

        df = df[required_cols]
        # Track the CSV directory for PDBQT path resolution
        csv_dir = os.path.dirname(docking_csv)
        df["base_dir"] = csv_dir

        # Sort by affinity and process top N per grid_id
        df = df.sort_values(by="affinity", ascending=True)
        task_list = []

        for grid_id, group in df.groupby("grid_id"):
            top_df = group.head(self.tops).reset_index(drop=True)
            out_file = os.path.join(
                top_by_grid_dir, f"grid_{grid_id}_top{self.tops}.csv"
            )
            top_df.to_csv(out_file, index=False)

            # Loop over top poses
            for rank, row in top_df.iterrows():
                ligand_id = row["ligand_id"]
                grid_id = row["grid_id"]
                pose_id = int(row["pose_id"])
                ligand_name = row["ID"]
                base_dir = row["base_dir"]

                # Resolve PDBQT path relative to CSV location
                # If CSV is in summaries/, look in sibling poses/ directory
                if "summaries" in base_dir:
                    docking_poses_dir = base_dir.replace("summaries", "poses")
                    pdbqt_path = os.path.join(
                        docking_poses_dir,
                        f"ligand_{ligand_id}_pocket_{grid_id}_docked.pdbqt",
                    )
                    # Also try search poses if production doesn't exist
                    if not os.path.exists(pdbqt_path):
                        search_poses_dir = docking_poses_dir.replace(
                            "production", "search"
                        )
                        pdbqt_path = os.path.join(
                            search_poses_dir,
                            f"ligand_{ligand_id}_grid_{grid_id}_docked.pdbqt",
                        )
                else:
                    # Direct path from CSV directory
                    pdbqt_path = os.path.join(
                        base_dir, f"ligand_{ligand_id}_pocket_{grid_id}_docked.pdbqt"
                    )

                # Check if PDBQT file exists before processing
                if not os.path.exists(pdbqt_path):
                    self.logger.warning(
                        f"PDBQT file not found for ligand {ligand_name}, pocket {grid_id}: {pdbqt_path}"
                    )
                    self.logger.warning(f"  Skipping this pose.")
                    continue

                # Create pose output directory in the output_poses_dir (from file_paths)
                out_dir = os.path.join(
                    output_poses_dir, f"{ligand_name}_pocket_{grid_id}_top{rank+1}"
                )
                os.makedirs(out_dir, exist_ok=True)

                # Extract pose from PDBQT
                pose_pdbqt = f"{out_dir}/ligand.pdbqt"
                extract_pose_from_pdbqt(pdbqt_path, pose_pdbqt, pose_id)

                task_list.append(f"{ligand_name}_pocket_{grid_id}_top{rank+1}")

        self.logger.debug(f"Task list: {task_list}")
        self.pose_list = task_list

    def _run_lig_prepare_batch(self):
        """
        Run ACPYPE for a list of ligand tasks in parallel, batching by number of CPUs.

        tasks_list: list of strings in format "{ligand_name}_pocket_{grid_id}"
        source_folder: folder containing pdbqt/mol2 files
        numCores: number of parallel processes
        charge_type: charge method for ACPYPE
        """
        # Split tasks into batches
        task_batches = np.array_split(self.pose_list, self.num_cores)
        self.logger.debug(f"Task batches: {task_batches}")

        processes = []
        try:
            for batch_idx, batch in enumerate(task_batches):
                p = Process(
                    target=self._run_single_acpype,
                    args=(batch, batch_idx, self.log_queue),
                )
                processes.append(p)
                p.start()

            # Wait for all processes to complete
            for p in processes:
                p.join()

        except KeyboardInterrupt:
            self.logger.info("Ligand preparation interrupted by user (Ctrl+C)")
            # Terminate all worker processes
            for p in processes:
                if p.is_alive():
                    p.terminate()
            # Wait for all to finish with timeout
            for p in processes:
                p.join(timeout=2)
                if p.is_alive():
                    # Force kill if still alive
                    p.kill()
                    p.join()
            self.logger.info(
                "Ligand preparation workers terminated, returning control to user"
            )
            raise  # Re-raise to propagate

        except Exception as e:
            self.logger.error(f"Error in _run_lig_prepare_batch: {e}")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            raise e
        finally:
            # Explicit cleanup to prevent semaphore leaks
            for p in processes:
                cleanup_process(p, timeout=1.0)

        self.logger.info("Ligand preparation completed successfully.")

    def _run_single_acpype(self, task_name, worker_id, log_queue):
        """Run ACPYPE for a batch of poses with queue logging and SIGINT checking."""
        # Configure worker logging
        configure_worker_logging(log_queue)
        logger = get_logger()
        logger.info(f"LigPrep worker {worker_id}: Processing {len(task_name)} poses")

        # Convert paths to absolute for worker processes (workers may have different cwd)
        poses_dir = os.path.abspath(self.file_paths["poses_dir"])
        smiles_file = os.path.abspath(self.smiles_file)
        for pose in task_name:
            # Check for SIGINT before processing each pose
            if is_sigint_pending():
                logger.info(f"LigPrep worker {worker_id}: SIGINT detected, exiting")
                return

            try:
                ligand_name, _, grid_rank = pose.partition("_pocket_")
                grid_id, _, rank = grid_rank.partition("_top")
                # Use absolute poses_dir from file_paths
                out_dir = os.path.join(
                    poses_dir, f"{ligand_name}_pocket_{grid_id}_top{rank}"
                )
                pose_pdbqt = f"{out_dir}/ligand.pdbqt"
                pose_mol2 = f"{out_dir}/ligand.mol2"
                restore_from_pdbqt(smiles_file, ligand_name, pose_pdbqt, pose_mol2)
                self._run_acpype(pose_mol2, "LIG", charge_type=self.charge_type)
                self._initialize_gro_run(out_dir)
            except Exception as e:
                logger.error(
                    f"LigPrep worker {worker_id}: Error processing pose {pose}: {e}"
                )
                # Continue with next pose - don't let one failure stop all poses
                continue

    def _run_acpype(self, mol2_file, resname, charge_type="bcc"):
        """
        amber_env = {}
        with open("amber_env.txt") as f:
            for line in f:
                key, _, val = line.strip().partition("=")
                amber_env[key] = val

        # Merge with current env
        env = os.environ.copy()
        env.update(amber_env)
        """
        net_charge = formal_charge(mol2_file)
        self.logger.debug(f"The net charge from file {mol2_file} is {net_charge}")

        workdir = os.path.dirname(mol2_file)
        self.logger.debug(f"Working directory: {workdir}")

        # Find acpype executable
        acpype_cmd = shutil.which("acpype")
        if acpype_cmd is None:
            # Try to find it via AMBERHOME
            amber_home = os.environ.get("AMBERHOME")
            if amber_home:
                acpype_cmd = shutil.which(
                    "acpype", path=os.path.join(amber_home, "bin")
                )

            # If still not found, try conda environment
            if acpype_cmd is None:
                conda_prefix = os.environ.get("CONDA_PREFIX")
                if conda_prefix:
                    # AmberTools is at $CONDA_PREFIX (amber.sh at $CONDA_PREFIX/amber.sh)
                    # acpype should be in $CONDA_PREFIX/bin
                    potential_path = os.path.join(conda_prefix, "bin", "acpype")
                    if os.path.exists(potential_path):
                        acpype_cmd = potential_path
                # Also try direct PATH search with conda bin
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

        # Construct the ACPYPE command
        cmd_bcc = [
            acpype_cmd,
            "-i",
            mol2_file,
            "-b",
            resname,
            "-c",
            charge_type,
            "-n",
            str(net_charge),
        ]

        # Run the command
        try:
            run_cmd(cmd_bcc, cwd=workdir, check=True)
            self.logger.debug("ACPYPE bcc completed successfully.")
        except subprocess.CalledProcessError:
            self.logger.warning(
                f"ACPYPE bcc failed for {mol2_file}, retrying with gas charges"
            )
            cmd_gas = [
                acpype_cmd,
                "-i",
                mol2_file,
                "-b",
                resname,
                "-c",
                "gas",
                "-n",
                str(net_charge),
            ]
            run_cmd(cmd_gas, cwd=workdir, check=True)
            self.logger.info(f"ACPYPE gas succeeded for {mol2_file}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"acpype executable not found at {acpype_cmd}. "
                "Please check your installation and PATH configuration."
            ) from e

    def _initialize_gro_run(self, pose_name):
        gmx_machine = self.gmx_binary

        ### prepare complex.gro
        protein_gro = self.file_paths["protein_gro"]
        combine_gro(
            protein_gro,
            f"{pose_name}/LIG.acpype/LIG_GMX.gro",
            f"{pose_name}/complex.gro",
        )

        ### prepare system.top
        protein_top = self.file_paths["protein_top"]
        add_ligand_topology_with_atomtypes(
            protein_top,
            f"{pose_name}/LIG.acpype/LIG_GMX.itp",
            f"{pose_name}/system.top",
        )

        ### solvation with waters
        editconf_cmd = [
            gmx_machine,
            "editconf",
            "-f",
            f"{pose_name}/complex.gro",
            "-o",
            f"{pose_name}/newbox.gro",
            "-bt",
            "cubic",
            "-d",
            str(self.water_margin),
        ]
        run_cmd(editconf_cmd, check=True)

        solvate_cmd = [
            gmx_machine,
            "solvate",
            "-cp",
            f"{pose_name}/newbox.gro",
            "-cs",
            "spc216.gro",
            "-p",
            f"{pose_name}/system.top",
            "-o",
            f"{pose_name}/solv.gro",
        ]
        run_cmd(solvate_cmd, check=True)

        ### add ions
        mdp_dir = get_mdp_dir()
        ions_mdp_path = os.path.join(mdp_dir, "ions.mdp")

        # posre.itp is created by pdb2gmx in the same directory as the protein topology
        # Priority: 1) explicit posre_itp in file_paths, 2) derive from protein_top, 3) protein_dir, 4) cwd
        posre_itp_path = self.file_paths.get("posre_itp")
        if not posre_itp_path or not os.path.exists(posre_itp_path):
            if protein_top:
                protein_dir = os.path.dirname(protein_top)
                posre_itp_path = os.path.join(protein_dir, "posre.itp")
            else:
                protein_dir = self.file_paths.get("protein_dir")
                if protein_dir:
                    posre_itp_path = os.path.join(protein_dir, "posre.itp")
                else:
                    posre_itp_path = os.path.join(self.root_path, "posre.itp")

        if not os.path.exists(ions_mdp_path):
            raise FileNotFoundError(
                f"Required MDP file not found: {ions_mdp_path}\n"
                f"Please ensure ions.mdp exists in {mdp_dir}"
            )
        if not os.path.exists(posre_itp_path):
            raise FileNotFoundError(
                f"Required ITP file not found: {posre_itp_path}\n"
                f"posre.itp is created by pdb2gmx (ProteinTopology step) alongside the topology file.\n"
                f"Searched in: {os.path.dirname(posre_itp_path)}\n"
                f"If starting from LigPrepare step, ensure protein_top points to a directory containing posre.itp."
            )

        shutil.copy(ions_mdp_path, pose_name)
        shutil.copy(posre_itp_path, pose_name)

        grompp_cmd = [
            gmx_machine,
            "grompp",
            "-f",
            f"{pose_name}/ions.mdp",
            "-c",
            f"{pose_name}/solv.gro",
            "-p",
            f"{pose_name}/system.top",
            "-o",
            f"{pose_name}/ions.tpr",
            "-po",
            f"{pose_name}/mdout.mdp",
        ]
        run_cmd(grompp_cmd, check=True)

        # Use input_str parameter instead of shell pipe for reliable input handling
        # This ensures genion properly receives the group selection and updates the topology
        genion_cmd = [
            gmx_machine,
            "genion",
            "-s",
            f"{pose_name}/ions.tpr",
            "-o",
            f"{pose_name}/solv_ions.gro",
            "-p",
            f"{pose_name}/system.top",
            "-pname",
            self.pname,
            "-nname",
            self.nname,
            "-conc",
            str(self.ion_conc),
            "-neutral",
        ]
        run_cmd(genion_cmd, input_str="SOL\n", check=True)

        ### prepare to run LIG restraining
        make_ndx_lig_cmd = [
            gmx_machine,
            "make_ndx",
            "-f",
            f"{pose_name}/LIG.acpype/LIG_GMX.gro",
            "-o",
            f"{pose_name}/index_LIG.ndx",
        ]
        run_cmd(make_ndx_lig_cmd, input_str="0 & ! a H*\nq\n", check=True)

        ### prepare to nvt npt run Protein_UNL restraining
        make_ndx_system_cmd = [
            gmx_machine,
            "make_ndx",
            "-f",
            f"{pose_name}/solv_ions.gro",
            "-o",
            f"{pose_name}/index.ndx",
        ]
        run_cmd(
            make_ndx_system_cmd, input_str="1 | 13\nq\n", check=True
        )  # 1=Protein, 13=UNL

        genrestr_cmd = [
            gmx_machine,
            "genrestr",
            "-f",
            f"{pose_name}/LIG.acpype/LIG_GMX.gro",
            "-n",
            f"{pose_name}/index_LIG.ndx",
            "-o",
            f"{pose_name}/posre_LIG.itp",
            "-fc",
            "1000",
            "1000",
            "1000",
        ]
        run_cmd(genrestr_cmd, input_str="3\n", check=True)

        ### min
        original_dir = os.getcwd()
        try:
            os.chdir(pose_name)
            clean_gro_file()

            min_mdp_path = os.path.join(mdp_dir, "min.mdp")
            if not os.path.exists(min_mdp_path):
                raise FileNotFoundError(
                    f"Required MDP file not found: {min_mdp_path}\n"
                    f"Please ensure min.mdp exists in {mdp_dir}"
                )
            shutil.copy(min_mdp_path, "./")

            grompp_min_cmd = [
                gmx_machine,
                "grompp",
                "-f",
                "min.mdp",
                "-c",
                "solv_ions.gro",
                "-p",
                "system.top",
                "-o",
                "min.tpr",
                "-po",
                "mdout.mdp",
                "-maxwarn",
                "100",
            ]
            run_cmd(grompp_min_cmd, check=True)

            # Run minimization synchronously so min.gro exists before next stage
            # When using CUDA GROMACS, must specify both -ntmpi and -ntomp
            if self.gromacs_binary_type == "cuda":
                # For GPU: 1 MPI rank per GPU, distribute OpenMP threads across GPUs
                ntomp = max(1, self.num_cores // self.num_gpus)
                mdrun_cmd = [
                    gmx_machine,
                    "mdrun",
                    "-deffnm",
                    "min",
                    "-ntmpi",
                    str(self.num_gpus),
                    "-ntomp",
                    str(ntomp),
                ]
                self.logger.info(
                    f"Energy minimization: using {self.num_gpus} GPU(s) with {ntomp} OpenMP threads per GPU"
                )
            else:
                mdrun_cmd = [gmx_machine, "mdrun", "-deffnm", "min", "-ntomp", "1"]
            run_cmd(mdrun_cmd, check=True)
        finally:
            os.chdir(original_dir)
