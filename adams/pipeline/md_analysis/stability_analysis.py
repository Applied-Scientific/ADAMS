"""
Stability Analysis Module - MD Analysis Pipeline Step 4

This module analyzes MD trajectories for stability metrics including RMSD (Root
Mean Square Deviation) and RMSF (Root Mean Square Fluctuation). It generates
summary reports ranking ligands by stability.

POSITION IN PIPELINE:
    Step 4 of 4: run_stability_analysis
    - Requires outputs from run_gro (MD-completed poses with md.tpr/md.xtc)
    - Final step in the MD analysis pipeline
    - Can be run independently if MD simulations already completed

INPUTS (from file_paths dictionary):
    - poses_dir: Directory containing MD-completed pose subdirectories (with md.tpr/md.xtc)
    - reports_dir: Directory to write analysis reports
    - gromacs_path: Path to GROMACS installation
    - ambertools_path: Path to AmberTools installation

OUTPUTS (added to file_paths dictionary):
    - summary_report: Comprehensive analysis report (md_analysis_summary_{range}.csv)
    - brief_report: Brief report sorted by ligand RMSD (brief_report_{range}.csv)

KEY FUNCTIONALITY:
    - Calculates protein backbone RMSD (relative to initial structure)
    - Calculates ligand RMSD (relative to initial pose)
    - Calculates protein RMSF (per-residue fluctuations)
    - Supports analysis of all frames or last N frames
    - Merges with Vina docking affinity data (optional)
    - Ranks ligands by stability (lowest RMSD = most stable)

EXTERNAL COMMANDS:
    - gmx make_ndx: Create index groups (if missing)
    - gmx rms: Calculate RMSD for protein backbone and ligand
    - gmx rmsf: Calculate RMSF for protein residues

ANALYSIS OPTIONS:
    - Range='all': Analyze entire trajectory
    - Range='last': Analyze only last N frames (specified by last_frames parameter)
"""

import glob
import os

import pandas as pd

from ...logger_utils import get_logger, log_step_execution
from ...utils import run_cmd
from .utils import get_gromacs_binary, mean_std, parse_pose_name, parse_xvg


class StabilityAnalysis:
    def __init__(
        self,
        file_paths,
        prefix: str = "md",
        Range: str = "all",
        last_frames: int = 100,
        vina_report: str = None,
    ):
        r"""
        Args:
            file_paths: dict: File paths dictionary - single source of truth (required).
                Must include:
                - poses_dir: Directory containing MD-completed pose subdirectories
                - reports_dir: Directory to write analysis reports
                - gromacs_path: Path to GROMACS installation
                - ambertools_path: Path to AmberTools installation
                - gromacs_binary_type: Type of GROMACS binary (from discover_paths)
            prefix: Prefix for MD files (default: 'md')
            Range: Analysis range: 'all' or 'last' (default: 'all')
            last_frames: Number of last data points when Range='last' (default: 100)
            vina_report: Path to docking report for affinity merging (default: None)
        """
        self.logger = get_logger()

        if file_paths is None:
            raise ValueError(
                "file_paths dictionary is required. Use build_file_paths() and discover_paths() first."
            )
        self.file_paths = file_paths

        self.validate_files()

        self.gromacs_path = file_paths["gromacs_path"]
        self.ambertools_path = file_paths["ambertools_path"]
        self.gromacs_binary_type = file_paths.get("gromacs_binary_type", "standard")
        self.gmx_binary = get_gromacs_binary(
            self.gromacs_path, binary_type=self.gromacs_binary_type, require_mpi=False
        )

        self.md_workdir = file_paths.get("md_root", ".")

        self.prefix = prefix
        self.range = Range
        if self.range == "last":
            self.last_frames = last_frames
        else:
            self.last_frames = None

        self.vina_report = vina_report

    def validate_files(self):
        """
        Validate required keys exist in file_paths.

        Checks that:
        - poses_dir exists in file_paths
        - reports_dir exists in file_paths
        - gromacs_path exists in file_paths
        - ambertools_path exists in file_paths

        Raises:
            ValueError: If required paths are missing from file_paths
        """
        poses_dir = self.file_paths.get("poses_dir")
        if not poses_dir:
            raise ValueError(
                "poses_dir required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Ensure Gro step has run or provide explicit poses_dir path."
            )

        reports_dir = self.file_paths.get("reports_dir")
        if not reports_dir:
            raise ValueError(
                "reports_dir required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Use build_file_paths() with md_workdir to create directory structure."
            )

        if not self.file_paths.get("gromacs_path"):
            raise ValueError(
                "gromacs_path required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Use discover_paths() to discover GROMACS and AmberTools paths."
            )

        if not self.file_paths.get("ambertools_path"):
            raise ValueError(
                "ambertools_path required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Use discover_paths() to discover GROMACS and AmberTools paths."
            )

        os.makedirs(reports_dir, exist_ok=True)

    def run(self) -> dict:
        """
        Run stability analysis on MD-completed poses.

        External commands called:
            - gmx make_ndx: Create index groups (if missing)
            - gmx rms: Calculate RMSD for protein backbone and ligand
            - gmx rmsf: Calculate RMSF for protein residues

        Returns:
            dict: Updated file_paths dictionary with analysis report paths
        """
        step_logger = log_step_execution("Stability Analysis", self.logger)
        with step_logger:
            md_workdir = self.md_workdir
            analysis_range = self.range
            md_prefix = self.prefix

            # Files are already validated in __init__, just get poses_dir
            poses_dir = self.file_paths["poses_dir"]

            # Discover MD-completed poses within the directory
            # Look for directories containing md.tpr and md.xtc (or prefix.tpr and prefix.xtc)
            pose_dirs = []
            for name in sorted(os.listdir(poses_dir)):
                pose_path = os.path.join(poses_dir, name)
                if os.path.isdir(pose_path):
                    # Check if MD has completed (has md.tpr and md.xtc)
                    tpr = os.path.join(pose_path, f"{md_prefix}.tpr")
                    xtc = os.path.join(pose_path, f"{md_prefix}.xtc")
                    if os.path.exists(tpr) and os.path.exists(xtc):
                        pose_dirs.append(name)

            with step_logger.timing("trajectory_analysis"):
                summary = []
                for pose in pose_dirs:
                    rec = self._analyze_pose(poses_dir, pose, md_prefix, analysis_range)
                    if rec:
                        summary.append(rec)

                df = pd.DataFrame(summary)

                # Check if any poses were successfully analyzed
                if df.empty:
                    self.logger.error(
                        "No poses were successfully analyzed. Cannot generate stability report."
                    )
                    self.logger.error(
                        "This may be due to missing MD simulation files (.tpr, .xtc) in pose directories."
                    )
                    raise ValueError(
                        "No valid poses found for stability analysis. Ensure MD simulations completed successfully."
                    )

                # df = pd.read_csv(f"md_analysis_summary_{analysis_range.replace(' ', '_')}.csv")
                # summary_file = os.path.join(md_workdir, f"md_analysis_summary_{analysis_range.replace(' ', '_')}.csv")
                # ensure rank is numeric
                df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

                # sort by ligand_ID, g_id, and rank
                df = df.sort_values(by=["ID", "grid_id", "Rank"]).reset_index(drop=True)

                # Merge with vina_report if provided
                # Merge with vina_reports if provided
                if self.vina_report is not None:
                    vina_files = []
                    if os.path.isdir(self.vina_report):
                        # Find all grid_*_top50.csv files in the folder
                        vina_files = glob.glob(
                            os.path.join(self.vina_report, "grid_*_top*.csv")
                        )
                    else:
                        # Support comma-separated list
                        vina_files = self.vina_report.split(",")

                    if len(vina_files) == 0:
                        self.logger.warning(
                            f"No vina reports found in {self.vina_report}, skipping merge."
                        )
                    else:
                        vina_dfs = []
                        for vf in vina_files:
                            if os.path.exists(vf):
                                tmp = pd.read_csv(vf)
                                vina_dfs.append(tmp)
                            else:
                                self.logger.warning(
                                    f"Vina report {vf} not found, skipping."
                                )

                        if vina_dfs:
                            vina_df = pd.concat(vina_dfs, ignore_index=True)

                            # Ensure consistent merge columns
                            merge_cols = ["ID", "grid_id"]
                            if not set(merge_cols).issubset(df.columns):
                                self.logger.error(
                                    f"{merge_cols} missing from summary dataframe, cannot merge."
                                )
                            elif not set(merge_cols).issubset(vina_df.columns):
                                self.logger.error(
                                    f"{merge_cols} missing from vina dataframe, cannot merge."
                                )
                            else:
                                df = pd.merge(df, vina_df, on=merge_cols, how="left")

            with step_logger.timing("report_generation"):
                # Get reports directory from file_paths (already validated in __init__)
                reports_dir = self.file_paths["reports_dir"]

                summary_file = os.path.join(
                    reports_dir,
                    f"md_analysis_summary_{analysis_range.replace(' ', '_')}.csv",
                )

                df.to_csv(summary_file, index=False)
                self.logger.info(f"\nSummary written to {summary_file}")
                self.logger.debug(f"Summary dataframe:\n{df}")

                # Select and rename columns for brief report
                brief_cols = [
                    "ID",
                    "grid_id",
                    "affinity",
                    "Rank",
                    "Avg_Ligand_RMSD(nm)",
                    "Std_Ligand_RMSD(nm)",
                ]

                # Ensure your dataframe has matching columns
                for col in ["affinity", "Avg_Ligand_RMSD(nm)", "Std_Ligand_RMSD(nm)"]:
                    if col not in df.columns:
                        df[col] = None  # fill with None if missing

                brief_df = df[brief_cols].copy()

                brief_df = brief_df.sort_values(
                    by="Avg_Ligand_RMSD(nm)", ascending=True
                ).reset_index(drop=True)

                # Save brief report
                brief_file = os.path.join(
                    reports_dir, f"brief_report_{analysis_range.replace(' ', '_')}.csv"
                )
                brief_df.to_csv(brief_file, index=False)

                self.logger.info(f"Brief report written to {brief_file}")
                self.logger.debug(f"Brief report (top 10):\n{brief_df.head(10)}")

                # Record report paths in file_paths
                self.file_paths["summary_report"] = summary_file
                self.file_paths["brief_report"] = brief_file

            return self.file_paths

    def _analyze_pose(self, md_workdir, pose_name, md_prefix, analysis_range):
        self.logger.info(f"\n=== Processing {pose_name} ===")

        pose_path = os.path.join(md_workdir, pose_name)
        # Use md prefix for the main simulation files
        tpr = os.path.join(pose_path, f"{md_prefix}.tpr")
        xtc = os.path.join(pose_path, f"{md_prefix}.xtc")
        gro = os.path.join(pose_path, f"{md_prefix}.gro")
        ndx = os.path.join(pose_path, "index.ndx")
        lig_ndx = os.path.join(pose_path, "index_LIG.ndx")

        if not (os.path.exists(tpr) and os.path.exists(xtc)):
            self.logger.warning(
                f"Skipping {pose_name}, missing simulation files ({md_prefix}.tpr / {md_prefix}.xtc)"
            )
            return None

        # Parse pose name metadata
        lig_id, g_id, rank = parse_pose_name(pose_name)

        # Use appropriate GROMACS binary - analysis tools don't need MPI
        gmx_machine = self.gmx_binary

        # Create index file if missing
        if not os.path.exists(ndx):
            make_ndx_cmd = [gmx_machine, "make_ndx", "-f", gro, "-o", ndx]
            run_cmd(make_ndx_cmd, input_str="1 | 13\nq\n", check=True)

        # Filenames for outputs — prefix "md_" for clarity
        rmsd_protein_file = os.path.join(pose_path, f"{md_prefix}_protein_rmsd.xvg")
        rmsf_protein_file = os.path.join(pose_path, f"{md_prefix}_protein_rmsf.xvg")
        rmsd_ligand_file = os.path.join(pose_path, f"{md_prefix}_ligand_rmsd.xvg")
        hbnum_file = os.path.join(pose_path, f"{md_prefix}_hbnum.xvg")
        com_file = os.path.join(pose_path, f"{md_prefix}_com_dist.xvg")

        # Run GROMACS analysis commands
        # RMSD - protein backbone (group 4 = Backbone)
        rms_protein_cmd = [
            gmx_machine,
            "rms",
            "-s",
            tpr,
            "-f",
            xtc,
            "-o",
            rmsd_protein_file,
            "-tu",
            "ns",
        ]
        run_cmd(rms_protein_cmd, input_str="4 4\n", check=True)

        # RMSF - protein (group 4 = Backbone)
        rmsf_protein_cmd = [
            gmx_machine,
            "rmsf",
            "-s",
            tpr,
            "-f",
            xtc,
            "-o",
            rmsf_protein_file,
            "-res",
        ]
        run_cmd(rmsf_protein_cmd, input_str="4\n", check=True)

        # RMSD - ligand (group 3 = heavy atoms from index_LIG.ndx)
        rms_ligand_cmd = [
            gmx_machine,
            "rms",
            "-s",
            tpr,
            "-f",
            xtc,
            "-n",
            lig_ndx,
            "-o",
            rmsd_ligand_file,
            "-tu",
            "ns",
        ]
        run_cmd(rms_ligand_cmd, input_str="3 3\n", check=True)

        # Parse the results
        times_protein_rmsd, vals_protein_rmsd = parse_xvg(rmsd_protein_file)
        times_protein_rmsf, vals_protein_rmsf = parse_xvg(rmsf_protein_file)
        times_ligand_rmsd, vals_ligand_rmsd = parse_xvg(rmsd_ligand_file)
        # times_com, vals_com = parse_xvg(com_file)

        # Apply analysis_range (e.g. "all" or "last 1 ns")
        def select_range(values):
            if analysis_range == "all" or len(values) == 0:
                return values
            elif analysis_range == "last":
                n = min(self.last_frames, len(values))
                return values[-n:]
            else:
                return values

        vals_protein_rmsd = select_range(vals_protein_rmsd)
        vals_ligand_rmsd = select_range(vals_ligand_rmsd)
        vals_protein_rmsf = select_range(vals_protein_rmsf)
        # vals_com = select_range(vals_com)

        # Compute averages and standard deviations
        avg_protein_rmsd, std_protein_rmsd = mean_std(vals_protein_rmsd)
        avg_ligand_rmsd, std_ligand_rmsd = mean_std(vals_ligand_rmsd)
        avg_protein_rmsf, std_protein_rmsf = mean_std(vals_protein_rmsf)
        # avg_com, std_com = mean_std(vals_com)

        # Return the summary record
        return {
            "Pose": pose_name,
            "ID": lig_id,
            "grid_id": g_id,
            "Rank": rank,
            "Avg_Ligand_RMSD(nm)": avg_ligand_rmsd,
            "Std_Ligand_RMSD(nm)": std_ligand_rmsd,
            "Avg_Protein_RMSD(nm)": avg_protein_rmsd,
            "Std_Protein_RMSD(nm)": std_protein_rmsd,
            "Avg_Protein_RMSF(nm)": avg_protein_rmsf,
            "Std_Protein_RMSF(nm)": std_protein_rmsf,
            # "Avg_COM_Dist(nm)": avg_com,
            # "Std_COM_Dist(nm)": std_com
        }
