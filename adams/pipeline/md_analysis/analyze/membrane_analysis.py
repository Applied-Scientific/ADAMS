"""
Membrane Analysis Module - Transmembrane Protein MD Pipeline

This module analyzes MD trajectories of transmembrane protein systems for
membrane-specific properties and protein stability metrics.

ANALYSES PERFORMED:
    1. Area per lipid - Calculated from box dimensions and lipid count
    2. Membrane thickness proxy - Peak-to-peak distance from membrane mass density
    3. Protein backbone RMSD - Structural stability over time
    4. Density profiles - Mass density along the membrane normal (z-axis)
    5. Water penetration check - Detects water in the membrane hydrophobic core

INPUTS (from file_paths dictionary):
    - membrane_md_tpr: Production MD TPR file
    - membrane_md_xtc: Production MD trajectory
    - membrane_md_gro: Final frame GRO
    - membrane_ndx: Index file with Protein/Membrane/Solvent_and_ions groups
    - membrane_top: Topology file (for lipid counting)
    - membrane_reports_dir: Directory for analysis reports
    - gromacs_path, ambertools_path, gromacs_binary_type

OUTPUTS (added to file_paths dictionary):
    - membrane_analysis_report: Path to CSV summary report
    - membrane_density_xvg: Density profile XVG file
    - membrane_rmsd_xvg: Protein RMSD XVG file

EXTERNAL COMMANDS:
    - gmx energy: Extract box dimensions and energies
    - gmx density: Compute density profiles along z-axis
    - gmx rms: Calculate protein backbone RMSD
    - gmx rmsf: Calculate protein backbone RMSF
"""

import os

import numpy as np
import pandas as pd

from ....logger_utils import get_logger, log_step_execution
from ....utils import run_cmd
from ..shared import (
    GromacsContext,
    LIPID_RESNAMES,
    compute_rmsd,
    compute_rmsf,
    ensure_pbc_corrected_trajectory,
    get_ndx_group_index,
    mean_std,
    parse_xvg,
    parse_xvg_columns,
    resolve_backbone_group,
    select_analysis_range,
)
from ..shared.ion_solvent import parse_topology_molecule_counts


class MembraneAnalysis:
    """
    Analyze membrane MD trajectories for stability and membrane properties.
    """

    def __init__(
        self,
        file_paths,
        prefix: str = "md",
        analysis_range: str = "all",
        last_frames: int = 100,
    ):
        """
        Args:
            file_paths: dict with required paths:
                - membrane_md_tpr, membrane_md_xtc, membrane_md_gro
                - membrane_ndx, membrane_top
                - membrane_reports_dir (or membrane_dir for default)
                - gromacs_path, ambertools_path, gromacs_binary_type
            prefix: MD file prefix (default: "md").
            analysis_range: "all" or "last" (default: "all").
            last_frames: Number of last frames when range="last" (default: 100).
        """
        self.logger = get_logger()

        if file_paths is None:
            raise ValueError("file_paths dictionary is required.")
        self.file_paths = file_paths
        self.validate_files()

        self.ctx = GromacsContext.from_file_paths(file_paths, gpu=False)
        self.gmx_binary = self.ctx.gmx_binary
        self.gromacs_path = file_paths["gromacs_path"]
        self.ambertools_path = file_paths["ambertools_path"]

        self.prefix = prefix
        self.analysis_range = analysis_range
        self.last_frames = last_frames if analysis_range == "last" else None

    def validate_files(self):
        """Validate required paths exist."""
        required = [
            "membrane_md_tpr", "membrane_md_xtc", "membrane_ndx",
            "membrane_top",
        ]
        missing = [k for k in required if not self.file_paths.get(k)]
        if missing:
            raise ValueError(
                f"Required paths missing from file_paths: {missing}\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Ensure run_membrane_gro has completed."
            )

        reports_dir = self.file_paths.get(
            "membrane_reports_dir",
            os.path.join(
                os.path.dirname(self.file_paths["membrane_md_tpr"]),
                "reports",
            ),
        )
        os.makedirs(reports_dir, exist_ok=True)
        self.file_paths["membrane_reports_dir"] = reports_dir

    def run(self) -> dict:
        """
        Run all membrane analyses and generate summary report.

        Returns:
            dict: Updated file_paths with analysis report paths.
        """
        step_logger = log_step_execution("Membrane Analysis", self.logger)
        with step_logger:
            reports_dir = self.file_paths["membrane_reports_dir"]
            membrane_dir = os.path.dirname(
                os.path.abspath(self.file_paths["membrane_md_tpr"])
            )

            fit_xtc = os.path.join(membrane_dir, f"{self.prefix}_fit.xtc")
            self._fit_xtc = ensure_pbc_corrected_trajectory(
                self.gmx_binary,
                self.file_paths["membrane_md_tpr"],
                self.file_paths["membrane_md_xtc"],
                self.file_paths["membrane_ndx"],
                fit_xtc,
                center_group="Protein",
                output_group="System",
            )

            results = {}

            with step_logger.timing("protein_rmsd"):
                rmsd_data = self._protein_rmsd(membrane_dir)
                results.update(rmsd_data)

            with step_logger.timing("protein_rmsf"):
                rmsf_data = self._protein_rmsf(membrane_dir)
                results.update(rmsf_data)

            with step_logger.timing("area_per_lipid"):
                apl_data = self._area_per_lipid(membrane_dir)
                results.update(apl_data)

            with step_logger.timing("density_profiles"):
                density_data = self._density_profiles(membrane_dir)
                results.update(density_data)

            with step_logger.timing("membrane_thickness"):
                thickness_data = self._membrane_thickness(membrane_dir)
                results.update(thickness_data)

            with step_logger.timing("water_penetration"):
                water_data = self._water_penetration_check(membrane_dir)
                results.update(water_data)

            with step_logger.timing("report_generation"):
                report = self._generate_report(results, reports_dir)
                self.file_paths["membrane_analysis_report"] = report

            self.logger.info("\n=== Membrane Analysis Summary ===")
            for key, value in results.items():
                if value is not None:
                    self.logger.info(f"  {key}: {value}")

            return self.file_paths

    # ------------------------------------------------------------------
    # Protein RMSD
    # ------------------------------------------------------------------

    def _protein_rmsd(self, membrane_dir):
        """Calculate protein backbone RMSD using PBC-corrected trajectory."""
        gmx = self.gmx_binary
        tpr = self.file_paths["membrane_md_tpr"]
        xtc = self._fit_xtc
        ndx = self.file_paths["membrane_ndx"]

        rmsd_file = os.path.join(membrane_dir, f"{self.prefix}_protein_rmsd.xvg")

        backbone_grp = resolve_backbone_group(ndx)
        if backbone_grp is None:
            self.logger.warning("Backbone/Protein group not found. Skipping RMSD.")
            return {"Avg_Protein_RMSD(nm)": None, "Std_Protein_RMSD(nm)": None}

        compute_rmsd(gmx, tpr, xtc, ndx, rmsd_file, backbone_grp, backbone_grp)

        _, values = parse_xvg(rmsd_file)
        values = select_analysis_range(values, self.analysis_range, self.last_frames)
        avg, std = mean_std(values)

        self.file_paths["membrane_rmsd_xvg"] = rmsd_file
        return {"Avg_Protein_RMSD(nm)": avg, "Std_Protein_RMSD(nm)": std}

    # ------------------------------------------------------------------
    # Protein RMSF
    # ------------------------------------------------------------------

    def _protein_rmsf(self, membrane_dir):
        """Calculate protein backbone RMSF using PBC-corrected trajectory."""
        gmx = self.gmx_binary
        tpr = self.file_paths["membrane_md_tpr"]
        xtc = self._fit_xtc
        ndx = self.file_paths["membrane_ndx"]

        rmsf_file = os.path.join(membrane_dir, f"{self.prefix}_protein_rmsf.xvg")

        backbone_grp = resolve_backbone_group(ndx)
        if backbone_grp is None:
            self.logger.warning("Backbone/Protein group not found. Skipping RMSF.")
            return {"Avg_Protein_RMSF(nm)": None, "Std_Protein_RMSF(nm)": None}

        compute_rmsf(gmx, tpr, xtc, ndx, rmsf_file, backbone_grp)

        _, values = parse_xvg(rmsf_file)
        avg, std = mean_std(values)
        return {"Avg_Protein_RMSF(nm)": avg, "Std_Protein_RMSF(nm)": std}

    # ------------------------------------------------------------------
    # Area per lipid
    # ------------------------------------------------------------------

    def _area_per_lipid(self, membrane_dir):
        """
        Calculate area per lipid from box dimensions and lipid count.

        APL = (Lx * Ly) / N_lipids_per_leaflet
        where N_lipids_per_leaflet = N_total_lipids / 2 (assuming symmetric bilayer).
        """
        gmx = self.gmx_binary
        tpr = self.file_paths["membrane_md_tpr"]
        xtc = self.file_paths["membrane_md_xtc"]

        # Extract box dimensions using gmx energy
        box_xvg = os.path.join(membrane_dir, f"{self.prefix}_box.xvg")
        energy_cmd = [
            gmx, "energy",
            "-f", os.path.join(membrane_dir, f"{self.prefix}.edr"),
            "-o", box_xvg,
        ]
        # Select Box-X and Box-Y (options depend on GROMACS version, typically items 22 23)
        run_cmd(energy_cmd, input_str="Box-X\nBox-Y\n0\n", check=False)

        if not os.path.exists(box_xvg):
            self.logger.warning(
                "Could not extract box dimensions from energy file. "
                "Skipping area per lipid calculation."
            )
            return {"Avg_APL(nm^2)": None, "Std_APL(nm^2)": None}

        # Box-X is data column 1, Box-Y is data column 2
        box_x_vals, box_y_vals = parse_xvg_columns(box_xvg, columns=(1, 2))

        if len(box_x_vals) == 0 or len(box_y_vals) == 0:
            self.logger.warning("Box dimensions extraction failed. Skipping APL.")
            return {"Avg_APL(nm^2)": None, "Std_APL(nm^2)": None}

        # Count lipids from topology
        top_path = self.file_paths["membrane_top"]
        try:
            counts = parse_topology_molecule_counts(top_path)
        except Exception as e:
            self.logger.warning(f"Could not parse topology for lipid count: {e}")
            return {"Avg_APL(nm^2)": None, "Std_APL(nm^2)": None}

        n_lipids = sum(
            count for name, count in counts.items()
            if name.upper() in LIPID_RESNAMES
        )
        if n_lipids == 0:
            self.logger.warning("No lipids found in topology. Skipping APL.")
            return {"Avg_APL(nm^2)": None, "Std_APL(nm^2)": None}

        n_per_leaflet = n_lipids / 2.0

        # APL = Lx * Ly / N_per_leaflet
        box_x_vals = select_analysis_range(box_x_vals, self.analysis_range, self.last_frames)
        box_y_vals = select_analysis_range(box_y_vals, self.analysis_range, self.last_frames)
        apl_values = (box_x_vals * box_y_vals) / n_per_leaflet

        avg_apl, std_apl = mean_std(apl_values)

        self.logger.info(
            f"Area per lipid: {avg_apl:.4f} +/- {std_apl:.4f} nm^2 "
            f"(total lipids: {n_lipids}, per leaflet: {n_per_leaflet:.0f})"
        )

        return {"Avg_APL(nm^2)": avg_apl, "Std_APL(nm^2)": std_apl}

    # ------------------------------------------------------------------
    # Density profiles
    # ------------------------------------------------------------------

    def _density_profiles(self, membrane_dir):
        """
        Calculate mass density profiles along the membrane normal (z-axis).

        Uses the PBC-corrected trajectory and gmx density with the Membrane
        group to get the lipid mass density distribution, from which a
        membrane-thickness proxy can be estimated.
        """
        gmx = self.gmx_binary
        tpr = self.file_paths["membrane_md_tpr"]
        xtc = self._fit_xtc
        ndx = self.file_paths["membrane_ndx"]

        # System density profile
        system_density_xvg = os.path.join(
            membrane_dir, f"{self.prefix}_density_system.xvg"
        )
        system_grp = None
        for group_name in ("System", "Protein_LIG", "Protein"):
            try:
                system_grp = get_ndx_group_index(ndx, group_name)
                break
            except ValueError:
                continue

        density_cmd = [
            gmx, "density",
            "-s", tpr, "-f", xtc, "-n", ndx,
            "-o", system_density_xvg,
            "-d", "Z",
        ]
        system_density_ok = False
        if system_grp is None:
            self.logger.warning(
                "Could not resolve a named system-like index group "
                "(tried: System, Protein_LIG, Protein). Skipping system density."
            )
        else:
            try:
                run_cmd(density_cmd, input_str=f"{system_grp}\n", check=True)
                system_density_ok = os.path.exists(system_density_xvg)
            except Exception as exc:
                self.logger.warning(f"System density calculation failed: {exc}")

        # Membrane density profile (for thickness estimation)
        membrane_density_xvg = os.path.join(
            membrane_dir, f"{self.prefix}_density_membrane.xvg"
        )
        membrane_density_ok = False
        try:
            membrane_grp = get_ndx_group_index(ndx, "Membrane")
            density_cmd_memb = [
                gmx, "density",
                "-s", tpr, "-f", xtc, "-n", ndx,
                "-o", membrane_density_xvg,
                "-d", "Z",
            ]
            run_cmd(density_cmd_memb, input_str=f"{membrane_grp}\n", check=True)
            membrane_density_ok = os.path.exists(membrane_density_xvg)
        except ValueError:
            self.logger.warning("Membrane group not found. Skipping membrane density.")
            membrane_density_xvg = None
        except Exception as exc:
            self.logger.warning(f"Membrane density calculation failed: {exc}")
            membrane_density_xvg = None

        chosen_density = None
        if membrane_density_ok and membrane_density_xvg:
            chosen_density = membrane_density_xvg
        elif system_density_ok:
            chosen_density = system_density_xvg
        self.file_paths["membrane_density_xvg"] = chosen_density

        status = "computed" if chosen_density else "failed"
        return {
            "density_profile": status,
            "density_profile_membrane": "computed" if membrane_density_ok else "failed",
            "density_profile_system": "computed" if system_density_ok else "failed",
        }

    def _leaflet_peak_positions(self, z_coords, density_vals):
        """Estimate leaflet peak positions from a z-density profile."""
        z_arr = np.asarray(z_coords)
        d_arr = np.asarray(density_vals)
        if z_arr.size < 10 or d_arr.size != z_arr.size:
            return None, None

        midplane_guess = 0.5 * (z_arr[0] + z_arr[-1])
        lower_mask = z_arr < midplane_guess
        upper_mask = z_arr >= midplane_guess
        if not np.any(lower_mask) or not np.any(upper_mask):
            return None, None

        lower_peak_z = z_arr[lower_mask][int(np.argmax(d_arr[lower_mask]))]
        upper_peak_z = z_arr[upper_mask][int(np.argmax(d_arr[upper_mask]))]
        return float(lower_peak_z), float(upper_peak_z)

    # ------------------------------------------------------------------
    # Membrane thickness
    # ------------------------------------------------------------------

    def _membrane_thickness(self, membrane_dir):
        """
        Estimate membrane thickness proxy from the density profile.

        Uses the peak-to-peak distance of the membrane mass density profile
        along the z-axis. This is a useful stability proxy, but it is not a
        strict phosphorus-phosphorus headgroup thickness.
        """
        density_xvg = self.file_paths.get("membrane_density_xvg")
        if not density_xvg or not os.path.exists(density_xvg):
            self.logger.warning(
                "Density profile not available. Skipping thickness estimate."
            )
            return {"Membrane_Thickness(nm)": None}

        z_coords, density_vals = parse_xvg(density_xvg)
        if len(z_coords) < 10:
            self.logger.warning("Density profile too short for thickness estimate.")
            return {"Membrane_Thickness(nm)": None}

        lower_peak_z, upper_peak_z = self._leaflet_peak_positions(
            z_coords, density_vals
        )
        if lower_peak_z is None or upper_peak_z is None:
            self.logger.warning("Could not identify both leaflet peaks.")
            return {"Membrane_Thickness(nm)": None}

        thickness = abs(upper_peak_z - lower_peak_z)

        self.logger.info(
            f"Estimated membrane thickness proxy: {thickness:.2f} nm "
            f"(peak-to-peak from membrane mass density)"
        )

        rounded = round(thickness, 3)
        return {
            "Membrane_Thickness_Proxy(nm)": rounded,
            "Membrane_Thickness(nm)": rounded,
        }

    # ------------------------------------------------------------------
    # Water penetration check
    # ------------------------------------------------------------------

    def _water_penetration_check(self, membrane_dir):
        """
        Check for water penetration into the membrane hydrophobic core.

        Computes the water density profile along z (using the PBC-corrected
        trajectory) and checks if there is significant water density in the
        central region of the membrane.
        """
        gmx = self.gmx_binary
        tpr = self.file_paths["membrane_md_tpr"]
        xtc = self._fit_xtc
        ndx = self.file_paths["membrane_ndx"]

        water_density_xvg = os.path.join(
            membrane_dir, f"{self.prefix}_density_water.xvg"
        )

        water_group = self._resolve_water_group_for_density(ndx)
        if water_group is None:
            self.logger.warning("No water-only index group found for penetration check.")
            return {"Water_Penetration": "unknown"}

        density_cmd = [
            gmx, "density",
            "-s", tpr, "-f", xtc, "-n", ndx,
            "-o", water_density_xvg,
            "-d", "Z",
        ]
        try:
            run_cmd(density_cmd, input_str=f"{water_group}\n", check=True)
        except Exception as exc:
            self.logger.warning(f"Water density calculation failed: {exc}")
            return {"Water_Penetration": "could_not_compute"}

        if not os.path.exists(water_density_xvg):
            return {"Water_Penetration": "could_not_compute"}

        z_coords, water_density = parse_xvg(water_density_xvg)
        if len(z_coords) < 10:
            return {"Water_Penetration": "insufficient_data"}

        # Get membrane midplane from leaflet peak positions (not single-peak center).
        membrane_density_xvg = self.file_paths.get("membrane_density_xvg")
        if membrane_density_xvg and os.path.exists(membrane_density_xvg):
            mz, md = parse_xvg(membrane_density_xvg)
            lower_peak_z, upper_peak_z = self._leaflet_peak_positions(mz, md)
            if lower_peak_z is not None and upper_peak_z is not None:
                center = 0.5 * (lower_peak_z + upper_peak_z)
                core_mask = np.abs(z_coords - center) < 1.0
            else:
                # Fallback: use middle third
                z_range = z_coords[-1] - z_coords[0]
                core_mask = np.abs(z_coords - np.mean(z_coords)) < z_range / 6.0
        else:
            z_range = z_coords[-1] - z_coords[0]
            core_mask = np.abs(z_coords - np.mean(z_coords)) < z_range / 6.0

        if not np.any(core_mask):
            return {"Water_Penetration": "could_not_define_core"}

        core_water_density = water_density[core_mask]
        bulk_water_density = np.max(water_density)

        if bulk_water_density <= 0:
            return {"Water_Penetration": "no_water_detected"}

        penetration_ratio = np.mean(core_water_density) / bulk_water_density

        if penetration_ratio > 0.1:
            status = "WARNING_significant"
            self.logger.warning(
                f"Significant water penetration detected in membrane core: "
                f"ratio={penetration_ratio:.3f} (core avg / bulk max). "
                "This may indicate a pore or equilibration issue."
            )
        elif penetration_ratio > 0.02:
            status = "minor"
            self.logger.info(
                f"Minor water penetration: ratio={penetration_ratio:.3f}"
            )
        else:
            status = "none"
            self.logger.info(
                f"No significant water penetration: ratio={penetration_ratio:.4f}"
            )

        return {
            "Water_Penetration": status,
            "Water_Core_Ratio": round(penetration_ratio, 4),
        }

    def _resolve_water_group_for_density(self, ndx_path):
        """
        Return index group id for water-only density.

        Prefer explicit water groups in the index.  As a fallback, use the first
        detected water residue name from topology molecule counts if present in ndx.
        """
        for group_name in ("SOL", "Water", "WAT", "HOH"):
            try:
                return get_ndx_group_index(ndx_path, group_name)
            except ValueError:
                continue

        try:
            counts = parse_topology_molecule_counts(self.file_paths["membrane_top"])
        except Exception:
            counts = {}

        water_like = {"SOL", "WAT", "HOH", "TIP3", "TIP4", "TIP5", "SPC", "SPCE", "OPC"}
        for resname in counts:
            if resname.upper() not in water_like:
                continue
            try:
                return get_ndx_group_index(ndx_path, resname)
            except ValueError:
                continue

        return None

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(self, results, reports_dir):
        """Generate a CSV summary report of all analyses."""
        report_path = os.path.join(
            reports_dir,
            f"membrane_analysis_{self.analysis_range}.csv",
        )

        # Convert to DataFrame for clean output
        df = pd.DataFrame([results])
        df.to_csv(report_path, index=False)
        self.logger.info(f"Membrane analysis report written to {report_path}")

        return report_path

