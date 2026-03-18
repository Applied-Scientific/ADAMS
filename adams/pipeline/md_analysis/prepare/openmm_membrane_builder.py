"""
OpenMM membrane-system builder for membrane MD workflow.

Builds a membrane-embedded protein system (GRO/TOP) from a protein PDB/CIF using
OpenMM Modeller.addMembrane and exports with ParmEd.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from openmm import unit
from openmm.app import ForceField, Modeller, PDBFile, PDBxFile

from ....error_handling import FatalError
from ....logger_utils import get_logger, log_step_execution

try:
    import parmed as pmd
except Exception:  # pragma: no cover - import error handled at runtime
    pmd = None


_SOLVENT_RESNAMES = {
    "SOL",
    "WAT",
    "HOH",
    "TIP3",
    "TIP4",
    "TIP5",
    "SPC",
    "SPCE",
    "OPC",
}
_ION_RESNAMES = {
    "NA",
    "NA+",
    "SOD",
    "CL",
    "CL-",
    "CLA",
    "K",
    "K+",
    "POT",
    "CA",
    "MG",
    "ZN",
}
_OPENMM_ION_MAP = {
    "K": "K+",
    "K+": "K+",
    "POT": "K+",
    "NA": "Na+",
    "NA+": "Na+",
    "SOD": "Na+",
    "CL": "Cl-",
    "CL-": "Cl-",
    "CLA": "Cl-",
}


def _norm_ion(value: str, fallback: str) -> str:
    if not value:
        return fallback
    key = value.strip().upper()
    return _OPENMM_ION_MAP.get(key, fallback)


def _load_structure(path: str):
    lower = path.lower()
    if lower.endswith((".cif", ".mmcif")):
        cif = PDBxFile(path)
        return cif.topology, cif.positions, "cif"
    pdb = PDBFile(path)
    return pdb.topology, pdb.positions, "pdb"


def _collect_protein_positions(topology, positions) -> list[Tuple[float, float, float]]:
    coords = []
    for atom, pos in zip(topology.atoms(), positions):
        resname = atom.residue.name.strip().upper()
        if resname in _SOLVENT_RESNAMES or resname in _ION_RESNAMES:
            continue
        x, y, z = pos.value_in_unit(unit.nanometer)
        coords.append((float(x), float(y), float(z)))
    return coords


def _orientation_report(points: list[Tuple[float, float, float]]) -> Dict[str, object]:
    if len(points) < 2:
        return {
            "status": "unknown",
            "reason": "not_enough_atoms",
            "z_span_nm": 0.0,
            "x_span_nm": 0.0,
            "y_span_nm": 0.0,
            "z_to_xy_ratio": 0.0,
            "recommended_oriented_for_membrane": False,
        }
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    z_span = max(zs) - min(zs)
    max_xy = max(x_span, y_span, 1e-6)
    z_ratio = z_span / max_xy
    oriented = bool(z_span >= 2.0 and z_ratio >= 0.6)
    return {
        "status": "ok" if oriented else "warn",
        "reason": "z_span_or_ratio_low" if not oriented else "passed",
        "z_span_nm": round(z_span, 6),
        "x_span_nm": round(x_span, 6),
        "y_span_nm": round(y_span, 6),
        "z_to_xy_ratio": round(z_ratio, 6),
        "recommended_oriented_for_membrane": oriented,
    }


class OpenMMMembraneBuilder:
    """
    Build membrane GRO/TOP from a membrane-oriented protein structure.

    This is additive to the existing pre-built membrane import pathway.
    """

    def __init__(
        self,
        file_paths: dict,
        lipid_type: str = "POPC",
        minimum_padding_nm: float = 2.0,
        ionic_strength_m: float = 0.15,
        positive_ion: str = "K+",
        negative_ion: str = "Cl-",
        orientation_policy: str = "warn",
    ):
        if file_paths is None:
            raise ValueError("file_paths dictionary is required.")
        self.file_paths = file_paths
        self.lipid_type = lipid_type or "POPC"
        self.minimum_padding_nm = float(minimum_padding_nm)
        self.ionic_strength_m = float(ionic_strength_m)
        self.positive_ion = _norm_ion(positive_ion, "K+")
        self.negative_ion = _norm_ion(negative_ion, "Cl-")
        self.orientation_policy = (orientation_policy or "warn").strip().lower()
        if self.orientation_policy not in {"warn", "strict", "off"}:
            raise ValueError("orientation_policy must be one of: warn, strict, off")
        self.logger = get_logger()

    def _validate_inputs(self):
        protein_file = self.file_paths.get("protein_file")
        membrane_dir = self.file_paths.get("membrane_dir")
        if not protein_file:
            raise FatalError(
                "protein_file is required to build membrane system from structure."
            )
        if not os.path.exists(protein_file):
            raise FileNotFoundError(f"protein_file not found: {protein_file}")
        if not membrane_dir:
            raise FatalError(
                "membrane_dir is required in file_paths for membrane builder outputs."
            )
        os.makedirs(membrane_dir, exist_ok=True)
        if pmd is None:
            raise FatalError(
                "ParmEd is required for membrane build export (GRO/TOP). "
                "Install dependency 'parmed' in the ADAMS environment."
            )

    def _write_json(self, path: str, payload: dict):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def run(self, force_rebuild: bool = False) -> dict:
        """
        Build membrane system unless valid pre-built system already exists.

        If pre-built membrane_system_gro/top exists and force_rebuild=False,
        this method keeps pre-built sources unchanged.
        """
        existing_gro = self.file_paths.get("membrane_system_gro")
        existing_top = self.file_paths.get("membrane_system_top")
        if (
            not force_rebuild
            and existing_gro
            and existing_top
            and os.path.exists(existing_gro)
            and os.path.exists(existing_top)
        ):
            if not self.file_paths.get("membrane_dir"):
                raise FatalError(
                    "membrane_dir is required in file_paths when using pre-built membrane inputs."
                )
            self.logger.info(
                "Using provided pre-built membrane system (force_rebuild=False)."
            )
            self.file_paths["membrane_source"] = "prebuilt"
            return self.file_paths

        self._validate_inputs()
        step_logger = log_step_execution("OpenMM Membrane Build", self.logger)
        with step_logger:
            membrane_dir = os.path.abspath(self.file_paths["membrane_dir"])

            protein_file = os.path.abspath(self.file_paths["protein_file"])
            topology, positions, input_format = _load_structure(protein_file)

            orientation_path = os.path.join(membrane_dir, "membrane_orientation_report.json")
            if self.orientation_policy != "off":
                orient = _orientation_report(_collect_protein_positions(topology, positions))
                orient["orientation_policy"] = self.orientation_policy
                orient["input_file"] = protein_file
                self._write_json(orientation_path, orient)
                self.file_paths["membrane_orientation_report"] = orientation_path
                if not orient.get("recommended_oriented_for_membrane", False):
                    msg = (
                        "Input protein may not be membrane-oriented: "
                        f"z_span_nm={orient['z_span_nm']}, "
                        f"z_to_xy_ratio={orient['z_to_xy_ratio']}. "
                        "Use OPM/PPM-oriented structure for robust embedding."
                    )
                    if self.orientation_policy == "strict":
                        raise FatalError(msg)
                    self.logger.warning(msg)

            ff, ff_files = self._load_forcefield()
            modeller = Modeller(topology, positions)
            try:
                modeller.addHydrogens(ff)
            except Exception as exc:
                self.logger.warning(
                    f"addHydrogens failed for membrane builder input; continuing without this step: {exc}"
                )

            modeller.addMembrane(
                ff,
                lipidType=self.lipid_type,
                minimumPadding=self.minimum_padding_nm * unit.nanometer,
                ionicStrength=self.ionic_strength_m * unit.molar,
                positiveIon=self.positive_ion,
                negativeIon=self.negative_ion,
            )

            # Use explicit app.PME and HBonds constraints for exported membrane system.
            from openmm.app import PME, HBonds

            system = ff.createSystem(
                modeller.topology,
                nonbondedMethod=PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=HBonds,
            )

            out_pdb = os.path.join(membrane_dir, "system_built.pdb")
            out_gro = os.path.join(membrane_dir, "system.gro")
            out_top = os.path.join(membrane_dir, "system.top")

            with open(out_pdb, "w", encoding="utf-8") as handle:
                PDBFile.writeFile(modeller.topology, modeller.positions, handle)

            structure = pmd.openmm.load_topology(
                modeller.topology, system=system, xyz=modeller.positions
            )
            structure.save(out_gro, overwrite=True)
            structure.save(out_top, overwrite=True)

            box_vectors = modeller.topology.getPeriodicBoxVectors()
            box_nm = None
            if box_vectors is not None:
                box_nm = [
                    [float(c) for c in vec.value_in_unit(unit.nanometer)]
                    for vec in box_vectors
                ]

            build_report = {
                "source_mode": "built_from_protein",
                "input_file": protein_file,
                "input_format": input_format,
                "lipid_type": self.lipid_type,
                "minimum_padding_nm": self.minimum_padding_nm,
                "ionic_strength_m": self.ionic_strength_m,
                "positive_ion": self.positive_ion,
                "negative_ion": self.negative_ion,
                "forcefield_files": list(ff_files),
                "num_atoms": sum(1 for _ in modeller.topology.atoms()),
                "num_residues": sum(1 for _ in modeller.topology.residues()),
                "num_chains": sum(1 for _ in modeller.topology.chains()),
                "periodic_box_vectors_nm": box_nm,
                "outputs": {
                    "system_built_pdb": out_pdb,
                    "membrane_system_gro": out_gro,
                    "membrane_system_top": out_top,
                },
            }
            build_report_path = os.path.join(membrane_dir, "membrane_build_report.json")
            self._write_json(build_report_path, build_report)

            self.file_paths["membrane_system_gro"] = out_gro
            self.file_paths["membrane_system_top"] = out_top
            self.file_paths["membrane_build_report"] = build_report_path
            self.file_paths["membrane_source"] = "built_from_protein"
            return self.file_paths

    def _load_forcefield(self):
        candidates = [
            (
                "amber14/protein.ff14SB.xml",
                "amber14/lipid17.xml",
                "amber14/tip3p.xml",
            ),
            ("amber14-all.xml", "lipid17.xml", "tip3p.xml"),
        ]
        errors = []
        for ff_files in candidates:
            try:
                return ForceField(*ff_files), ff_files
            except Exception as exc:  # pragma: no cover - depends on local OpenMM data
                errors.append(f"{ff_files}: {type(exc).__name__}: {exc}")
        details = "\n".join(errors)
        raise FatalError(
            "Failed to load OpenMM membrane forcefield stack for Amber14SB+Lipid17+TIP3P. "
            "Tried known ffxml combinations and none were available. "
            "Ensure OpenMM forcefield data is installed in the active environment.\n"
            f"{details}"
        )
