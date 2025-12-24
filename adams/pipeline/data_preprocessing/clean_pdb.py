"""
data_preprocessing/clean_pdb.py

Description:
    Prepare protein and ligand PDBs, then combine.
"""

import os
import tempfile

from openmm.app import PDBFile, Topology
from openmm.vec3 import Vec3
from pdbfixer import PDBFixer

from ...logger_utils import get_logger
from ..file_organization import setup_preprocessing_dirs


class CleanPDB:
    def __init__(
        self,
        input_pdb: str,
        outpath: str = "output",
        ligand: bool = False,
        chain_to_keep: str = "A",
    ):
        r"""
        Args:
            input_pdb: str: Input PDB file
            outpath: str: Output directory (default: ./output)
            ligand: bool: Generate ligand with target (default: False)
            chain_to_keep: str: Chain ID to keep (default: A)
        """
        self.input_pdb = input_pdb
        self.outpath = outpath
        self.ligand = ligand
        self.chain_to_keep = chain_to_keep
        self.logger = get_logger()

        # Set up organized directory structure
        self.dir_structure = setup_preprocessing_dirs(outpath)
        os.makedirs(self.dir_structure["receptors"], exist_ok=True)
        if ligand:
            os.makedirs(self.dir_structure["ligands"], exist_ok=True)

        self.input_prefix = os.path.splitext(os.path.basename(input_pdb))[0]

    def clean(self):
        r"""
        Cleans the PDB file and returns the output file.
        """

        output_file = self._cleaned_protein()

        if self.ligand:
            ligand_file_name = self._ligand_pdb()
            if ligand_file_name is not None:
                output_file = self._combine(output_file, ligand_file_name)
            else:
                self.logger.warning(
                    "Ligand extraction failed or no ligand found. Returning cleaned protein only."
                )

        self.logger.info(f"Finished cleaning PDB file and saved to {output_file}")

        return output_file

    def _cleaned_protein(self):
        r"""
        Cleans the PDB file and returns the output file.
        """
        # Load PDB
        fixer = PDBFixer(filename=self.input_pdb)

        # Keep only the protein (remove ligands, ions, and water)
        fixer.removeHeterogens(keepWater=False)

        # Identify chains
        all_chain_ids = [chain.id for chain in fixer.topology.chains()]
        chains_to_remove = [cid for cid in all_chain_ids if cid != self.chain_to_keep]

        # Remove the unwanted chains
        fixer.removeChains(chainIds=chains_to_remove)

        # Add missing residues/atoms
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # Add missing hydrogens (pH 7.0 by default)
        fixer.addMissingHydrogens(pH=7.0)

        # Build output filename in organized directory
        output_pdb = os.path.join(
            self.dir_structure["receptors"],
            f"{self.input_prefix}_{self.chain_to_keep}_clean_h.pdb",
        )

        # Save the cleaned structure
        with open(output_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        self.logger.info(f"Fixed protein saved to {output_pdb}")

        return output_pdb

    def _ligand_pdb(self):
        r"""
        Generates the ligand PDB file.
        """
        with open(self.input_pdb, "r") as f:
            lines = f.readlines()

        ligand_lines = []
        ligand_name = None  # store the ligand name
        ligands = []
        for line in lines:
            if line.startswith("HETATM"):
                # PDB columns are fixed-width:
                ligand_name = line[17:20].strip()
                ligand_chain = line[21].strip()
                if ligand_name != "HOH" and ligand_chain == "A":
                    ligand_lines.append(line)
                    ligands.append(ligand_name)
        unique_ligands = list(dict.fromkeys(ligands))  # keep order
        ligand_set_name = "_".join(unique_ligands)
        self.logger.info(f"Ligand set name: {ligand_set_name}")
        if not ligand_lines:
            self.logger.warning("No ligand found in chain A (excluding water).")
            return

        # Save ligand to organized directory
        output_ligand_pdb = os.path.join(
            self.dir_structure["ligands"], f"{self.input_prefix}_{ligand_set_name}.pdb"
        )
        with open(output_ligand_pdb, "w") as f:
            f.writelines(ligand_lines)

        self.logger.info(f"Ligand saved to {output_ligand_pdb}")

        return output_ligand_pdb

    def _combine(self, pdb1_file, pdb2_file):
        r"""
        Combines two PDB files.
        Args:
            pdb1_file: str: First PDB file
            pdb2_file: str: Second PDB file
        Returns:
            str: Combined PDB file
        """

        # Read PDB files
        pdb1 = PDBFile(pdb1_file)
        pdb2 = PDBFile(pdb2_file)

        base_name = os.path.splitext(os.path.basename(pdb1_file))[0]
        output_file = os.path.join(
            self.dir_structure["receptors"], f"{base_name}_ligand_h.pdb"
        )
        self.logger.info(
            f"Combining PDB files: {pdb1_file} and {pdb2_file} -> {output_file}"
        )
        # Create new topology
        combined_topology = Topology()
        # Mapping from old atoms to new atoms (optional)
        atom_map = {}

        # Copy chains and residues from pdb1
        for chain in pdb1.topology.chains():
            new_chain = combined_topology.addChain(chain.id)
            for res in chain.residues():
                new_res = combined_topology.addResidue(res.name, new_chain, res.id)
                for atom in res.atoms():
                    new_atom = combined_topology.addAtom(
                        atom.name, atom.element, new_res
                    )
                    atom_map[atom] = new_atom

        # Copy chains and residues from pdb2
        # Use new chain IDs if needed (avoid duplicates)
        for chain in pdb2.topology.chains():
            new_chain_id = chain.id
            # ensure unique chain id
            existing_chain_ids = [c.id for c in combined_topology.chains()]
            if new_chain_id in existing_chain_ids:
                # Find next available chain ID (A-Z, then AA-ZZ, etc.)
                single_letter_ids = [cid for cid in existing_chain_ids if len(cid) == 1]
                if single_letter_ids:
                    max_ord = max(ord(cid) for cid in single_letter_ids)
                    if max_ord < ord("Z"):
                        new_chain_id = chr(max_ord + 1)
                    else:
                        # All A-Z used, find first available single letter
                        for candidate in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                            if candidate not in existing_chain_ids:
                                new_chain_id = candidate
                                break
                        else:
                            # If all single letters used, append number to original ID
                            new_chain_id = f"{chain.id}_1"
                else:
                    # No single-letter IDs exist, use first available letter
                    for candidate in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        if candidate not in existing_chain_ids:
                            new_chain_id = candidate
                            break
                    else:
                        # If all single letters used, append number to original ID
                        new_chain_id = f"{chain.id}_1"
            new_chain = combined_topology.addChain(new_chain_id)
            for res in chain.residues():
                new_res = combined_topology.addResidue(res.name, new_chain, res.id)
                for atom in res.atoms():
                    new_atom = combined_topology.addAtom(
                        atom.name, atom.element, new_res
                    )
                    atom_map[atom] = new_atom

        # Combine positions

        # Convert positions to Vec3
        positions1 = [
            Vec3(pos.x * 10.0, pos.y * 10.0, pos.z * 10.0)
            for pos in pdb1.getPositions()
        ]
        positions2 = [
            Vec3(pos.x * 10.0, pos.y * 10.0, pos.z * 10.0)
            for pos in pdb2.getPositions()
        ]

        combined_positions = positions1 + positions2

        # Write combined PDB to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as temp_file:
            temp_pdb_path = temp_file.name
            PDBFile.writeFile(combined_topology, combined_positions, temp_file)

        # Read and process the temporary file
        with open(temp_pdb_path, "r") as f:
            lines = f.readlines()

        # Remove the last line that starts with TER
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("TER"):
                lines.pop(i)
                break  # remove only the last TER

        # Save the modified PDB
        with open(output_file, "w") as f:
            f.writelines(lines)

        self.logger.info(f"Combined PDB saved as {output_file}")
        os.remove(temp_pdb_path)

        return output_file
