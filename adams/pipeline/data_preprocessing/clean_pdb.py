"""
data_preprocessing/clean_pdb.py

Description:
    Prepare protein and ligand PDBs, then combine.
    Supports keeping essential heterogens (cofactors, ions) and optional waters.
"""

import os
import shlex
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple, Union

from openmm.app import PDBFile, Topology
from openmm.vec3 import Vec3
from pdbfixer import PDBFixer

from ...logger_utils import get_logger
from ..file_organization import setup_preprocessing_dirs
from .pdb_utils import (
    ESSENTIAL_HETEROGENS_TO_KEEP,
    identify_gaps,
    write_pdb_with_chain_breaks,
)

WATER_RESNAMES = {
    "HOH", "WAT", "SOL", "H2O", "DOD", "D2O", "OH2",
    "TIP", "TP3", "T3P", "SPC",
}


def _parse_poly_seq_scheme_rows(path: str) -> Dict[str, List[Tuple[int, str, bool]]]:
    """
    Parse mmCIF _pdbx_poly_seq_scheme rows.

    Returns:
        Dict mapping chain_id -> list of tuples (seq_id, residue_name, has_coordinates)
    """
    rows_by_chain: Dict[str, List[Tuple[int, str, bool]]] = {}
    if not os.path.isfile(path):
        return rows_by_chain

    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    target_cols = {
        "_pdbx_poly_seq_scheme.asym_id",
        "_pdbx_poly_seq_scheme.seq_id",
        "_pdbx_poly_seq_scheme.mon_id",
        "_pdbx_poly_seq_scheme.pdb_mon_id",
    }

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line != "loop_":
            i += 1
            continue

        i += 1
        headers: List[str] = []
        while i < n and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        if not headers:
            continue

        if not target_cols.issubset(set(headers)):
            while i < n:
                raw = lines[i].strip()
                if raw.startswith("#"):
                    i += 1
                    break
                if raw == "loop_" or raw.startswith("_"):
                    break
                i += 1
            continue

        col_idx = {name: idx for idx, name in enumerate(headers)}
        c_asym = col_idx["_pdbx_poly_seq_scheme.asym_id"]
        c_seq = col_idx["_pdbx_poly_seq_scheme.seq_id"]
        c_mon = col_idx["_pdbx_poly_seq_scheme.mon_id"]
        c_pdb_mon = col_idx["_pdbx_poly_seq_scheme.pdb_mon_id"]
        num_cols = len(headers)
        token_buffer: List[str] = []

        while i < n:
            raw = lines[i].strip()
            if not raw:
                i += 1
                continue
            if raw.startswith("#"):
                i += 1
                break
            if raw == "loop_" or raw.startswith("_"):
                break
            if raw.startswith(";"):
                i += 1
                while i < n and lines[i].strip() != ";":
                    i += 1
                if i < n:
                    i += 1
                continue

            token_buffer.extend(shlex.split(raw, posix=True))
            while len(token_buffer) >= num_cols:
                row = token_buffer[:num_cols]
                del token_buffer[:num_cols]
                asym_id = row[c_asym]
                seq_text = row[c_seq]
                mon_id = row[c_mon]
                pdb_mon = row[c_pdb_mon]

                if asym_id in (".", "?") or seq_text in (".", "?"):
                    continue
                try:
                    seq_id = int(seq_text)
                except ValueError:
                    continue

                has_coordinates = pdb_mon not in (".", "?")
                rows_by_chain.setdefault(asym_id, []).append(
                    (seq_id, mon_id, has_coordinates)
                )
            i += 1

    return rows_by_chain


def _infer_missing_residues_from_mmcif(
    path: str, fixer: PDBFixer
) -> Dict[Tuple[int, int], List[str]]:
    """
    Infer missing-residue blocks from mmCIF polymer sequence table.

    This is a fallback for structures where PDBFixer cannot infer missing residues
    from numbering alone.
    """
    rows_by_chain = _parse_poly_seq_scheme_rows(path)
    if not rows_by_chain:
        return {}

    chains = list(fixer.topology.chains())
    chain_index_by_id = {chain.id: chain.index for chain in chains}
    inferred: Dict[Tuple[int, int], List[str]] = {}

    for chain_id, rows in rows_by_chain.items():
        if chain_id not in chain_index_by_id:
            continue
        rows = sorted(rows, key=lambda row: row[0])
        chain_index = chain_index_by_id[chain_id]
        present_before = 0
        j = 0

        while j < len(rows):
            _seq_id, mon_id, has_coord = rows[j]
            if has_coord:
                present_before += 1
                j += 1
                continue

            names: List[str] = [mon_id]
            j += 1
            while j < len(rows) and not rows[j][2]:
                names.append(rows[j][1])
                j += 1

            inferred[(chain_index, present_before)] = names

    return inferred


class CleanPDB:
    def __init__(
        self,
        input_pdb: str,
        outpath: str = "output",
        ligand: bool = False,
        chain_to_keep: Optional[Union[str, Sequence[str]]] = "all",
        keep_water: bool = False,
        keep_heterogens: Optional[Union[Sequence[str], str]] = "essential",
        model_missing_residues: bool = True,
        max_missing_residues_per_gap: int = 12,
        allow_terminal_missing_residues: bool = False,
        pH: float = 7.4,
    ):
        r"""
        Args:
            input_pdb: str: Input PDB file
            outpath: str: Output directory (default: ./output)
            ligand: bool: Generate ligand with target (default: False)
            chain_to_keep: Chain selector.
                - "all" or None (default): keep all chains
                - "A": keep one chain
                - "A,B,C" or ["A","B","C"]: keep selected chains
            keep_water: bool: If True, retain water molecules (e.g. structural waters).
                Default: False.
            keep_heterogens: "essential" (default) = keep ESSENTIAL_HETEROGENS_TO_KEEP. None or [] = remove all.
                A list or single 3-letter str = keep only those.
            model_missing_residues: If True (default), model selected missing-residue
                blocks during cleanup.
            max_missing_residues_per_gap: Safety cap for modeled gap size. Gaps larger
                than this are left as chain breaks. Default: 12.
            allow_terminal_missing_residues: If False (default), do not model N/C-terminal
                missing stretches.
            pH: float: pH value (default: 7.4). Passed to run_protonate_receptor.
        """
        self.input_pdb = input_pdb
        self.outpath = outpath
        self.ligand = ligand
        self.chain_to_keep = chain_to_keep
        self._chains_to_keep = self._normalize_chain_selection(chain_to_keep)
        self.keep_water = keep_water
        self.model_missing_residues = model_missing_residues
        self.max_missing_residues_per_gap = max_missing_residues_per_gap
        self.allow_terminal_missing_residues = allow_terminal_missing_residues
        self.pH = pH
        # "essential" (default) or explicit "essential" → essential set; None or [] → remove all; list or single str → keep those
        if keep_heterogens is None:
            self._keep_heterogens_set = None
        elif keep_heterogens == "essential":
            self._keep_heterogens_set = frozenset(
                r.strip().upper()[:3] for r in ESSENTIAL_HETEROGENS_TO_KEEP
            )
        elif isinstance(keep_heterogens, (list, tuple)) and len(keep_heterogens) == 0:
            self._keep_heterogens_set = None
        else:
            names = [keep_heterogens] if isinstance(keep_heterogens, str) else keep_heterogens
            self._keep_heterogens_set = frozenset(n.strip().upper()[:3] for n in names)
        self.logger = get_logger()

        # Set up organized directory structure
        self.dir_structure = setup_preprocessing_dirs(outpath)
        os.makedirs(self.dir_structure["receptors"], exist_ok=True)
        if ligand:
            os.makedirs(self.dir_structure["ligands"], exist_ok=True)

        self.input_prefix = os.path.splitext(os.path.basename(input_pdb))[0]

    def _partition_missing_residue_blocks(
        self, fixer: PDBFixer
    ) -> Tuple[Dict[Tuple[int, int], List[str]], Dict[Tuple[int, int], List[str]], List[str]]:
        """
        Split detected missing-residue blocks into:
        - blocks_to_model
        - blocks_to_leave_as_gaps
        with conservative safety filters.
        """
        blocks_to_model: Dict[Tuple[int, int], List[str]] = {}
        blocks_to_leave: Dict[Tuple[int, int], List[str]] = {}
        messages: List[str] = []

        if not fixer.missingResidues:
            return blocks_to_model, blocks_to_leave, messages

        chains = list(fixer.topology.chains())
        chain_len_by_index = {
            chain.index: len(list(chain.residues()))
            for chain in chains
        }
        chain_id_by_index = {chain.index: chain.id for chain in chains}

        for key, residue_names in sorted(fixer.missingResidues.items()):
            if not isinstance(key, tuple) or len(key) < 2:
                blocks_to_leave[key] = residue_names
                messages.append(
                    f"Skipping malformed missing-residue key {key}; leaving as gap."
                )
                continue

            chain_index, insert_index = key[0], key[1]
            chain_id = chain_id_by_index.get(chain_index, str(chain_index))
            chain_len = chain_len_by_index.get(chain_index, None)
            is_terminal = (
                insert_index == 0
                or (chain_len is not None and insert_index >= chain_len)
            )
            block_size = len(residue_names)

            if not self.model_missing_residues:
                blocks_to_leave[key] = residue_names
                continue

            if (
                self.max_missing_residues_per_gap is not None
                and block_size > self.max_missing_residues_per_gap
            ):
                blocks_to_leave[key] = residue_names
                messages.append(
                    f"Leaving gap {chain_id}@{insert_index} unmodeled: "
                    f"block size {block_size} exceeds max_missing_residues_per_gap="
                    f"{self.max_missing_residues_per_gap}."
                )
                continue

            if is_terminal and not self.allow_terminal_missing_residues:
                blocks_to_leave[key] = residue_names
                messages.append(
                    f"Leaving terminal gap {chain_id}@{insert_index} unmodeled "
                    f"(block size {block_size})."
                )
                continue

            blocks_to_model[key] = residue_names

        return blocks_to_model, blocks_to_leave, messages

    @staticmethod
    def _normalize_chain_selection(
        chain_to_keep: Optional[Union[str, Sequence[str]]]
    ) -> Optional[frozenset[str]]:
        """
        Normalize chain selection into a frozenset. None means keep all chains.
        """
        if chain_to_keep is None:
            return None

        if isinstance(chain_to_keep, str):
            raw = chain_to_keep.strip()
            if not raw:
                raise ValueError("chain_to_keep cannot be empty.")
            if raw.lower() in {"all", "*", "any"}:
                return None
            normalized = raw.replace(",", " ").replace(";", " ")
            tokens = [tok.strip() for tok in normalized.split() if tok.strip()]
        else:
            tokens = [str(tok).strip() for tok in chain_to_keep if str(tok).strip()]

        if not tokens:
            raise ValueError("No valid chain IDs provided in chain_to_keep.")

        if any(tok.lower() in {"all", "*", "any"} for tok in tokens):
            return None

        return frozenset(tokens)

    def _chain_label(self) -> str:
        if self._chains_to_keep is None:
            return "all"
        if len(self._chains_to_keep) == 1:
            return next(iter(self._chains_to_keep))
        return "-".join(sorted(self._chains_to_keep))

    def _chain_is_selected(self, chain_id: str) -> bool:
        if self._chains_to_keep is None:
            return True
        return chain_id in self._chains_to_keep

    def _hetero_chain_is_selected(self, chain_id: str) -> bool:
        """
        HETATM chain selection:
        - In all-chain mode, keep all.
        - In selected-chain mode, keep selected chains and blank-chain records.
          Blank chain IDs are common for crystallographic waters/cofactors.
        """
        if self._chains_to_keep is None:
            return True
        return (not chain_id) or (chain_id in self._chains_to_keep)

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

    def _convert_structure_to_temp_pdb(self, path: str) -> str:
        """
        Convert a non-PDB receptor structure (e.g. mmCIF) into a temporary PDB.
        This enables residue-name based heterogen filtering with _filter_heterogens_pdb.
        """
        fixer = PDBFixer(filename=path)
        fd, temp_path = tempfile.mkstemp(suffix=".pdb", prefix="clean_pdb_src_")
        try:
            with os.fdopen(fd, "w") as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
            return temp_path
        except Exception:
            os.close(fd)
            raise


    def _filter_heterogens_pdb(self, path: str) -> str:
        r"""
        Pre-filter PDB to keep only ATOM lines and allowed HETATMs (waters and/or
        keep_heterogens). Returns path to a temporary PDB file. Caller must delete it.

        PDBFixer.removeHeterogens(keepWater=bool) only supports keepWater; it does not
        accept a list of residue names to keep (see openmm/pdbfixer pdbfixer.py).
        So we filter the PDB first, then run PDBFixer without calling removeHeterogens.
        """
        keep_water = self.keep_water
        keep_set = self._keep_heterogens_set or frozenset()
        out_lines = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("ATOM  "):
                    line_chain = line[21:22].strip() if len(line) > 21 else ""
                    if self._chain_is_selected(line_chain):
                        out_lines.append(line)
                    continue
                if line.startswith("HETATM"):
                    resname = (line[17:20] or "").strip().upper()
                    line_chain = line[21:22].strip() if len(line) > 21 else ""
                    if (
                        keep_water
                        and resname in WATER_RESNAMES
                        and self._hetero_chain_is_selected(line_chain)
                    ):
                        out_lines.append(line)
                    elif resname in keep_set and self._hetero_chain_is_selected(line_chain):
                        out_lines.append(line)
                    # else: drop this heterogen
                    continue
                if line.startswith("CONECT") or line.startswith("LINK"):
                    out_lines.append(line)
                    continue
                # MODEL/ENDMDL, TER, END, headers, etc.: keep to preserve structure
                if line.startswith("TER") or line.startswith("END") or line.startswith("MODEL") or line.startswith("ENDMDL"):
                    out_lines.append(line)
                    continue
                if line.startswith("HEADER") or line.startswith("TITLE") or line.startswith("REMARK") or line.startswith("CRYST1") or line.startswith("ANISOU"):
                    out_lines.append(line)
                    continue
        fd, temp_path = tempfile.mkstemp(suffix=".pdb", prefix="clean_pdb_")
        try:
            with os.fdopen(fd, "w") as f:
                f.writelines(out_lines)
            return temp_path
        except Exception:
            os.close(fd)
            raise

    def _cleaned_protein(self):
        r"""
        Cleans the PDB file and returns the output file.
        """
        pdb_to_fix = self.input_pdb
        temp_files: List[str] = []
        use_prefilter = self.keep_water or (
            self._keep_heterogens_set is not None and len(self._keep_heterogens_set) > 0
        )
        is_pdb_like_input = self.input_pdb.lower().endswith((".pdb", ".ent"))

        if use_prefilter and is_pdb_like_input:
            pdb_to_fix = self._filter_heterogens_pdb(self.input_pdb)
            temp_files.append(pdb_to_fix)
            self.logger.info(
                "Keeping selected heterogens/waters; using pre-filtered structure for fixing."
            )
        elif use_prefilter and not is_pdb_like_input:
            try:
                prefilter_source = self._convert_structure_to_temp_pdb(self.input_pdb)
                temp_files.append(prefilter_source)
                pdb_to_fix = self._filter_heterogens_pdb(prefilter_source)
                temp_files.append(pdb_to_fix)
                self.logger.info(
                    "Converted non-PDB input to temporary PDB for selective heterogen filtering."
                )
            except Exception as e:
                self.logger.warning(
                    "Selective heterogen filtering for this input format failed (%s). "
                    "Falling back to PDBFixer removeHeterogens(keepWater=%s).",
                    e,
                    self.keep_water,
                )
                pdb_to_fix = self.input_pdb

        try:
            fixer = PDBFixer(filename=pdb_to_fix)
            if pdb_to_fix == self.input_pdb:
                if use_prefilter:
                    if self._keep_heterogens_set:
                        self.logger.warning(
                            "Selective keep_heterogens is unavailable without pre-filtering for this input format. "
                            "Applying removeHeterogens(keepWater=%s) instead.",
                            self.keep_water,
                        )
                    fixer.removeHeterogens(keepWater=self.keep_water)
                else:
                    # No filtering requested: remove all heterogens and waters via PDBFixer.
                    fixer.removeHeterogens(keepWater=False)

            # Identify chains
            all_chain_ids = [chain.id for chain in fixer.topology.chains()]

            if pdb_to_fix != self.input_pdb:
                # Chain filtering is already applied in _filter_heterogens_pdb to ATOM lines.
                # Avoid fixer.removeChains() here, otherwise blank-chain waters/cofactors can be dropped.
                self.logger.info(
                    "Skipping fixer.removeChains(): ATOM chain filtering was already applied during pre-filter."
                )
                if self._chains_to_keep is not None:
                    present_selected = [cid for cid in all_chain_ids if cid in self._chains_to_keep]
                    if not present_selected:
                        requested = ",".join(sorted(self._chains_to_keep))
                        available = ",".join(all_chain_ids) or "<none>"
                        raise ValueError(
                            f"None of the requested chains ({requested}) were found after pre-filtering. "
                            f"Available chains: {available}"
                        )
            else:
                if self._chains_to_keep is None:
                    chains_to_remove = []
                    self.logger.info("Keeping all chains in receptor cleanup.")
                else:
                    missing_chains = sorted(set(self._chains_to_keep) - set(all_chain_ids))
                    if missing_chains:
                        self.logger.warning(
                            "Requested chain(s) not found and will be ignored: %s. Available chains: %s",
                            ",".join(missing_chains),
                            ",".join(all_chain_ids),
                        )

                    selected_in_structure = [cid for cid in all_chain_ids if cid in self._chains_to_keep]
                    if not selected_in_structure:
                        requested = ",".join(sorted(self._chains_to_keep))
                        available = ",".join(all_chain_ids) or "<none>"
                        raise ValueError(
                            f"None of the requested chains ({requested}) were found in structure. "
                            f"Available chains: {available}"
                        )

                    chains_to_remove = [cid for cid in all_chain_ids if cid not in self._chains_to_keep]

                # Remove the unwanted chains
                if chains_to_remove:
                    fixer.removeChains(chainIds=chains_to_remove)

            # Find missing residues and attempt to model selected blocks.
            fixer.findMissingResidues()

            # mmCIF fallback: infer missing blocks from _pdbx_poly_seq_scheme when
            # PDBFixer cannot infer from residue numbering.
            if (
                not fixer.missingResidues
                and self.input_pdb.lower().endswith((".cif", ".mmcif", ".pdbx"))
            ):
                inferred = _infer_missing_residues_from_mmcif(self.input_pdb, fixer)
                if inferred:
                    fixer.missingResidues = inferred
                    self.logger.info(
                        "Inferred %d missing-residue block(s) from mmCIF "
                        "_pdbx_poly_seq_scheme fallback.",
                        len(inferred),
                    )

            total_blocks = len(fixer.missingResidues)
            total_missing_res = sum(len(v) for v in fixer.missingResidues.values())
            if total_blocks:
                self.logger.info(
                    "Detected %d missing-residue block(s) (%d residues total) before filtering.",
                    total_blocks,
                    total_missing_res,
                )

            blocks_to_model, blocks_to_leave, partition_msgs = self._partition_missing_residue_blocks(
                fixer
            )
            for msg in partition_msgs:
                self.logger.info(msg)

            if blocks_to_model:
                model_count = sum(len(v) for v in blocks_to_model.values())
                self.logger.info(
                    "Will model %d missing-residue block(s) (%d residues total).",
                    len(blocks_to_model),
                    model_count,
                )
            if blocks_to_leave:
                leave_count = sum(len(v) for v in blocks_to_leave.values())
                self.logger.info(
                    "Leaving %d missing-residue block(s) (%d residues total) as chain gaps.",
                    len(blocks_to_leave),
                    leave_count,
                )
            
            single_chain_mode = (
                self._chains_to_keep is not None and len(self._chains_to_keep) == 1
            )
            target_chain = next(iter(self._chains_to_keep)) if single_chain_mode else None

            # For chain-gap writing, keep only unmodeled blocks in missingResidues.
            fixer.missingResidues = blocks_to_leave

            gap_positions = []
            if single_chain_mode and target_chain is not None:
                # Check for gaps (always check, even if missingResidues is empty,
                # as identify_gaps can detect structural gaps like out-of-order residues)
                gap_positions = identify_gaps(fixer, target_chain)
                if gap_positions:
                    self.logger.info(
                        f"Found {len(gap_positions)} gap(s) in chain {target_chain}. "
                        f"Chain will be split at gap positions to avoid artificial bonding."
                    )
                # Store gap positions for use in PDB writing
                fixer._gap_positions = gap_positions
            else:
                self.logger.info(
                    "Gap-based TER insertion is applied only in single-chain mode. "
                    "Writing multi-chain structure with standard PDB writer."
                )
            
            # Switch to blocks selected for modeling before addMissingAtoms().
            fixer.missingResidues = blocks_to_model

            # Some PDBFixer versions expose addMissingResidues(); if available,
            # call it first, then run addMissingAtoms().
            if hasattr(fixer, "addMissingResidues"):
                try:
                    fixer.addMissingResidues()
                    self.logger.info(
                        "Applied addMissingResidues() prior to addMissingAtoms()."
                    )
                except Exception as e:
                    self.logger.warning(
                        "addMissingResidues() failed (%s); continuing with addMissingAtoms().",
                        e,
                    )

            fixer.findMissingAtoms()
            fixer.addMissingAtoms()

            # Build output filename in organized directory
            output_pdb = os.path.join(
                self.dir_structure["receptors"],
                f"{self.input_prefix}_{self._chain_label()}_clean.pdb",
            )

            if single_chain_mode and target_chain is not None:
                # Save the cleaned structure with explicit chain breaks at gaps.
                write_pdb_with_chain_breaks(
                    fixer, output_pdb, target_chain, gap_positions, self.logger
                )
            else:
                with open(output_pdb, "w") as f:
                    PDBFile.writeFile(fixer.topology, fixer.positions, f)

            self.logger.info(f"Fixed protein saved to {output_pdb}")

            return output_pdb
        finally:
            for temp_file in sorted(set(temp_files)):
                if temp_file != self.input_pdb and os.path.isfile(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass


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
                if ligand_name != "HOH" and self._chain_is_selected(ligand_chain):
                    ligand_lines.append(line)
                    ligands.append(ligand_name)
        unique_ligands = list(dict.fromkeys(ligands))  # keep order
        ligand_set_name = "_".join(unique_ligands)
        self.logger.info(f"Ligand set name: {ligand_set_name}")
        if not ligand_lines:
            chain_scope = (
                "all chains"
                if self._chains_to_keep is None
                else f"chain(s) {','.join(sorted(self._chains_to_keep))}"
            )
            self.logger.warning(
                f"No ligand found in {chain_scope} (excluding water)."
            )
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
        # Note: This combines cleaned protein (no hydrogens) with ligand
        # Protonation should be done on the protein-only file before combining
        output_file = os.path.join(
            self.dir_structure["receptors"], f"{base_name}_ligand.pdb"
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
