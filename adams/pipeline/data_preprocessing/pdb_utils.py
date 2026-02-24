"""
data_preprocessing/pdb_utils.py

General-purpose utilities for PDB file manipulation and processing.
"""

import os
import math
import re
import tempfile
from typing import Optional, Tuple, Set

from openmm import unit
from openmm.app import PDBFile
from pdbfixer import PDBFixer


# Essential heterogens to keep to avoid stripping cofactors/metal centers:
# This list aims to cover common enzymatic prosthetic groups, redox cofactors, nucleotides,
# metal ions, and metal clusters seen in PDBs.
ESSENTIAL_HETEROGENS_TO_KEEP = [
    # Heme and close variants
    "HEM", "HEA", "HEB", "HEC", "HEO", "HEV",
    # Flavins
    "FAD", "FMN",
    # NAD / NADP family (common PDB residue names)
    "NAD", "NAP", "NAI", "NDP", "NMA", "NMN", "NHE",
    # Nucleotides (common enzymatic ligands)
    "ATP", "ADP", "AMP",
    "GTP", "GDP", "GMP",
    "CTP", "CDP", "CMP",
    "UTP", "UDP", "UMP",
    "IMP", "IDP",
    # Coenzyme A and acyl carrier
    "COA", "ACP",
    # PLP / B6 family
    "PLP", "PMP", "PNP",
    # Biopterin / folates
    "BH4", "H4B", "FOL", "THF",
    # S-adenosyl
    "SAM", "SAH",
    # Glutathione
    "GSH", "GSS",
    # Thiamine / biotin / lipoate
    "TPP", "BTN", "LPA", "LPP",
    # Quinones / specialized redox cofactors (present in many enzymes)
    "PQQ", "F42", "F420", "UQ", "UQ1", "UQ2", "MQ",
    # Retinal (opsins, etc.)
    "RET",
    # Metal ions commonly essential for structure/function
    "MG", "MN", "ZN", "CA", "FE", "FE2", "FE3", "CU", "CU1", "CO", "NI",
    # Common structural/catalytic metals seen in PDBs (sometimes experimental, but often functional)
    "CD", "SR",
    # Iron-sulfur clusters (very important to keep)
    "SF4", "FES", "FS4", "FE2S",
]


def parse_residue_id(res_id) -> Tuple[int, Optional[str]]:
    """
    Parse residue ID handling insertion codes and negative numbers.
    
    Args:
        res_id: Residue ID (can be int, tuple, or string)
        
    Returns:
        Tuple of (sequence_number, insertion_code) where insertion_code can be None
        
    Raises:
        ValueError: If residue ID cannot be parsed
    """
    if isinstance(res_id, int):
        return (res_id, None)
    elif isinstance(res_id, tuple):
        # OpenMM format: (chain_index, sequence_number) or (chain_index, sequence_number, insertion_code)
        if len(res_id) >= 2:
            seq_num = res_id[1]
            ins_code = res_id[2] if len(res_id) > 2 else None
            return (seq_num, ins_code)
    elif isinstance(res_id, str):
        # Parse strings like "100", "100A", "-5", "-5B"
        m = re.match(r"^(-?\d+)([A-Za-z]?)$", res_id.strip())
        if m:
            seq_num = int(m.group(1))
            ins_code = m.group(2) or None
            return (seq_num, ins_code)
    
    raise ValueError(f"Cannot parse residue ID: {res_id}")


def parse_pdb_residue_id(line: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Parse residue ID from PDB ATOM/HETATM line.
    
    Args:
        line: PDB format line
        
    Returns:
        Tuple of (sequence_number, insertion_code, chain_id) or (None, None, None) if parsing fails
    """
    if len(line) < 26:
        return (None, None, None)
    
    try:
        chain_id = line[21:22].strip() if len(line) > 21 else None
        seq_str = line[22:26].strip() if len(line) > 26 else ""
        ins_code = line[26:27].strip() if len(line) > 26 and line[26:27].strip() else None
        
        # Handle negative numbers (they may have a minus sign)
        seq_num = int(seq_str)
        
        return (seq_num, ins_code, chain_id)
    except (ValueError, IndexError):
        return (None, None, None)


def compare_residue_sequence(seq1: Tuple[int, Optional[str]], seq2: Tuple[int, Optional[str]]) -> int:
    """
    Compare two residue sequence identifiers.
    
    Args:
        seq1: First residue sequence (sequence_number, insertion_code)
        seq2: Second residue sequence (sequence_number, insertion_code)
    
    Returns:
        Negative if seq1 < seq2, positive if seq1 > seq2, 0 if equal
    """
    num1, ins1 = seq1
    num2, ins2 = seq2
    
    if num1 != num2:
        return num1 - num2
    
    # Same sequence number, compare insertion codes
    if ins1 is None and ins2 is None:
        return 0
    if ins1 is None:
        return -1  # No insertion code comes before insertion code
    if ins2 is None:
        return 1
    # Both have insertion codes, compare alphabetically
    return (ins1 > ins2) - (ins1 < ins2)


def build_missing_residue_ids(fixer: PDBFixer, target_chain) -> Set[Tuple[int, Optional[str]]]:
    """
    Build a set of missing residue identifiers from PDBFixer missingResidues dict.
    
    PDBFixer's missingResidues dictionary structure:
    - Keys: tuples of (chain.index, residue_index) where:
      * chain.index: integer index of the chain in the topology
      * residue_index: integer position within that chain where residues should be inserted
    - Values: lists of residue names (strings) to insert at that position
    
    Args:
        fixer: PDBFixer instance with missingResidues populated
        target_chain: Chain object to extract missing residues for
        
    Returns:
        Set of (sequence_number, insertion_code) tuples for residues adjacent to gaps
    """
    missing_residue_ids = set()
    
    if not fixer.missingResidues:
        return missing_residue_ids
    
    residues = list(target_chain.residues())
    target_chain_index = target_chain.index
    
    # Iterate through missingResidues dictionary
    # Keys are tuples: (chain.index, residue_index) where both are integers
    for key, residue_names in fixer.missingResidues.items():
        # Safety check: ensure key is a tuple with at least 2 elements
        if not isinstance(key, tuple) or len(key) < 2:
            continue
        
        chain_index, residue_index = key[0], key[1]
        
        # Check if this missing residue belongs to our target chain
        if chain_index != target_chain_index:
            continue
        
        # Ensure residue_index is an integer
        if not isinstance(residue_index, int):
            continue
        
        # residue_index refers to the position where residues should be inserted
        # Map this to the actual sequence numbers from the topology
        
        if residue_index == 0:
            # Missing residues at the beginning of the chain
            if residues:
                try:
                    first_seq, first_ins = parse_residue_id(residues[0].id)
                    missing_residue_ids.add((first_seq, first_ins))
                except (ValueError, TypeError, AttributeError):
                    pass
        elif 0 < residue_index <= len(residues):
            # Missing residues at position residue_index
            # The gap is between residue_index-1 and residue_index
            try:
                # Get the residue before the insertion point
                prev_res = residues[residue_index - 1]
                prev_seq, prev_ins = parse_residue_id(prev_res.id)
                missing_residue_ids.add((prev_seq, prev_ins))
                
                # If there's a residue at residue_index, also add its sequence number
                if residue_index < len(residues):
                    next_res = residues[residue_index]
                    next_seq, next_ins = parse_residue_id(next_res.id)
                    missing_residue_ids.add((next_seq, next_ins))
            except (ValueError, TypeError, AttributeError, IndexError):
                pass
        elif residue_index > len(residues):
            # Missing residues after the last residue
            if residues:
                try:
                    last_seq, last_ins = parse_residue_id(residues[-1].id)
                    missing_residue_ids.add((last_seq, last_ins))
                except (ValueError, TypeError, AttributeError):
                    pass
    
    return missing_residue_ids


def check_missing_residues_between(
    current_seq: Tuple[int, Optional[str]],
    next_seq: Tuple[int, Optional[str]],
    missing_residue_ids: Set[Tuple[int, Optional[str]]]
) -> bool:
    """
    Check if there are missing residues between two consecutive residues.
    
    Args:
        current_seq: Current residue sequence (sequence_number, insertion_code)
        next_seq: Next residue sequence (sequence_number, insertion_code)
        missing_residue_ids: Set of missing residue identifiers
        
    Returns:
        True if there are missing residues between current and next
    """
    current_num, current_ins = current_seq
    next_num, next_ins = next_seq
    
    for missing_seq, missing_ins in missing_residue_ids:
        # Missing residue is between current and next sequence numbers
        if current_num < missing_seq < next_num:
            return True
        # Missing residue has same sequence number as current but different insertion code
        # (e.g., current is 100A, missing is 100B or 100)
        if missing_seq == current_num and missing_ins != current_ins:
            return True
        # Missing residue has same sequence number as next but different insertion code
        # (e.g., next is 101, missing is 101A)
        if missing_seq == next_num and missing_ins != next_ins:
            return True
    
    return False


def identify_gaps(fixer: PDBFixer, chain_id: str) -> list:
    """
    Identify gaps (missing residues) in a chain, handling edge cases:
    - Insertion codes (e.g., 100A → 100B)
    - Negative residue numbers
    - HETATM interruptions
    
    Args:
        fixer: PDBFixer instance with missing residues already identified
        chain_id: Chain ID to process
        
    Returns:
        List of residue indices where gaps occur (positions to split after)
    """
    # Find the target chain
    target_chain = None
    for chain in fixer.topology.chains():
        if chain.id == chain_id:
            target_chain = chain
            break
    
    if target_chain is None:
        return []
    
    # Get all residues in the chain
    residues = list(target_chain.residues())
    if len(residues) < 2:
        return []
    
    # Build a set of missing residue identifiers for quick lookup
    # This correctly parses PDBFixer's missingResidues dictionary structure
    # Even if missing_residues is empty, we still check for structural gaps
    # (out-of-order residues, distance-based gaps)
    missing_residue_ids = build_missing_residue_ids(fixer, target_chain)
    
    # Identify gap positions: where we have a jump in residue sequence numbers
    gap_positions = []
    for i in range(len(residues) - 1):
        current_res = residues[i]
        next_res = residues[i + 1]
        
        try:
            # Parse residue IDs handling insertion codes and negative numbers
            current_seq = parse_residue_id(current_res.id)
            next_seq = parse_residue_id(next_res.id)
            
            # Check if there's a gap in sequence numbers
            seq_diff = next_seq[0] - current_seq[0]
            current_num, current_ins = current_seq
            next_num, next_ins = next_seq
            
            # Check if there are missing residues between current and next
            has_missing_between = check_missing_residues_between(
                current_seq, next_seq, missing_residue_ids
            )
            
            # Calculate distance between residues for additional validation
            # This helps catch gaps even when sequence numbers are consecutive but structure is broken
            try:
                current_atoms = list(current_res.atoms())
                next_atoms = list(next_res.atoms())
                if current_atoms and next_atoms:
                    current_ca = next((a for a in current_atoms if a.name == 'CA'), current_atoms[0])
                    next_ca = next((a for a in next_atoms if a.name == 'CA'), next_atoms[0])
                    current_pos = fixer.positions[current_ca.index]
                    next_pos = fixer.positions[next_ca.index]
                    dist = math.sqrt(
                        sum((current_pos[j].value_in_unit(unit.angstrom) - 
                             next_pos[j].value_in_unit(unit.angstrom))**2 
                            for j in range(3))
                    )
                else:
                    dist = None
            except Exception:
                dist = None
            
            # Flag gaps based on sequence jumps, explicit missing residues, or large distances
            if seq_diff > 1:
                # Large sequence jump (>1) - always flag as gap
                # This handles cases where PDBFixer didn't detect missing residues but gaps exist
                gap_positions.append(i + 1)  # Split after residue i
            elif seq_diff == 1:
                # Consecutive sequence numbers - flag as gap if:
                # 1. There are explicit missing residues, OR
                # 2. Distance is very large (>8Å) indicating structural break
                if has_missing_between or (dist is not None and dist > 8.0):
                    gap_positions.append(i + 1)
            elif seq_diff == 0:
                # Same sequence number - check if there are missing residues with different insertion codes
                # Only flag if there are actual missing residues, not just because insertion codes differ
                # (e.g., 100A → 100B is normal if no residues are missing)
                for missing_seq, missing_ins in missing_residue_ids:
                    if missing_seq == current_num:
                        # Check if missing residue falls between current and next in insertion code order
                        if current_ins is not None and next_ins is not None:
                            # Both have insertion codes - check ordering
                            if compare_residue_sequence((current_num, current_ins), (missing_seq, missing_ins)) < 0 and \
                               compare_residue_sequence((missing_seq, missing_ins), (next_num, next_ins)) < 0:
                                gap_positions.append(i + 1)
                                break
                        elif current_ins is None and next_ins is not None:
                            # Current has no insertion code, next does - missing should be between
                            if missing_ins is not None:
                                gap_positions.append(i + 1)
                                break
                        elif current_ins is not None and next_ins is None:
                            # Current has insertion code, next doesn't - this shouldn't happen with same seq_num
                            # But if it does and there's a missing residue, flag it
                            if missing_ins is None or missing_ins != current_ins:
                                gap_positions.append(i + 1)
                                break
            elif seq_diff < 0:
                # Negative sequence difference means next_num < current_num (out-of-order)
                # This indicates a structural discontinuity/chain break - flag as gap
                # Examples: 100 → 50 (out-of-order), -3 → -5 (negative out-of-order)
                gap_positions.append(i + 1)
            
        except (TypeError, IndexError, AttributeError, ValueError):
            # If we can't determine sequence numbers, try distance-based detection
            try:
                current_atoms = list(current_res.atoms())
                next_atoms = list(next_res.atoms())
                
                if current_atoms and next_atoms:
                    # Get CA atoms if available, otherwise first atom
                    current_ca = next((a for a in current_atoms if a.name == 'CA'), current_atoms[0])
                    next_ca = next((a for a in next_atoms if a.name == 'CA'), next_atoms[0])
                    
                    current_pos = fixer.positions[current_ca.index]
                    next_pos = fixer.positions[next_ca.index]
                    
                    # Calculate distance (in Angstroms)
                    dist = math.sqrt(
                        sum((current_pos[i].value_in_unit(unit.angstrom) - 
                             next_pos[i].value_in_unit(unit.angstrom))**2 
                            for i in range(3))
                    )
                    # If distance is very large (>20 Angstroms), likely a gap
                    # Normal peptide bond CA-CA distance is ~3.8 Angstroms
                    if dist > 8.0:
                        gap_positions.append(i + 1)
            except Exception:
                # If all else fails, skip this residue pair
                continue
    
    return gap_positions


def write_pdb_with_chain_breaks(
    fixer: PDBFixer,
    output_path: str,
    chain_id: str,
    gap_positions: list,
    logger=None
):
    """
    Write PDB file with explicit TER records at chain gaps to avoid artificial bonding.
    
    TER placement rules:
    - TER must follow the last atom of the residue before the gap
    - TER must appear before the next residue begins
    - Chain ID formatting is preserved (same chain ID, TER creates logical break)
    
    Note on chain IDs: This method keeps the same chain ID throughout and inserts TER
    records at gaps. This is sufficient to prevent artificial bonding in most parsers.
    The TER record signals the end of a continuous segment, effectively creating a
    logical chain break without changing the chain ID. This approach is compatible with
    downstream tools that expect consistent chain IDs.
    
    Args:
        fixer: PDBFixer instance
        output_path: Path to write the output PDB file
        chain_id: Chain ID being processed (same ID is preserved throughout)
        gap_positions: List of residue indices where gaps occur
        logger: Optional logger instance for warnings
    """
    if not gap_positions:
        # No gaps, use standard PDB writing
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return
    
    # Find the target chain and get residues
    target_chain = None
    for chain in fixer.topology.chains():
        if chain.id == chain_id:
            target_chain = chain
            break
    
    if target_chain is None:
        # Chain not found, use standard writing
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return
    
    residues = list(target_chain.residues())
    if len(residues) < 2:
        # No gaps possible
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return
    
    # Build set of residue identifiers that come after gaps
    # gap_positions contains indices where gaps occur (positions to split after)
    # So gap_idx is the index of the residue that comes AFTER the gap
    gap_residue_ids: Set[Tuple[int, Optional[str]]] = set()
    for gap_idx in sorted(set(gap_positions)):  # Remove duplicates and sort for consistency
        if 0 < gap_idx < len(residues):
            try:
                res = residues[gap_idx]
                seq_num, ins_code = parse_residue_id(res.id)
                if seq_num is not None:
                    gap_residue_ids.add((seq_num, ins_code))
            except (TypeError, IndexError, AttributeError, ValueError):
                continue
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_file:
        temp_path = temp_file.name
        PDBFile.writeFile(fixer.topology, fixer.positions, temp_file)
    
    try:
        # Read the temporary PDB file
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        # Insert TER records at gap positions
        output_lines = []
        last_residue_id: Optional[Tuple[int, Optional[str]]] = None
        last_residue_chain: Optional[str] = None
        
        for line in lines:
            if line.startswith('ATOM  ') or line.startswith('HETATM'):
                # Parse residue ID from PDB line (handles insertion codes and negative numbers)
                seq_num, ins_code, line_chain = parse_pdb_residue_id(line)
                
                if line_chain == chain_id and seq_num is not None:
                    current_residue_id = (seq_num, ins_code)
                    
                    # Check if we're starting a new residue
                    if current_residue_id != last_residue_id:
                        # If this residue comes after a gap, insert TER before it
                        if last_residue_id is not None and current_residue_id in gap_residue_ids:
                            # Insert TER after the last atom of the previous residue
                            # Find the last ATOM/HETATM line for the previous residue
                            ter_inserted = False
                            for j in range(len(output_lines) - 1, -1, -1):
                                if output_lines[j].startswith(('ATOM  ', 'HETATM')):
                                    # Verify this is still the same residue
                                    prev_seq, prev_ins, prev_chain = parse_pdb_residue_id(output_lines[j])
                                    if prev_chain == chain_id and (prev_seq, prev_ins) == last_residue_id:
                                        # Insert TER after this line (last atom of previous residue)
                                        ter_line = _format_ter_line(output_lines[j])
                                        output_lines.insert(j + 1, ter_line)
                                        ter_inserted = True
                                        break
                            
                            if not ter_inserted and logger:
                                # Fallback: insert TER at the end of output_lines
                                # This shouldn't happen, but handle gracefully
                                logger.warning(
                                    f"Could not find insertion point for TER before residue {seq_num}"
                                )
                        
                        last_residue_id = current_residue_id
                        last_residue_chain = line_chain
                
                output_lines.append(line)
            elif line.startswith('TER'):
                # Keep existing TER records (they may be from original structure)
                output_lines.append(line)
                # Reset tracking after TER
                last_residue_id = None
                last_residue_chain = None
            else:
                # Keep all other lines (HEADER, REMARK, END, etc.)
                output_lines.append(line)
        
        # Write the modified PDB file
        with open(output_path, 'w') as f:
            f.writelines(output_lines)
            
    finally:
        # Clean up temporary file
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _format_ter_line(atom_line: str) -> str:
    """
    Format a TER record using fields from the last atom line in the residue.
    This improves compatibility with parsers that expect full TER records.
    """
    serial_text = atom_line[6:11].strip() if len(atom_line) >= 11 else ""
    try:
        serial = int(serial_text) + 1
    except ValueError:
        serial = 1

    resname = atom_line[17:20].strip() if len(atom_line) >= 20 else "UNK"
    if not resname:
        resname = "UNK"

    chain = atom_line[21] if len(atom_line) > 21 else " "

    resseq_text = atom_line[22:26].strip() if len(atom_line) >= 26 else ""
    try:
        resseq = f"{int(resseq_text):4d}"
    except ValueError:
        resseq = "   1"

    icode = atom_line[26] if len(atom_line) > 26 else " "
    return f"TER   {serial:5d}      {resname:>3} {chain}{resseq}{icode}\n"
