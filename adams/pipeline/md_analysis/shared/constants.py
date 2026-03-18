"""
Shared constants for GROMACS helpers.
"""

from bisect import bisect_right

# ---------------------------------------------------------------------------
# 5-smooth numbers (prime factors 2, 3, 5 only) up to 256.
# GROMACS domain decomposition works best with these ranks counts.
# ---------------------------------------------------------------------------
GROMACS_FRIENDLY_RANKS = (
    1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32,
    36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120,
    125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256,
)


def get_gromacs_friendly_ranks(n: int) -> int:
    """Find the largest 5-smooth number <= *n* for optimal GROMACS decomposition."""
    if n <= 1:
        return 1
    if n >= 256:
        return 256
    return GROMACS_FRIENDLY_RANKS[bisect_right(GROMACS_FRIENDLY_RANKS, n) - 1]


# Ligand residue name used by ACPYPE and referenced in MDP/index.
LIGAND_RESNAME = "LIG"
# Name of ligand heavy-atom index group in index_<resname>.ndx.
LIG_HEAVY_GROUP_NAME = "LIG_heavy"

# Thermostat coupling index groups: must exist in index.ndx for NVT/NPT/MD (tc-grps in MDP).
# Created by make_system_index(); use these names in MDP tc-grps and in validation.
NDX_GROUP_PROTEIN_LIG = "Protein_LIG"
NDX_GROUP_WATER_IONS = "Water_and_ions"
REQUIRED_NDX_GROUPS_FOR_TC = (NDX_GROUP_PROTEIN_LIG, NDX_GROUP_WATER_IONS)

# Common solvent/ion residue names for Water_and_ions index group.
# Includes aliases seen in CHARMM-GUI and mixed force-field exports.
WATER_ION_RESNAMES = (
    "SOL", "WAT", "HOH", "TIP3", "TIP4", "TIP5", "SPC", "SPCE", "OPC",
    "NA", "SOD", "NA+", "CL", "CLA", "CL-", "K", "POT", "K+",
    "MG", "MG2", "CA", "CAL", "ZN", "CES",
)

# Lipid residue names used to auto-detect membrane lipids in GRO/TOP files.
LIPID_RESNAMES = frozenset({
    # Phosphatidylcholines
    "POPC", "DPPC", "DMPC", "DOPC", "DLPC", "DSPC", "DAPC", "SOPC", "PLPC", "PAPC",
    "SDPC", "PFPC",
    # Phosphatidylethanolamines
    "POPE", "DPPE", "DMPE", "DOPE", "DLPE", "DSPE",
    # Phosphatidylserines
    "POPS", "DOPS",
    # Phosphatidylglycerols
    "POPG", "DOPG",
    # Phosphatidylinositols
    "POPI", "DOPI",
    # Sphingomyelins
    "SM", "PSM", "SSM",
    # Cholesterol
    "CHL1", "CHOL",
    # Ceramides
    "CER", "DPCE",
    # Cardiolipin components
    "CDL0", "CDL1", "CDL2",
    # Amber lipid naming (Lipid17/21)
    "PA", "PC", "PE", "PG", "PI", "PS", "OL", "MY", "ST", "AR",
})

# Ion residue names for membrane systems.
ION_RESNAMES = frozenset({
    "NA", "SOD", "NA+",
    "CL", "CLA", "CL-",
    "K", "POT", "K+",
    "MG", "MG2",
    "CA", "CAL",
    "ZN", "CES",
})

# Force constants for staged position restraint release (kJ/mol/nm^2).
RESTRAINT_FC_VALUES = (500.0, 200.0)
