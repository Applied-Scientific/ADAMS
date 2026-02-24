# Release Notes

## 2.0 (Candidate) — 2026-02-21

### Docking Reliability

- CPU Vina now retries once with an expanded box when Vina reports `ligand is outside the grid box`.
- Docking pose collection now writes structured failure reports to:
  - `docking/*/summaries/failed_combinations.csv`
  - `docking/*/metadata/failed_combinations.csv`
- Run logs now include a docking-combination summary:
  - expected combinations
  - combinations with pose files
  - combinations with valid affinities
  - failures and partial pose files

### Output Readability

- Production runs now generate readable pose file copies in:
  - `docking/production/poses_named/`
- Manifest now includes `Pose_PDBQT_File_Named` for direct mapping to readable pose files.

### Receptor Preparation

- Full TER line formatting is enforced for chain-break insertion (serial/resname/chain/resseq/icode fields), improving compatibility with strict PDB parsers.

### Known Limitations

- PDB2PQR/PROPKA warnings can still be numerous for gapped or strained membrane-protein inputs.
- RDKit can still report valence/kekulization warnings for rejected generated microstates.
- Pose files from backend executables remain index-based originals (`ligand_{idx}_...`); human-readable copies are generated in `poses_named`.
