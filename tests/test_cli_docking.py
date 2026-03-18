import argparse

from adams.cli_docking import add_dock_subparser


def test_add_dock_subparser_parses_minimal_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_dock_subparser(subparsers)

    args = parser.parse_args([
        "dock",
        "--receptor", "/tmp/receptor.pdb",
        "--ligand-folder", "/tmp/ligands",
        "--outdir", "/tmp/out",
        "--docking-centers", "1,2,3",
    ])

    assert args.command == "dock"
    assert args.receptor == "/tmp/receptor.pdb"
    assert args.ligand_folder == "/tmp/ligands"
    assert args.backend == "vina_gpu"
    assert args.num_poses == 20
