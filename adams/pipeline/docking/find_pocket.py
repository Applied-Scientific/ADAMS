"""
Description:
    Cluster docking coordinates from CSV
"""

import os

import pandas as pd
from sklearn.cluster import DBSCAN

from ...logger_utils import get_logger, log_step_execution
from ..file_organization import setup_docking_dirs
from .utils import write_cluser_centers_pdb


class FindPocket:
    def __init__(
        self, input_file: str, affinity_cutoff: float = -4, out_path: str = "output"
    ):
        r"""
        Args:
            input_file: str: Path to the CSV file containing docking coordinates (e.g., best_search_docking_centers.csv)
            affinity_cutoff: float: Affinity cutoff to filter docking poses (default: -4)
            out_path: str: Output directory (default: ./output)
        """
        self.logger = get_logger()

        # Validate input_file
        if not input_file or not input_file.strip():
            raise ValueError(
                "input_file cannot be empty. FindPocket requires a valid CSV file path from search docking results.\n"
                "Tip: Make sure search docking completed successfully and returned a valid output path."
            )

        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Input file not found: {input_file}\n"
                f"FindPocket requires the output CSV from search docking (e.g., best_search_docking_centers.csv).\n"
                f"Tip: Check that search docking completed successfully and the output file exists."
            )

        self.input_file = input_file
        self.affinity_cutoff = affinity_cutoff
        self.out_path = out_path

        # Set up organized directory structure (search mode since FindPocket is used after search)
        self.dir_structure = setup_docking_dirs(out_path, mode="search")

    def run(self):
        r"""
        Runs the find pocket pipeline.
        """
        step_logger = log_step_execution("Find Pocket", self.logger)
        with step_logger:
            df = pd.read_csv(self.input_file)
            self.logger.info(f"Loaded {len(df)} rows from {self.input_file}")

            # Make sure output directory exists
            os.makedirs(self.dir_structure["summaries"], exist_ok=True)

            # Run clustering
            with step_logger.timing("clustering"):
                cluster_df = self._DBSCAN_coords_clustering(df, self.affinity_cutoff)
                cluster_df.to_csv(
                    os.path.join(
                        self.dir_structure["summaries"], "dock_sites_clustered.csv"
                    ),
                    index=False,
                )

            with step_logger.timing("analysis"):
                cluster_summary = self._cluster_centroids(cluster_df)
                cluster_summary.to_csv(
                    os.path.join(
                        self.dir_structure["summaries"], "cluster_summary.csv"
                    ),
                    index=False,
                )

                self.logger.debug(f"Cluster summary:\n{cluster_summary.head()}")
                write_cluser_centers_pdb(
                    cluster_summary,
                    os.path.join(
                        self.dir_structure["summaries"], "dock_sites_clustered.pdb"
                    ),
                )

    def _DBSCAN_coords_clustering(self, df, affinity_cutoff):
        r"""
        DBSCAN clustering of docking coordinates.
        Args:
            df: pd.DataFrame: DataFrame containing docking coordinates
            affinity_cutoff: float: Affinity cutoff to filter docking poses
        Returns:
            df_extract: pd.DataFrame: DataFrame containing clustered docking coordinates
        """

        df_extract = df[df["affinity"] < affinity_cutoff].copy()
        # === Step 2: Extract coordinates ===
        coords = df_extract[["COM_x", "COM_y", "COM_z"]].values

        # === Step 3: Run clustering (DBSCAN) ===
        # eps = max distance between points in the same neighborhood
        # min_samples = min number of points to form a cluster core (set = 1 to allow singletons)
        db = DBSCAN(eps=5.0, min_samples=1).fit(coords)

        # === Step 4: Assign cluster labels ===
        df_extract["cluster_id"] = db.labels_

        return df_extract

    def _cluster_centroids(self, df):
        r"""
        Cluster centroids of docking coordinates.
        Args:
            df: pd.DataFrame: DataFrame containing clustered docking coordinates
        Returns:
            cluster_summary: pd.DataFrame: DataFrame containing cluster centroids
        """
        # === Step 1: Group by cluster and compute centroid ===
        cluster_summary = (
            df.groupby("cluster_id")
            .agg(
                centroid_x=("COM_x", "mean"),
                centroid_y=("COM_y", "mean"),
                centroid_z=("COM_z", "mean"),
                cluster_size=("cluster_id", "size"),
                mean_affinity=("affinity", "mean"),  # <-- mean affinity
                best_affinity=(
                    "affinity",
                    "min",
                ),  # <-- strongest binding (most negative)
            )
            .reset_index()
        )
        return cluster_summary
