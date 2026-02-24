"""
    File Parser Agent - Extracts structured statistics from pipeline outputs
    (docking results CSV) to enable parameter extraction and
    result-based decision making.
"""

from pathlib import Path

from agents import Agent, ModelSettings, function_tool

from ...pipeline.references.reference_file_reader import read_reference_file
from .file_parser_tools import parse_docking_results_impl


@function_tool
def parse_docking_results(csv_path: str) -> dict:
    """
    Parse docking results CSV and extract statistics for parameter decisions.

    This function reads a docking results CSV file (e.g., production_docking_results.csv)
    and extracts comprehensive statistics including affinity metrics, pose counts,
    pocket analysis, and affinity distributions. Use this to extract optimal parameters
    from previous run results or to summarize docking outcomes.

    Use this when:
    - You need to inform parameter choices based on pose counts per ligand
    - You want to analyze affinity distribution to inform parameter decisions
    - You need to identify which pockets had the best results
    - You want to summarize docking results for the user
    - You need to extract statistics to recommend next-step parameters

    Args:
        csv_path (str): Path to docking results CSV file. Can be relative to agent_data
            root or an absolute path. Expected file: production_docking_results.csv or
            similar docking results CSV.
            Example: "outputs/run_20251203/docking/production/summaries/production_docking_results.csv"

    Returns:
        dict: Dictionary containing:
            - 'csv_path' (str): Full path to CSV file parsed
            - 'statistics' (dict): Affinity statistics:
                - 'best_affinity' (float): Minimum (most negative) affinity value
                - 'worst_affinity' (float): Maximum (least negative) affinity value
                - 'avg_affinity' (float): Mean affinity across all poses
                - 'median_affinity' (float): Median affinity value
                - 'affinity_std' (float): Standard deviation of affinities
            - 'counts' (dict): Count statistics:
                - 'total_poses' (int): Total number of poses in CSV
                - 'unique_ligands' (int): Number of unique ligand_id values
                - 'unique_pockets' (int): Number of unique grid_id values
                - 'poses_per_ligand_avg' (float): Average poses per ligand
                - 'poses_per_ligand_min' (int): Minimum poses per ligand
                - 'poses_per_ligand_max' (int): Maximum poses per ligand
            - 'pocket_stats' (dict): Statistics per pocket (grid_id):
                - Maps grid_id to dict with 'count', 'best_affinity', 'avg_affinity'
            - 'top_pockets' (list): List of grid_ids sorted by best affinity (best first)
            - 'affinity_percentiles' (dict): Percentile values (p10, p25, p50, p75, p90)
            - 'affinity_ranges' (dict): Count of poses in affinity ranges:
                - 'very_strong' (int): Count with affinity < -8.0
                - 'strong' (int): Count with affinity -8.0 to -6.0
                - 'moderate' (int): Count with affinity -6.0 to -4.0
                - 'weak' (int): Count with affinity >= -4.0
            - 'error' (str or None): Error message if parsing failed, None if successful

    Example:
        >>> result = parse_docking_results("outputs/run_20251203/docking/production/summaries/production_docking_results.csv")
        >>> # Returns statistics including best affinity, pose counts, pocket analysis
        >>> print(result['statistics']['best_affinity'])  # -8.5
        >>> print(result['counts']['poses_per_ligand_avg'])  # 5.2
        >>> print(result['top_pockets'])  # [0, 2, 1] (pocket 0 has best affinity)
    """
    return parse_docking_results_impl(csv_path)


prompt_path = Path(__file__).parent / "file_parser_prompt.md"
system_prompt = prompt_path.read_text()

file_parser_agent = Agent(
    model="gpt-5-mini",
    name="File Parser Agent",
    instructions=system_prompt,
    tools=[
        read_reference_file,
        parse_docking_results,
    ],
    model_settings=ModelSettings(tool_choice="auto"),
)
