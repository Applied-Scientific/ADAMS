"""
Setup script for ADAMS (Agent-Driven Autonomous Molecular Simulations)
"""

from setuptools import find_packages, setup

setup(
    name="adams",
    version="0.1.0",
    description="Agent-Driven Autonomous Molecular Simulations",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # Runtime prompt/reference assets and templates loaded from disk
        "adams": [
            "*.md",
            "pipeline/*.md",
            "pipeline/references/*.md",
            "pipeline/*/*_prompt.md",
            "helper_agents/*/*_prompt.md",
            "pipeline/md_analysis/mdp/*.mdp",
            # Bundled Vina-GPU artifacts (executable + OpenCL kernels)
            "pipeline/docking/vina_gpu/*",
        ]
    },
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "acpype",
        "meeko==0.7.1",
        "mol-kit==0.0.1",
        "vina==1.2.7",
        "openai-agents",
        "gemmi",
        "prompt_toolkit",
        "psutil",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "adams=adams.cli:main",
        ],
    },
)
