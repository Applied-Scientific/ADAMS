# Use micromamba for a fast and efficient base image with Conda support
FROM mambaorg/micromamba:latest

# Switch to root to install system dependencies
USER root

# Install system dependencies required for building and running scientific tools
# git: for pip install git+... if needed
# build-essential, cmake: for compiling extensions or tools if needed (e.g. gromacs)
# procps: for psutil
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    procps \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the micromamba user
USER $MAMBA_USER

# Install Conda dependencies into the 'base' environment
# We explicitly install Python 3.12 as requested
# Note: ambertools removed due to lack of ARM64 support on conda-forge
RUN micromamba install -y -n base -c conda-forge \
    python=3.12 \
    rdkit \
    openbabel \
    openmm \
    pdbfixer \
    pdb2pqr \
    propka \
    jupyter \
    ipykernel \
    ipywidgets \
    vina \
    gromacs \
    && micromamba clean --all --yes

# Workaround for GROMACS 2024.x activation script bug which fails if no completion file exists
# We remove the broken activation script so that the environment can be activated successfully.
RUN find $MAMBA_ROOT_PREFIX/etc/conda/activate.d -name "*gromacs*" -delete || true

# Activate the base environment for all subsequent commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Set the working directory
WORKDIR /app

# Copy the application code
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Install the Python package and its dependencies
# --no-cache-dir to keep the image small
RUN pip install . --no-cache-dir

# Create directory for agent data and ensure it's writable
RUN mkdir -p agent_data && chown $MAMBA_USER:$MAMBA_USER agent_data

# Expose Port
EXPOSE 8888

# Set the entrypoint to the installed TUI tool
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "adams"]
