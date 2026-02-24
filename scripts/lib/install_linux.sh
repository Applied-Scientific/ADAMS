#!/bin/bash
# install_linux.sh - Linux-specific installation for ADAMS

ENV_NAME="adams"
GROMACS_VERSION="2024.4"
GITHUB_REPO="https://github.com/Applied-Scientific/ADAMS.git"

# ============================================================================
# Step 1: Environment setup
# ============================================================================

install_step_env() {
    step "Step 1: Setting up conda environment"

    if env_exists "$ENV_NAME"; then
        warn "Environment '$ENV_NAME' already exists"
        choose_rda "What would you like to do?"
        local choice=$?

        case $choice in
            0)  # Reuse
                info "Reusing existing environment"
                ;;
            1)  # Delete and recreate
                remove_env "$ENV_NAME" || return 1
                create_env "$ENV_NAME" || return 1
                ;;
            2)  # Abort
                error "Installation aborted by user"
                exit 1
                ;;
        esac
    else
        create_env "$ENV_NAME" || return 1
    fi

    activate_env "$ENV_NAME" || {
        error "Failed to activate environment"
        return 1
    }

    # Verify Python version
    local python_version=$(python --version 2>&1 | awk '{print $2}')
    if [[ ! "$python_version" =~ ^3\.12 ]]; then
        error "Wrong Python version: $python_version (expected 3.12.x)"
        error "Environment activation may have failed"
        return 1
    fi

    success "Environment '$ENV_NAME' is ready (Python $python_version)"
    mark_step_done "env"
}

# ============================================================================
# Helper: Ensure we're in the correct environment
# ============================================================================

ensure_env_active() {
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
        warn "Not in $ENV_NAME environment, attempting to activate..."
        activate_env "$ENV_NAME" || {
            error "Failed to activate $ENV_NAME environment"
            return 1
        }
    fi

    # Double-check Python version
    local python_version=$(python --version 2>&1 | awk '{print $2}')
    if [[ ! "$python_version" =~ ^3\.12 ]]; then
        error "Wrong Python version: $python_version (expected 3.12.x)"
        error "Please ensure you're in the '$ENV_NAME' conda environment"
        return 1
    fi
}

# ============================================================================
# Step 2: Conda dependencies
# ============================================================================

install_step_conda_deps() {
    step "Step 2: Installing conda dependencies"
    ensure_env_active || return 1

    info "Installing rdkit, openbabel, openmm, pdbfixer, pdb2pqr, propka, jupyter, ipykernel, ipywidgets..."
    run_cmd "Installing molecular simulation packages" \
        "$CONDA_CMD install -y -c conda-forge rdkit openbabel openmm pdbfixer pdb2pqr propka jupyter ipykernel ipywidgets" || {
        error "Failed to install conda dependencies"
        echo ""
        echo "Suggestions:"
        echo "  - Check your internet connection"
        echo "  - Try: $CONDA_CMD install -y -c conda-forge rdkit openbabel openmm pdbfixer pdb2pqr propka jupyter ipykernel ipywidgets"
        if confirm_no "Continue anyway?"; then
            warn "Continuing without some dependencies..."
        else
            return 1
        fi
    }

    success "Conda dependencies installed"
    mark_step_done "conda_deps"
}

# ============================================================================
# Step 3: AmberTools
# ============================================================================

install_step_ambertools() {
    step "Step 3: Installing AmberTools"
    ensure_env_active || return 1

    run_cmd "Installing AmberTools" \
        "$CONDA_CMD install -y conda-forge::ambertools" || {
        error "Failed to install AmberTools"
        echo ""
        echo "Suggestions:"
        echo "  - Check your internet connection"
        echo "  - Try: $CONDA_CMD install -y conda-forge::ambertools"
        if confirm_no "Continue anyway?"; then
            warn "Continuing without AmberTools..."
        else
            return 1
        fi
    }

    # Verify installation
    if command -v antechamber &> /dev/null; then
        success "AmberTools installed (antechamber found)"
    else
        warn "AmberTools installed but antechamber not in PATH"
    fi

    mark_step_done "ambertools"
}

# ============================================================================
# Step 4: GROMACS
# ============================================================================

install_gromacs_cpu() {
    info "Installing GROMACS (CPU version) via conda..."
    run_cmd "Installing GROMACS" \
        "$CONDA_CMD install -y conda-forge::gromacs" || {
        error "Failed to install GROMACS"
        return 1
    }
    success "GROMACS CPU version installed"
}

install_gromacs_gpu() {
    info "Installing GROMACS with CUDA support (compiling from source)..."
    echo "This may take 10-30 minutes depending on your system."
    echo ""

    # Check prerequisites
    if ! command -v cmake &> /dev/null; then
        error "cmake not found"
        echo ""
        echo "Please install cmake first:"
        echo "  Ubuntu/Debian: sudo apt install cmake"
        echo "  Fedora/RHEL:   sudo dnf install cmake"
        echo ""
        if confirm_no "Continue without GROMACS GPU support?"; then
            warn "Skipping GROMACS GPU build. Installing CPU version instead..."
            install_gromacs_cpu
            return $?
        else
            return 1
        fi
    fi

    if ! command -v gcc &> /dev/null; then
        error "gcc not found"
        echo ""
        echo "Please install gcc first:"
        echo "  Ubuntu/Debian: sudo apt install build-essential"
        echo "  Fedora/RHEL:   sudo dnf install gcc gcc-c++"
        echo ""
        if confirm_no "Continue without GROMACS GPU support?"; then
            warn "Skipping GROMACS GPU build. Installing CPU version instead..."
            install_gromacs_cpu
            return $?
        else
            return 1
        fi
    fi

    # Check for CUDA toolkit
    if [[ "$CUDA_TOOLKIT_AVAILABLE" != true ]]; then
        warn "CUDA toolkit not found"
        echo ""
        echo "You can install it via conda:"
        if confirm_yes "Install CUDA toolkit via conda?"; then
            run_cmd "Installing CUDA toolkit" \
                "$CONDA_CMD install -y cuda-cudart=12.2 cuda-version=12.2" || {
                error "Failed to install CUDA toolkit"
                if confirm_no "Continue without GROMACS GPU support?"; then
                    install_gromacs_cpu
                    return $?
                else
                    return 1
                fi
            }
        else
            warn "Skipping GROMACS GPU build. Installing CPU version instead..."
            install_gromacs_cpu
            return $?
        fi
    fi

    # Create temp directory for build
    local original_dir=$(pwd)
    local build_dir=$(mktemp -d)
    local gromacs_log="$HOME/adams_install_gromacs.log"

    info "Downloading GROMACS ${GROMACS_VERSION}..."
    cd "$build_dir" || return 1

    if ! wget -q "https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz"; then
        error "Failed to download GROMACS"
        rm -rf "$build_dir"
        return 1
    fi

    info "Extracting..."
    tar xfz "gromacs-${GROMACS_VERSION}.tar.gz" || {
        error "Failed to extract GROMACS"
        rm -rf "$build_dir"
        return 1
    }

    cd "gromacs-${GROMACS_VERSION}" || return 1
    mkdir -p build && cd build || return 1

    info "Configuring GROMACS with CUDA..."
    cmake .. \
        -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX/gromacs-${GROMACS_VERSION}/cuda" \
        -DGMX_GPU=CUDA \
        -DGMX_MPI=OFF \
        -DGMX_DOUBLE=OFF \
        -DGMX_BUILD_OWN_FFTW=ON \
        > "$gromacs_log" 2>&1 || {
        error "GROMACS configure failed. See $gromacs_log for details."
        rm -rf "$build_dir"
        if confirm_no "Continue without GROMACS GPU support?"; then
            install_gromacs_cpu
            return $?
        else
            return 1
        fi
    }

    info "Compiling GROMACS (this will take a while)..."
    local num_cores=$(nproc)
    make -j"$num_cores" >> "$gromacs_log" 2>&1 || {
        error "GROMACS compile failed. See $gromacs_log for details."
        rm -rf "$build_dir"
        if confirm_no "Continue without GROMACS GPU support?"; then
            install_gromacs_cpu
            return $?
        else
            return 1
        fi
    }

    info "Installing GROMACS..."
    make install >> "$gromacs_log" 2>&1 || {
        error "GROMACS install failed. See $gromacs_log for details."
        rm -rf "$build_dir"
        return 1
    }

    # Clean up
    cd "$original_dir"
    rm -rf "$build_dir"

    # Set up PATH in conda environment
    local activate_dir="$CONDA_PREFIX/etc/conda/activate.d"
    mkdir -p "$activate_dir"

    cat > "$activate_dir/adams_gromacs.sh" << EOF
# ADAMS GROMACS GPU configuration
export PATH="\$CONDA_PREFIX/gromacs-${GROMACS_VERSION}/cuda/bin:\$PATH"
export OMPI_MCA_opal_cuda_support=true
EOF

    # Source it now
    source "$activate_dir/adams_gromacs.sh"

    success "GROMACS GPU version installed"
    info "GROMACS compile log saved to: $gromacs_log"
}

install_step_gromacs() {
    step "Step 4: Installing GROMACS"
    ensure_env_active || return 1

    if [[ "$INSTALL_GPU" == true ]]; then
        install_gromacs_gpu
    else
        install_gromacs_cpu
    fi

    # Verify installation
    if command -v gmx &> /dev/null; then
        local gmx_version=$(gmx --version 2>/dev/null | head -1)
        success "GROMACS installed: $gmx_version"
    else
        warn "GROMACS installed but gmx not in PATH"
        warn "You may need to restart your shell or run: source $CONDA_PREFIX/etc/conda/activate.d/adams_gromacs.sh"
    fi

    mark_step_done "gromacs"
}

# ============================================================================
# Step 4.5: Configure OpenCL for GPU docking (NVIDIA)
# ============================================================================

configure_opencl() {
    step "Configuring OpenCL for GPU docking"

    # Only run if GPU is available and OpenCL needs fixing
    if [[ "$GPU_AVAILABLE" != true ]]; then
        info "No GPU detected, skipping OpenCL configuration"
        return 0
    fi

    if [[ "$OPENCL_NVIDIA_ICD_MISSING" != true ]]; then
        if [[ "$OPENCL_PLATFORMS" -gt 0 ]]; then
            success "OpenCL already configured ($OPENCL_PLATFORMS platform(s) detected)"
        else
            warn "OpenCL installed but no platforms detected (not a known issue)"
        fi
        return 0
    fi

    # Fix the missing NVIDIA ICD file
    info "OpenCL libraries found but no platforms detected"
    info "This is required for AutoDock-Vina-GPU to work"
    echo ""

    if [[ ! -d /etc/OpenCL/vendors ]]; then
        warn "Creating /etc/OpenCL/vendors directory (requires sudo)"
        if ! sudo mkdir -p /etc/OpenCL/vendors; then
            error "Failed to create /etc/OpenCL/vendors directory"
            error "GPU docking (AutoDock-Vina-GPU) will not work without OpenCL"
            if confirm_no "Continue anyway?"; then
                return 0
            else
                return 1
            fi
        fi
    fi

    info "Creating NVIDIA OpenCL vendor file (requires sudo)"
    echo ""
    echo "  This will create: /etc/OpenCL/vendors/nvidia.icd"
    echo "  Contents: libnvidia-opencl.so.1"
    echo ""

    if sudo bash -c 'echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd'; then
        success "Created /etc/OpenCL/vendors/nvidia.icd"

        # Verify the fix worked
        if command -v clinfo &> /dev/null; then
            info "Verifying OpenCL configuration..."
            local platforms=$(clinfo 2>/dev/null | grep "Number of platforms" | awk '{print $NF}' | head -1)
            if [[ -n "$platforms" ]] && [[ "$platforms" -gt 0 ]]; then
                success "OpenCL now detects $platforms platform(s)!"
            else
                warn "OpenCL configured but still showing 0 platforms"
                warn "You may need to restart your shell or reboot"
            fi
        fi
    else
        error "Failed to create NVIDIA ICD file"
        error "GPU docking (AutoDock-Vina-GPU) will not work without this"
        if confirm_no "Continue anyway?"; then
            return 0
        else
            return 1
        fi
    fi
}

# ============================================================================
# Step 4.6: Configure Vina-GPU runtime libraries
# ============================================================================

configure_vina_gpu_runtime() {
    step "Configuring Vina-GPU runtime libraries"
    ensure_env_active || return 1

    local runtime_env="vinagpu_rt"
    local conda_base
    conda_base=$($CONDA_CMD info --base 2>/dev/null)
    if [[ -z "$conda_base" ]]; then
        conda_base="$HOME/miniconda3"
    fi
    local runtime_libdir="$conda_base/envs/${runtime_env}/lib"

    if [[ ! -d "$runtime_libdir" ]]; then
        info "Creating Vina-GPU runtime env '$runtime_env' with Boost 1.83 runtime libs..."
        run_cmd "Creating ${runtime_env} environment" \
            "$CONDA_CMD create -y -n ${runtime_env} -c conda-forge --override-channels libboost=1.83.0 libgcc-ng libstdcxx-ng" || {
            warn "Could not create ${runtime_env} runtime environment."
            warn "GPU docking may fail if Boost 1.83 runtime libs are unavailable."
            return 0
        }
    else
        info "Using existing Vina-GPU runtime environment: $runtime_env"
    fi

    if [[ ! -d "$runtime_libdir" ]]; then
        warn "Vina-GPU runtime library directory not found: $runtime_libdir"
        warn "GPU docking may fail until runtime libraries are installed."
        return 0
    fi

    local activate_dir="$CONDA_PREFIX/etc/conda/activate.d"
    local deactivate_dir="$CONDA_PREFIX/etc/conda/deactivate.d"
    mkdir -p "$activate_dir" "$deactivate_dir"

    cat > "$activate_dir/vinagpu_rt.sh" <<EOF
# Keep global linker state untouched; Vina-GPU backend injects LD_LIBRARY_PATH per subprocess.
export VINA_GPU_LIBDIR="$runtime_libdir"
unset _ADAMS_OLD_LD_LIBRARY_PATH
EOF

    cat > "$deactivate_dir/vinagpu_rt.sh" <<'EOF'
unset VINA_GPU_LIBDIR
unset _ADAMS_OLD_LD_LIBRARY_PATH
EOF

    success "Configured Vina-GPU runtime hooks for '$ENV_NAME'."
    info "Re-activate '$ENV_NAME' to apply VINA_GPU_LIBDIR automatically."
}


# ============================================================================
# Step 4.7: Install UniDock (Linux GPU backend)
# ============================================================================

install_step_unidock() {
    step "Step 4.7: Installing UniDock"
    ensure_env_active || return 1

    if command -v unidock &> /dev/null; then
        local unidock_path
        unidock_path=$(command -v unidock)
        success "UniDock already installed: $unidock_path"
        mark_step_done "unidock"
        return 0
    fi

    info "Installing UniDock from conda-forge..."
    run_cmd "Installing UniDock" \
        "$CONDA_CMD install -y -c conda-forge unidock" || {
        error "Failed to install UniDock from conda-forge."
        echo ""
        echo "UniDock is required for backend='unidock' runs."
        echo "Try manually:"
        echo "  $CONDA_CMD install -y -c conda-forge unidock"
        echo "Or provide an explicit binary at runtime via:"
        echo "  gpu_executable='/absolute/path/to/unidock'"
        return 1
    }

    if ! command -v unidock &> /dev/null; then
        error "UniDock installation completed but 'unidock' is not in PATH."
        error "Re-activate your conda environment and retry."
        return 1
    fi

    success "UniDock installed successfully"
    mark_step_done "unidock"
}


# ============================================================================
# Step 5: Install ADAMS package
# ============================================================================

install_step_package() {
    step "Step 5: Installing ADAMS package"
    ensure_env_active || return 1

    if [[ "$IN_REPO" == true ]]; then
        info "Detected cloned repository at: $REPO_ROOT"
        if confirm_no "Install in editable mode?"; then
            run_cmd "Installing ADAMS (editable mode)" \
                "pip install -e '$REPO_ROOT'" || return 1
        else
            run_cmd "Installing ADAMS" \
                "pip install '$REPO_ROOT'" || return 1
        fi
    else
        run_cmd "Installing ADAMS from GitHub" \
            "pip install git+${GITHUB_REPO}" || return 1
    fi

    success "ADAMS package installed"
    mark_step_done "package"
}

# ============================================================================
# Step 6: Verification
# ============================================================================

verify_installation() {
    step "Verifying installation"
    local all_ok=true

    echo ""
    # Python
    if python --version &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Python $(python --version 2>&1 | cut -d' ' -f2)"
    else
        echo -e "  ${RED}✗${NC} Python"
        all_ok=false
    fi

    # rdkit
    if python -c "import rdkit" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} rdkit"
    else
        echo -e "  ${RED}✗${NC} rdkit"
        all_ok=false
    fi

    # dimorphite_dl (ligand protonation state enumeration)
    if python -c "import dimorphite_dl" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} dimorphite_dl"
    else
        echo -e "  ${RED}✗${NC} dimorphite_dl"
        all_ok=false
    fi

    # openbabel
    if command -v obabel &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} openbabel (obabel)"
    else
        echo -e "  ${RED}✗${NC} openbabel"
        all_ok=false
    fi

    # openmm
    if python -c "import openmm" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} openmm"
    else
        echo -e "  ${RED}✗${NC} openmm"
        all_ok=false
    fi

    # vina
    if python -c "import vina" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} vina"
    else
        echo -e "  ${RED}✗${NC} vina"
        all_ok=false
    fi

    # pdb2pqr (required for protonation)
    if command -v pdb2pqr &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} pdb2pqr"
    else
        echo -e "  ${RED}✗${NC} pdb2pqr"
        all_ok=false
    fi

    # propka3 (required for protonation)
    if command -v propka3 &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} propka3"
    else
        echo -e "  ${RED}✗${NC} propka3"
        all_ok=false
    fi

    # antechamber (AmberTools)
    if command -v antechamber &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} antechamber (AmberTools)"
    else
        echo -e "  ${YELLOW}!${NC} antechamber (AmberTools) - not in PATH"
    fi

    # gmx (GROMACS)
    if command -v gmx &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} gmx (GROMACS)"
    else
        echo -e "  ${YELLOW}!${NC} gmx (GROMACS) - not in PATH"
    fi

    # unidock (required for Linux GPU backend)
    if [[ "$INSTALL_GPU" == true ]]; then
        if command -v unidock &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} unidock"
        else
            echo -e "  ${RED}✗${NC} unidock (required for backend=unidock)"
            all_ok=false
        fi
    else
        if command -v unidock &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} unidock"
        else
            echo -e "  ${YELLOW}!${NC} unidock - not installed (required only for backend=unidock)"
        fi
    fi

    # adams package
    if command -v adams &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} adams package"
    else
        echo -e "  ${RED}✗${NC} adams package"
        all_ok=false
    fi

    echo ""

    if [[ "$all_ok" == true ]]; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# Main installation function
# ============================================================================

run_linux_install() {
    echo ""
    echo "========================================"
    echo "  ADAMS Installation - Linux"
    echo "========================================"

    # Ask about GPU installation
    INSTALL_GPU=false
    if [[ "$GPU_AVAILABLE" == true ]]; then
        echo ""
        info "NVIDIA GPU detected: $GPU_MODEL"
        if confirm_yes "Install with GPU support?"; then
            INSTALL_GPU=true
        fi
    fi

    # Check for resume
    if init_progress; then
        if choose_resume; then
            info "Resuming previous installation..."
        else
            info "Starting fresh installation..."
            clear_progress
        fi
    fi

    # Run installation steps
    is_step_done "env" || install_step_env || exit 1
    is_step_done "conda_deps" || install_step_conda_deps || exit 1
    is_step_done "ambertools" || install_step_ambertools || exit 1
    is_step_done "gromacs" || install_step_gromacs || exit 1

    # Configure OpenCL if GPU installation was selected
    if [[ "$INSTALL_GPU" == true ]]; then
        is_step_done "opencl" || {
            configure_opencl || exit 1
            mark_step_done "opencl"
        }
        is_step_done "vina_gpu_runtime" || {
            configure_vina_gpu_runtime || exit 1
            mark_step_done "vina_gpu_runtime"
        }
        is_step_done "unidock" || install_step_unidock || exit 1
    fi

    is_step_done "package" || install_step_package || exit 1
    is_step_done "extra_forcefields" || install_step_extra_forcefields

    # Verification
    verify_installation

    # Shell integration
    is_step_done "shell_function" || {
        setup_shell_function "$ENV_NAME"
        mark_step_done "shell_function"
    }

    # Test ADAMS command (warm up imports)
    step "Testing ADAMS installation"
    info "Running ADAMS for the first time (this may take a moment)..."
    conda run -n "$ENV_NAME" --no-capture-output python -c "import adams_tui.app" >> "$LOG_FILE" 2>&1 || true
    success "ADAMS is ready to use!"

    # Success message
    echo ""
    echo "========================================"
    echo -e "  ${GREEN}✓ ADAMS installed successfully!${NC}"
    echo "========================================"
    echo ""
    echo "  To get started, run:"
    echo "    adams"
    echo ""
    echo "  Documentation:"
    echo "    https://github.com/Applied-Scientific/ADAMS"
    echo ""
    echo "  Install log saved to: $LOG_FILE"
    echo ""

    # Clean up progress file on success
    clear_progress
}
