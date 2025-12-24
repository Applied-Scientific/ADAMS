#!/bin/bash
# detect.sh - Platform and GPU detection for ADAMS installation

# ============================================================================
# Platform detection
# ============================================================================

detect_platform() {
    OS_TYPE=$(uname -s)
    ARCH_TYPE=$(uname -m)

    case "$OS_TYPE" in
        Linux)
            PLATFORM="linux"
            ;;
        Darwin)
            PLATFORM="macos"
            ;;
        *)
            PLATFORM="unknown"
            ;;
    esac

    case "$ARCH_TYPE" in
        x86_64|amd64)
            ARCH="x86_64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        *)
            ARCH="unknown"
            ;;
    esac

    export PLATFORM ARCH OS_TYPE ARCH_TYPE
}

# ============================================================================
# GPU detection (NVIDIA only)
# ============================================================================

detect_gpu() {
    GPU_AVAILABLE=false
    GPU_MODEL=""
    CUDA_DRIVER_VERSION=""
    CUDA_TOOLKIT_AVAILABLE=false
    CUDA_TOOLKIT_VERSION=""

    # Only check for GPU on Linux
    if [[ "$PLATFORM" != "linux" ]]; then
        export GPU_AVAILABLE GPU_MODEL CUDA_DRIVER_VERSION CUDA_TOOLKIT_AVAILABLE CUDA_TOOLKIT_VERSION
        return 0
    fi

    # Check for nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        # Get GPU info
        local nvidia_output
        nvidia_output=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)

        if [[ -n "$nvidia_output" ]]; then
            GPU_AVAILABLE=true
            GPU_MODEL=$(echo "$nvidia_output" | cut -d',' -f1 | xargs)
            CUDA_DRIVER_VERSION=$(echo "$nvidia_output" | cut -d',' -f2 | xargs)
        fi
    fi

    # Check for CUDA toolkit (nvcc)
    if command -v nvcc &> /dev/null; then
        CUDA_TOOLKIT_AVAILABLE=true
        CUDA_TOOLKIT_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    fi

    export GPU_AVAILABLE GPU_MODEL CUDA_DRIVER_VERSION CUDA_TOOLKIT_AVAILABLE CUDA_TOOLKIT_VERSION
}

# ============================================================================
# OpenCL detection
# ============================================================================

detect_opencl() {
    OPENCL_AVAILABLE=false
    OPENCL_PLATFORMS=0
    OPENCL_NVIDIA_ICD_MISSING=false

    # Only check on Linux
    if [[ "$PLATFORM" != "linux" ]]; then
        export OPENCL_AVAILABLE OPENCL_PLATFORMS OPENCL_NVIDIA_ICD_MISSING
        return 0
    fi

    # Check for OpenCL libraries
    if ldconfig -p 2>/dev/null | grep -q "libOpenCL.so"; then
        OPENCL_AVAILABLE=true
    fi

    # Check for clinfo to detect platforms
    if command -v clinfo &> /dev/null; then
        OPENCL_PLATFORMS=$(clinfo 2>/dev/null | grep "Number of platforms" | awk '{print $NF}' | head -1)
        if [[ -z "$OPENCL_PLATFORMS" ]]; then
            OPENCL_PLATFORMS=0
        fi
    fi

    # Check if NVIDIA ICD file is missing (when GPU is available but no OpenCL platforms detected)
    if [[ "$GPU_AVAILABLE" == true ]] && [[ "$OPENCL_PLATFORMS" == "0" ]]; then
        # Check if NVIDIA OpenCL library exists but ICD file is missing
        if ldconfig -p 2>/dev/null | grep -q "libnvidia-opencl.so"; then
            if [[ ! -f /etc/OpenCL/vendors/nvidia.icd ]]; then
                OPENCL_NVIDIA_ICD_MISSING=true
            fi
        fi
    fi

    export OPENCL_AVAILABLE OPENCL_PLATFORMS OPENCL_NVIDIA_ICD_MISSING
}

# ============================================================================
# Repository detection
# ============================================================================

detect_repo() {
    IN_REPO=false
    REPO_ROOT=""

    # Check if we're in a cloned repo by looking for setup.py
    local current_dir="$PWD"
    local check_dir="$current_dir"

    # Walk up the directory tree looking for setup.py and adams/
    for _ in {1..5}; do
        if [[ -f "$check_dir/setup.py" ]] && [[ -d "$check_dir/adams" ]]; then
            IN_REPO=true
            REPO_ROOT="$check_dir"
            break
        fi
        check_dir=$(dirname "$check_dir")
        if [[ "$check_dir" == "/" ]]; then
            break
        fi
    done

    # Also check if SCRIPT_DIR is set and points to a repo
    if [[ "$IN_REPO" == false ]] && [[ -n "$SCRIPT_DIR" ]]; then
        check_dir=$(dirname "$SCRIPT_DIR")
        if [[ -f "$check_dir/setup.py" ]] && [[ -d "$check_dir/adams" ]]; then
            IN_REPO=true
            REPO_ROOT="$check_dir"
        fi
    fi

    export IN_REPO REPO_ROOT
}

# ============================================================================
# Print detection summary
# ============================================================================

print_detection_summary() {
    echo ""
    echo "========================================"
    echo "  System Detection Summary"
    echo "========================================"
    echo ""
    echo "  Platform:     $PLATFORM ($ARCH)"
    echo "  Conda:        ${CONDA_CMD:-not found} ${CONDA_VERSION:+($CONDA_VERSION)}"

    if [[ "$PLATFORM" == "linux" ]]; then
        if [[ "$GPU_AVAILABLE" == true ]]; then
            echo "  GPU:          $GPU_MODEL"
            echo "  CUDA Driver:  $CUDA_DRIVER_VERSION"
            if [[ "$CUDA_TOOLKIT_AVAILABLE" == true ]]; then
                echo "  CUDA Toolkit: $CUDA_TOOLKIT_VERSION"
            else
                echo "  CUDA Toolkit: not found"
            fi

            # OpenCL status (relevant for GPU docking)
            if [[ "$OPENCL_AVAILABLE" == true ]]; then
                if [[ "$OPENCL_PLATFORMS" -gt 0 ]]; then
                    echo "  OpenCL:       $OPENCL_PLATFORMS platform(s) detected"
                elif [[ "$OPENCL_NVIDIA_ICD_MISSING" == true ]]; then
                    echo "  OpenCL:       installed but not visible (will be fixed)"
                else
                    echo "  OpenCL:       installed (0 platforms)"
                fi
            else
                echo "  OpenCL:       not found"
            fi
        else
            echo "  GPU:          not detected"
        fi
    elif [[ "$PLATFORM" == "macos" ]]; then
        echo "  GPU:          N/A (CPU-only on macOS)"
    fi

    if [[ "$IN_REPO" == true ]]; then
        echo "  Repository:   $REPO_ROOT"
    else
        echo "  Repository:   not in cloned repo"
    fi

    echo ""
    echo "========================================"
    echo ""
}

# ============================================================================
# Run all detection
# ============================================================================

run_detection() {
    detect_platform
    detect_gpu
    detect_opencl
    detect_repo
}
