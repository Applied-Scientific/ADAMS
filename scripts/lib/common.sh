#!/bin/bash
# common.sh - Shared utilities for ADAMS installation scripts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking file
PROGRESS_FILE="$HOME/.adams_install_progress"
LOG_FILE="$HOME/adams_install.log"

# ============================================================================
# Logging functions
# ============================================================================

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[!]${NC} $1"
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

step() {
    echo -e "\n${CYAN}==>${NC} $1"
    echo "[STEP] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

# ============================================================================
# User interaction functions
# ============================================================================

# Confirm with default yes
confirm_yes() {
    local prompt="$1"
    local response
    read -r -p "$prompt [Y/n]: " response
    case "$response" in
        [nN][oO]|[nN]) return 1 ;;
        *) return 0 ;;
    esac
}

# Confirm with default no
confirm_no() {
    local prompt="$1"
    local response
    read -r -p "$prompt [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY]) return 0 ;;
        *) return 1 ;;
    esac
}

# Choose from options (R)euse / (D)elete / (A)bort
choose_rda() {
    local prompt="$1"
    local response
    while true; do
        read -r -p "$prompt (R)euse / (D)elete and recreate / (A)bort: " response
        case "$response" in
            [rR]) return 0 ;;  # Reuse
            [dD]) return 1 ;;  # Delete
            [aA]) return 2 ;;  # Abort
            *) echo "Please enter R, D, or A" ;;
        esac
    done
}

# Choose resume or start fresh
choose_resume() {
    local response
    while true; do
        read -r -p "Previous install incomplete. (R)esume / (S)tart fresh: " response
        case "$response" in
            [rR]) return 0 ;;  # Resume
            [sS]) return 1 ;;  # Start fresh
            *) echo "Please enter R or S" ;;
        esac
    done
}

# ============================================================================
# Progress tracking functions
# ============================================================================

init_progress() {
    if [[ -f "$PROGRESS_FILE" ]]; then
        return 0  # Progress file exists
    else
        touch "$PROGRESS_FILE"
        return 1  # Fresh install
    fi
}

mark_step_done() {
    local step_name="$1"
    echo "$step_name" >> "$PROGRESS_FILE"
}

is_step_done() {
    local step_name="$1"
    if [[ -f "$PROGRESS_FILE" ]]; then
        grep -q "^${step_name}$" "$PROGRESS_FILE"
        return $?
    fi
    return 1
}

clear_progress() {
    rm -f "$PROGRESS_FILE"
}

# ============================================================================
# Pre-flight check functions
# ============================================================================

check_conda() {
    # Check for mamba first (faster), then conda
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
        CONDA_VERSION=$(mamba --version | head -1)
        success "Found mamba: $CONDA_VERSION"
        return 0
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
        CONDA_VERSION=$(conda --version)
        success "Found conda: $CONDA_VERSION"
        return 0
    else
        error "Neither conda nor mamba found"
        echo ""
        echo "Please install Miniconda or Anaconda first:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        echo ""
        echo "Or install Mamba (faster alternative):"
        echo "  https://mamba.readthedocs.io/en/latest/installation.html"
        return 1
    fi
}

check_internet() {
    info "Checking internet connectivity..."
    if curl -s --head --connect-timeout 5 https://github.com > /dev/null; then
        success "Internet connection OK"
        return 0
    else
        error "Cannot reach github.com"
        echo "Please check your internet connection and try again."
        return 1
    fi
}

check_disk_space() {
    local required_gb=${1:-5}
    local available_kb

    if [[ "$OSTYPE" == "darwin"* ]]; then
        available_kb=$(df -k "$HOME" | awk 'NR==2 {print $4}')
    else
        available_kb=$(df -k "$HOME" | awk 'NR==2 {print $4}')
    fi

    local available_gb=$((available_kb / 1024 / 1024))

    if [[ $available_gb -ge $required_gb ]]; then
        success "Disk space OK (${available_gb}GB available, ${required_gb}GB required)"
        return 0
    else
        error "Insufficient disk space: ${available_gb}GB available, ${required_gb}GB required"
        return 1
    fi
}

# ============================================================================
# Environment management functions
# ============================================================================

env_exists() {
    local env_name="$1"
    $CONDA_CMD env list | grep -q "^${env_name} "
    return $?
}

create_env() {
    local env_name="$1"
    info "Creating conda environment '$env_name' with Python 3.12..."
    $CONDA_CMD create -n "$env_name" python=3.12 -y
    return $?
}

remove_env() {
    local env_name="$1"
    info "Removing existing environment '$env_name'..."
    $CONDA_CMD env remove -n "$env_name" -y
    return $?
}

activate_env() {
    local env_name="$1"

    # Source conda.sh to enable conda activate in script
    if [[ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]]; then
        source "$CONDA_PREFIX/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    fi

    conda activate "$env_name"
    return $?
}

# ============================================================================
# Command execution with error handling
# ============================================================================

run_cmd() {
    local description="$1"
    shift
    local cmd="$*"

    info "$description"
    echo "Running: $cmd" >> "$LOG_FILE"

    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        success "$description"
        return 0
    else
        error "Failed: $description"
        echo ""
        echo "Command was: $cmd"
        echo "Check $LOG_FILE for details."
        return 1
    fi
}

run_cmd_interactive() {
    local description="$1"
    shift
    local cmd="$*"

    info "$description"
    echo "Running: $cmd" >> "$LOG_FILE"

    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        success "$description"
        return 0
    else
        error "Failed: $description"
        return 1
    fi
}

# ============================================================================
# Shell integration
# ============================================================================

setup_shell_function() {
    local env_name="${1:-adams}"  # Default to 'adams' if not provided
    step "Setting up shell integration"

    # Detect user's shell
    local user_shell=""
    local rc_file=""

    if [[ -n "${SHELL:-}" ]]; then
        case "$SHELL" in
            */zsh)
                user_shell="zsh"
                rc_file="$HOME/.zshrc"
                ;;
            */bash)
                user_shell="bash"
                rc_file="$HOME/.bashrc"
                ;;
            *)
                warn "Unknown shell: $SHELL"
                info "Defaulting to bash"
                user_shell="bash"
                rc_file="$HOME/.bashrc"
                ;;
        esac
    else
        info "SHELL variable not set, defaulting to bash"
        user_shell="bash"
        rc_file="$HOME/.bashrc"
    fi

    info "Detected shell: $user_shell"
    info "RC file: $rc_file"

    # Check if function already exists
    if [[ -f "$rc_file" ]] && grep -q "# ADAMS shell function" "$rc_file" 2>/dev/null; then
        warn "ADAMS shell function already exists in $rc_file"
        if confirm_yes "Update it?"; then
            # Remove old function
            sed -i.bak '/# ADAMS shell function/,/^}$/d' "$rc_file"
            info "Removed old function"
        else
            info "Skipping shell function setup"
            return 0
        fi
    fi

    # Get conda base path (most reliable method)
    local conda_base
    conda_base=$(conda info --base 2>/dev/null)

    if [[ -z "$conda_base" ]]; then
        warn "Could not determine conda base path"
        warn "Shell function will assume conda is in PATH"
        conda_base=""
    else
        success "Found conda at: $conda_base"
    fi

    # Create the shell function
    info "Adding ADAMS function to $rc_file..."

    cat >> "$rc_file" << EOF

# ADAMS shell function - auto-activates conda environment
adams() {
    conda run -n $env_name --no-capture-output python -m adams.cli "\$@"
}
EOF

    success "Shell function added to $rc_file"
    echo ""
    echo "You can now run 'adams' from any directory without activating the environment!"
    echo ""
    echo "To use it immediately in this shell, run:"
    echo "  source $rc_file"
    echo ""
}

# ============================================================================
# Initialization
# ============================================================================

init_log() {
    echo "======================================" >> "$LOG_FILE"
    echo "ADAMS Installation Log" >> "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "======================================" >> "$LOG_FILE"
}
