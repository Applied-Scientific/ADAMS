#!/bin/bash
# install.sh - Main entry point for ADAMS installation
#
# Usage:
#   From cloned repo:
#     ./scripts/install.sh
#
#   Via curl (downloads everything):
#     curl -fsSL https://raw.githubusercontent.com/Applied-Scientific/ADAMS/main/scripts/install.sh | bash
#
set -eo pipefail

# ============================================================================
# Configuration
# ============================================================================

GITHUB_RAW_BASE="https://raw.githubusercontent.com/Applied-Scientific/ADAMS/main/scripts"

# ============================================================================
# Determine script location
# ============================================================================

# Get the directory where this script is located
if [[ -n "${BASH_SOURCE[0]:-}" ]] && [[ -f "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    # Running via curl pipe - we'll need to download lib files
    SCRIPT_DIR=""
fi

# ============================================================================
# Source or download library files
# ============================================================================

download_lib() {
    local lib_name="$1"
    local temp_dir="$2"
    local url="${GITHUB_RAW_BASE}/lib/${lib_name}"

    if ! curl -fsSL "$url" -o "$temp_dir/$lib_name"; then
        echo "Error: Failed to download $lib_name"
        exit 1
    fi
}

load_libraries() {
    if [[ -n "$SCRIPT_DIR" ]] && [[ -d "$SCRIPT_DIR/lib" ]]; then
        # Running from cloned repo - source local files
        echo "Loading libraries from $SCRIPT_DIR/lib..."
        source "$SCRIPT_DIR/lib/common.sh" || {
            echo "Error: Failed to load common.sh"
            exit 1
        }
        source "$SCRIPT_DIR/lib/detect.sh" || {
            echo "Error: Failed to load detect.sh"
            exit 1
        }

        case "$PLATFORM" in
            linux)
                source "$SCRIPT_DIR/lib/install_linux.sh" || {
                    echo "Error: Failed to load install_linux.sh"
                    exit 1
                }
                ;;
            macos)
                source "$SCRIPT_DIR/lib/install_macos.sh" || {
                    echo "Error: Failed to load install_macos.sh"
                    exit 1
                }
                ;;
        esac
    else
        # Running via curl - download to temp directory
        TEMP_LIB_DIR=$(mktemp -d)
        trap "rm -rf '$TEMP_LIB_DIR'" EXIT

        echo "Downloading installation scripts..."
        download_lib "common.sh" "$TEMP_LIB_DIR"
        download_lib "detect.sh" "$TEMP_LIB_DIR"

        source "$TEMP_LIB_DIR/common.sh" || {
            echo "Error: Failed to load common.sh"
            exit 1
        }
        source "$TEMP_LIB_DIR/detect.sh" || {
            echo "Error: Failed to load detect.sh"
            exit 1
        }

        # Detect platform first, then download appropriate install script
        detect_platform

        case "$PLATFORM" in
            linux)
                download_lib "install_linux.sh" "$TEMP_LIB_DIR"
                source "$TEMP_LIB_DIR/install_linux.sh" || {
                    echo "Error: Failed to load install_linux.sh"
                    exit 1
                }
                ;;
            macos)
                download_lib "install_macos.sh" "$TEMP_LIB_DIR"
                source "$TEMP_LIB_DIR/install_macos.sh" || {
                    echo "Error: Failed to load install_macos.sh"
                    exit 1
                }
                ;;
            *)
                echo "Error: Unsupported platform: $PLATFORM"
                exit 1
                ;;
        esac
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo ""
    echo "========================================"
    echo "  ADAMS Installation Script"
    echo "  Agent-Driven Autonomous Molecular Simulations"
    echo "========================================"
    echo ""

    # Quick platform detection for library loading
    echo "Detecting platform..."
    case "$(uname -s)" in
        Linux) PLATFORM="linux" ;;
        Darwin) PLATFORM="macos" ;;
        *)
            echo "Error: Unsupported operating system: $(uname -s)"
            echo "ADAMS supports Linux and macOS only."
            exit 1
            ;;
    esac
    echo "Platform: $PLATFORM"

    # Load all library files
    load_libraries

    # Initialize logging
    echo "Initializing log..."
    init_log

    # Run full detection
    echo "Running system detection..."
    run_detection

    # Check prerequisites
    echo "Checking prerequisites..."
    check_conda || exit 1
    check_internet || exit 1
    check_disk_space 5 || exit 1

    # Print detection summary
    print_detection_summary

    # Run platform-specific installation
    case "$PLATFORM" in
        linux)
            run_linux_install
            ;;
        macos)
            run_macos_install
            ;;
    esac

    # Best-effort hardening: if user has an existing ~/.adams API key file, tighten permissions.
    # Note: ADAMS may optionally create this file when the user runs the CLI and opts into saving their key.
    if [[ -f "$HOME/.adams" ]]; then
        chmod 600 "$HOME/.adams" 2>/dev/null || true
    fi
}

# Run main
main "$@"
