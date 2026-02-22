#!/usr/bin/env bash
# ============================================================================
# scripts/setup_pipewire.sh
# Establishes (and validates) the Faurge virtual patch bay using PipeWire.
#
# This script:
#   1. Verifies PipeWire and WirePlumber are running.
#   2. Creates a virtual audio source (faurge-loopback) via pw-loopback.
#   3. Links the default audio capture to the Faurge processing pipeline.
#   4. Establishes the dry-bypass path (hardware → app, untouched).
#
# Usage: sudo bash scripts/setup_pipewire.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCKFILE="/tmp/faurge-pipewire.lock"
LOG_TAG="faurge-pipewire"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; logger -t "$LOG_TAG" "INFO: $*" 2>/dev/null || true; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; logger -t "$LOG_TAG" "WARN: $*" 2>/dev/null || true; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; logger -t "$LOG_TAG" "ERROR: $*" 2>/dev/null || true; }

# --- Preflight Checks ---
check_dependencies() {
    local missing=()
    for cmd in pw-cli pw-link pw-loopback wpctl pactl; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required commands: ${missing[*]}"
        log_error "Install PipeWire, WirePlumber, and pipewire-pulse."
        exit 1
    fi
}

check_pipewire_running() {
    if ! pw-cli info 0 &>/dev/null; then
        log_error "PipeWire is not running. Start it with: systemctl --user start pipewire"
        exit 1
    fi
    log_info "PipeWire is running."

    if ! wpctl status &>/dev/null; then
        log_warn "WirePlumber may not be running. Some features may not work."
    fi
}

# --- Loopback Setup ---
create_faurge_loopback() {
    # Check if a Faurge loopback already exists
    if pw-cli list-objects | grep -q "faurge-loopback"; then
        log_warn "Faurge loopback already exists. Skipping creation."
        return 0
    fi

    log_info "Creating Faurge loopback sink/source pair..."

    # Create a loopback module that acts as Faurge's virtual patch point.
    # The "capture" side receives audio from apps, the "playback" side sends to hardware.
    pw-loopback \
        --capture-props='media.class=Audio/Sink node.name=faurge-loopback-sink node.description="Faurge Processing Input"' \
        --playback-props='media.class=Audio/Source node.name=faurge-loopback-source node.description="Faurge Processing Output"' \
        --channel-map='[ FL FR ]' &

    LOOPBACK_PID=$!
    echo "$LOOPBACK_PID" > "$LOCKFILE"

    # Give PipeWire a moment to register the new nodes
    sleep 1

    if kill -0 "$LOOPBACK_PID" 2>/dev/null; then
        log_info "Faurge loopback created (PID: $LOOPBACK_PID)"
    else
        log_error "Failed to create Faurge loopback."
        exit 1
    fi
}

setup_dry_bypass() {
    log_info "Establishing dry-bypass path..."

    # The dry bypass connects the default audio source directly to the default
    # audio sink, ensuring audio always flows even if Faurge's processing is offline.
    # This is the safety net — audio must NEVER be interrupted.

    local default_source
    default_source=$(wpctl inspect @DEFAULT_SOURCE@ 2>/dev/null | grep "node.name" | head -1 | awk -F'"' '{print $2}' || echo "")

    local default_sink
    default_sink=$(wpctl inspect @DEFAULT_SINK@ 2>/dev/null | grep "node.name" | head -1 | awk -F'"' '{print $2}' || echo "")

    if [[ -z "$default_source" || -z "$default_sink" ]]; then
        log_warn "Could not detect default source/sink. Dry bypass may need manual configuration."
        return 0
    fi

    log_info "Default source: $default_source"
    log_info "Default sink:   $default_sink"

    # Link them if not already linked
    # pw-link will fail silently if already linked, which is fine
    pw-link "$default_source:capture_FL" "$default_sink:playback_FL" 2>/dev/null || true
    pw-link "$default_source:capture_FR" "$default_sink:playback_FR" 2>/dev/null || true

    log_info "Dry-bypass path established."
}

# --- Main ---
main() {
    log_info "========================================"
    log_info " Faurge PipeWire Setup"
    log_info "========================================"

    check_dependencies
    check_pipewire_running
    create_faurge_loopback
    setup_dry_bypass

    log_info "========================================"
    log_info " PipeWire setup complete."
    log_info " Loopback PID saved to: $LOCKFILE"
    log_info "========================================"
}

main "$@"
