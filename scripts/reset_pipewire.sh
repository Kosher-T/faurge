#!/usr/bin/env bash
# ============================================================================
# scripts/reset_pipewire.sh
# Safely tears down the Faurge virtual patch bay.
#
# This script:
#   1. Kills the Faurge loopback process.
#   2. Removes any Faurge-specific PipeWire links.
#   3. Restores the default audio routing.
#
# Usage: bash scripts/reset_pipewire.sh
# ============================================================================

set -euo pipefail

LOCKFILE="/tmp/faurge-pipewire.lock"
LOG_TAG="faurge-pipewire"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; logger -t "$LOG_TAG" "INFO: $*" 2>/dev/null || true; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; logger -t "$LOG_TAG" "WARN: $*" 2>/dev/null || true; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; logger -t "$LOG_TAG" "ERROR: $*" 2>/dev/null || true; }

# --- Tear Down Loopback ---
kill_faurge_loopback() {
    if [[ -f "$LOCKFILE" ]]; then
        local pid
        pid="$(cat "$LOCKFILE")"

        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping Faurge loopback (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 0.5

            # Force kill if still alive
            if kill -0 "$pid" 2>/dev/null; then
                log_warn "Loopback did not stop gracefully. Sending SIGKILL..."
                kill -9 "$pid" 2>/dev/null || true
            fi

            log_info "Faurge loopback stopped."
        else
            log_warn "Loopback process (PID: $pid) is not running."
        fi

        rm -f "$LOCKFILE"
    else
        log_info "No lockfile found. Searching for orphaned loopback processes..."

        # Find and kill any orphaned pw-loopback processes with faurge in the name
        local pids
        pids=$(pgrep -f "pw-loopback.*faurge" 2>/dev/null || true)

        if [[ -n "$pids" ]]; then
            log_warn "Found orphaned Faurge loopback processes: $pids"
            echo "$pids" | xargs kill 2>/dev/null || true
            sleep 0.5
            log_info "Orphaned processes terminated."
        else
            log_info "No Faurge loopback processes found."
        fi
    fi
}

# --- Remove Faurge Links ---
remove_faurge_links() {
    log_info "Removing Faurge-specific PipeWire links..."

    # Disconnect any links involving faurge nodes
    local faurge_links
    faurge_links=$(pw-link -l 2>/dev/null | grep -i "faurge" || true)

    if [[ -n "$faurge_links" ]]; then
        # pw-link -d disconnects links
        while IFS= read -r link; do
            local output_port input_port
            output_port=$(echo "$link" | awk '{print $1}')
            input_port=$(echo "$link" | awk '{print $3}')
            if [[ -n "$output_port" && -n "$input_port" ]]; then
                pw-link -d "$output_port" "$input_port" 2>/dev/null || true
            fi
        done <<< "$faurge_links"
        log_info "Faurge links removed."
    else
        log_info "No Faurge-specific links found."
    fi
}

# --- Restore Default Routing ---
restore_defaults() {
    log_info "Restoring default WirePlumber policies..."

    # WirePlumber will automatically re-establish default routing
    # once Faurge nodes are removed. Give it a moment.
    if command -v wpctl &>/dev/null; then
        wpctl set-default "$(wpctl status 2>/dev/null | grep -A1 'Sinks:' | tail -1 | awk '{print $1}' | tr -d '.')" 2>/dev/null || true
        log_info "Default sink restored via WirePlumber."
    fi
}

# --- Main ---
main() {
    log_info "========================================"
    log_info " Faurge PipeWire Teardown"
    log_info "========================================"

    kill_faurge_loopback
    remove_faurge_links
    restore_defaults

    log_info "========================================"
    log_info " PipeWire teardown complete."
    log_info " Audio routing restored to defaults."
    log_info "========================================"
}

main "$@"
