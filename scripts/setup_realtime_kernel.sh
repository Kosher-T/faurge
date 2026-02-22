#!/usr/bin/env bash
# ============================================================================
# scripts/setup_realtime_kernel.sh
# Configures the Linux system for low-latency audio processing.
#
# This script:
#   1. Sets up real-time scheduling priorities (rtprio) for the audio group.
#   2. Configures PipeWire for low-latency operation.
#   3. Optionally installs a PREEMPT_RT kernel (if requested).
#   4. Verifies the current kernel preemption model.
#
# Usage: sudo bash scripts/setup_realtime_kernel.sh [--install-rt-kernel]
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }

# --- Check Root ---
require_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (sudo)."
        exit 1
    fi
}

# --- Step 1: Configure Real-Time Scheduling Limits ---
setup_rtprio() {
    log_step "Configuring real-time scheduling limits..."

    local limits_file="/etc/security/limits.d/99-faurge-realtime.conf"

    cat > "$limits_file" << 'EOF'
# Faurge Real-Time Audio Configuration
# Allows members of the 'audio' group to use real-time scheduling.
#
# <domain>    <type>  <item>    <value>
@audio        -       rtprio    95
@audio        -       memlock   unlimited
@audio        -       nice      -19
EOF

    chmod 644 "$limits_file"
    log_info "Real-time limits written to: $limits_file"

    # Ensure current user is in the audio group
    local target_user="${SUDO_USER:-$USER}"
    if id -nG "$target_user" | grep -qw "audio"; then
        log_info "User '$target_user' is already in the 'audio' group."
    else
        log_warn "User '$target_user' is NOT in the 'audio' group."
        log_info "Adding '$target_user' to 'audio' group..."
        usermod -aG audio "$target_user"
        log_info "User added. A logout/login is required for this to take effect."
    fi
}

# --- Step 2: Configure PipeWire for Low Latency ---
setup_pipewire_realtime() {
    log_step "Configuring PipeWire for low-latency operation..."

    local pw_conf_dir="/etc/pipewire/pipewire.conf.d"
    mkdir -p "$pw_conf_dir"

    local faurge_conf="$pw_conf_dir/99-faurge-lowlatency.conf"

    cat > "$faurge_conf" << 'EOF'
# Faurge Low-Latency PipeWire Configuration
# Reduces quantum (buffer size) for lower latency.
# Default quantum is 1024; we reduce to 256 for ~5.3ms at 48kHz.

context.properties = {
    default.clock.quantum     = 256
    default.clock.min-quantum = 64
    default.clock.max-quantum = 1024
    default.clock.rate        = 48000
}

# Enable real-time scheduling for PipeWire
context.modules = [
    {
        name = libpipewire-module-rt
        args = {
            nice.level   = -11
            rt.prio      = 88
            rt.time.soft = 2000000   # 2s soft limit
            rt.time.hard = 2000000   # 2s hard limit
        }
        flags = [ ifexists nofail ]
    }
]
EOF

    chmod 644 "$faurge_conf"
    log_info "PipeWire low-latency config written to: $faurge_conf"
}

# --- Step 3: Verify Kernel Preemption Model ---
check_kernel_preemption() {
    log_step "Checking kernel preemption model..."

    local preempt_file="/sys/kernel/realtime"
    if [[ -f "$preempt_file" ]] && [[ "$(cat "$preempt_file")" == "1" ]]; then
        log_info "✓ Running a PREEMPT_RT kernel. Optimal for real-time audio."
        return 0
    fi

    local kernel_config="/boot/config-$(uname -r)"
    if [[ -f "$kernel_config" ]]; then
        if grep -q "CONFIG_PREEMPT_RT=y" "$kernel_config"; then
            log_info "✓ Kernel has PREEMPT_RT compiled in."
        elif grep -q "CONFIG_PREEMPT=y" "$kernel_config"; then
            log_warn "Kernel uses PREEMPT (voluntary), not PREEMPT_RT."
            log_warn "This is acceptable for most use cases, but PREEMPT_RT is preferred."
        else
            log_warn "Kernel preemption model could not be determined from config."
        fi
    else
        log_warn "Kernel config not found at $kernel_config. Cannot verify preemption model."
    fi

    log_info "Current kernel: $(uname -r)"
}

# --- Step 4 (Optional): Install RT Kernel ---
install_rt_kernel() {
    log_step "Installing PREEMPT_RT kernel..."

    if command -v apt-get &>/dev/null; then
        # Debian/Ubuntu/Pop!_OS
        local rt_pkg
        rt_pkg=$(apt-cache search "linux-image.*rt" 2>/dev/null | head -1 | awk '{print $1}')

        if [[ -z "$rt_pkg" ]]; then
            log_error "No RT kernel package found in apt repositories."
            log_info "You may need to add a PPA or build from source."
            log_info "See: https://wiki.linuxfoundation.org/realtime/start"
            return 1
        fi

        log_info "Installing: $rt_pkg"
        apt-get install -y "$rt_pkg"
        log_info "RT kernel installed. Reboot to activate."
    elif command -v dnf &>/dev/null; then
        # Fedora
        log_info "Installing kernel-rt via dnf..."
        dnf install -y kernel-rt kernel-rt-devel
        log_info "RT kernel installed. Reboot to activate."
    else
        log_error "Unsupported package manager. Please install PREEMPT_RT manually."
        return 1
    fi
}

# --- Summary ---
print_summary() {
    echo ""
    log_info "========================================"
    log_info " Faurge Real-Time Setup Summary"
    log_info "========================================"
    log_info "  rtprio limits:     /etc/security/limits.d/99-faurge-realtime.conf"
    log_info "  PipeWire config:   /etc/pipewire/pipewire.conf.d/99-faurge-lowlatency.conf"
    log_info "  Kernel:            $(uname -r)"
    echo ""
    log_warn "ACTION REQUIRED:"
    log_warn "  1. Log out and log back in for group changes to take effect."
    log_warn "  2. Restart PipeWire:  systemctl --user restart pipewire pipewire-pulse wireplumber"
    log_warn "  3. If you installed an RT kernel, reboot your system."
    echo ""
}

# --- Main ---
main() {
    require_root

    log_info "========================================"
    log_info " Faurge Real-Time Kernel Setup"
    log_info "========================================"

    setup_rtprio
    setup_pipewire_realtime
    check_kernel_preemption

    # Optional: install RT kernel if flag is passed
    if [[ "${1:-}" == "--install-rt-kernel" ]]; then
        install_rt_kernel
    fi

    print_summary
}

main "$@"
