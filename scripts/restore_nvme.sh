#!/bin/bash
# ntransformer: Restore NVMe to kernel driver
#
# Undoes setup_nvme.sh: unbinds from vfio-pci, rebinds to the kernel
# nvme driver so the device appears as /dev/nvmeXn1 again.
#
# Useful for:
#   - Mounting the NVMe filesystem after a gpu-nvme-direct session
#   - Writing a new GGUF to the NVMe with dd
#   - Returning the system to normal state
#
# Usage: sudo ./restore_nvme.sh [PCI_BDF]
#   Default BDF: 0000:01:00.0
#
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

BDF="${1:-0000:01:00.0}"
SYSFS="/sys/bus/pci/devices/$BDF"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}OK${NC}   $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; exit 1; }

echo "=== Restore NVMe to kernel driver ==="
echo "Device: $BDF"
echo ""

if [ "$(id -u)" -ne 0 ]; then
    fail "Must run as root (sudo)"
fi

if [ ! -d "$SYSFS" ]; then
    fail "Device $BDF not found in sysfs"
fi

CURRENT_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")
echo "Current driver: $CURRENT_DRIVER"

if [ "$CURRENT_DRIVER" = "nvme" ]; then
    ok "Already using kernel nvme driver"
    echo ""
    # Show block devices
    echo "Block devices:"
    ls -1 /dev/nvme* 2>/dev/null | sed 's/^/  /' || echo "  (none yet)"
    exit 0
fi

# ── 1. Unbind from vfio-pci ──
echo "[1/3] Unbinding from vfio-pci..."
if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
    sleep 0.5
    ok "Unbound from vfio-pci"
elif [ "$CURRENT_DRIVER" = "none" ]; then
    ok "No driver bound (skipping unbind)"
else
    warn "Unexpected driver '$CURRENT_DRIVER', unbinding..."
    echo "$BDF" > "$SYSFS/driver/unbind" 2>/dev/null || true
    sleep 0.5
fi

# ── 2. Clear driver_override and trigger probe ──
echo "[2/3] Triggering kernel driver probe..."
echo "" > "$SYSFS/driver_override" 2>/dev/null || true
echo "$BDF" > /sys/bus/pci/drivers_probe 2>/dev/null || true
sleep 1

# ── 3. Verify ──
echo "[3/3] Verifying..."
NEW_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")

if [ "$NEW_DRIVER" = "nvme" ]; then
    ok "Rebound to kernel nvme driver"
    echo ""

    # Wait for block device to appear
    sleep 1
    echo "Block devices:"
    ls -1 /dev/nvme* 2>/dev/null | sed 's/^/  /' || echo "  (waiting for enumeration...)"
    echo ""
    echo -e "${GREEN}=== NVMe restored ===${NC}"
    echo ""
    echo "You can now:"
    echo "  - Write a GGUF:  sudo dd if=model.gguf of=/dev/nvme0n1 bs=1M oflag=direct status=progress"
    echo "  - Mount:         sudo mount /dev/nvme0n1p1 /mnt"
    echo "  - Re-setup:      sudo ./scripts/setup_nvme.sh $BDF"
else
    warn "Driver is '$NEW_DRIVER' (expected 'nvme')"
    echo ""
    echo "Try manual rebind:"
    echo "  echo $BDF > /sys/bus/pci/drivers/nvme/bind"
fi
