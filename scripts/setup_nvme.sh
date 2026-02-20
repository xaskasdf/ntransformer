#!/bin/bash
# ntransformer: NVMe pre-configuration for gpu-nvme-direct backend
#
# Ensures the NVMe device is ready for GPU-initiated I/O:
#   1. Loads VFIO modules (unsafe noiommu mode)
#   2. Binds NVMe to vfio-pci (calls gpu-nvme-direct's setup_vfio.sh)
#   3. Forces PCIe power state D0
#   4. Enables Memory Space + Bus Master in PCI Command register
#   5. Verifies BAR0 resource is accessible
#
# Usage: sudo ./setup_nvme.sh [PCI_BDF]
#   Default BDF: 0000:01:00.0 (WD SN740 PCIe 4.0 in our test rig)
#
# Run this after every reboot, before launching ntransformer with
# GPUNVME_PCI_BDF and GPUNVME_GGUF_LBA environment variables.
#
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

BDF="${1:-0000:01:00.0}"
SYSFS="/sys/bus/pci/devices/$BDF"
GPUNVME_DIR="$(cd "$(dirname "$0")/../.." && pwd)/gpu-nvme-direct"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}OK${NC}   $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; exit 1; }

echo "=== ntransformer NVMe Setup ==="
echo "Device: $BDF"
echo ""

# ── 0. Check root ──
if [ "$(id -u)" -ne 0 ]; then
    fail "Must run as root (sudo)"
fi

# ── 1. Check device exists ──
if [ ! -d "$SYSFS" ]; then
    echo "Device $BDF not found. Available NVMe devices:"
    lspci -nn | grep -i "NVMe\|Non-Volatile" || echo "  (none)"
    fail "Device $BDF not in sysfs"
fi

DEVNAME=$(lspci -s "$BDF" 2>/dev/null | cut -d: -f3- | sed 's/^ //')
echo "Found: $DEVNAME"
echo ""

# ── 2. Safety: refuse boot drive ──
ROOT_DEV=$(findmnt -n -o SOURCE / 2>/dev/null || echo "")
if echo "$ROOT_DEV" | grep -q "nvme"; then
    ROOT_NVME=$(readlink -f "/sys/block/$(echo "$ROOT_DEV" | sed 's|/dev/||;s|p[0-9]*||')/device/device" 2>/dev/null || echo "")
    if echo "$ROOT_NVME" | grep -q "$BDF"; then
        fail "$BDF is your boot NVMe! Refusing."
    fi
fi

# ── 3. Load VFIO modules ──
echo "[1/5] Loading VFIO modules..."
modprobe vfio enable_unsafe_noiommu_mode=1
ok "vfio (unsafe noiommu)"
modprobe vfio-pci
ok "vfio-pci"

# ── 4. Bind to vfio-pci ──
echo "[2/5] Binding to vfio-pci..."
CURRENT_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")

if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
    ok "Already bound to vfio-pci"
elif [ -x "$GPUNVME_DIR/scripts/setup_vfio.sh" ]; then
    "$GPUNVME_DIR/scripts/setup_vfio.sh" "$BDF" 2>&1 | sed 's/^/  /'
    CURRENT_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")
    if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
        ok "Bound to vfio-pci"
    else
        fail "Binding failed (driver=$CURRENT_DRIVER)"
    fi
else
    # Inline fallback if gpu-nvme-direct not at expected path
    warn "gpu-nvme-direct not found at $GPUNVME_DIR, doing inline bind"
    VENDOR=$(cat "$SYSFS/vendor" 2>/dev/null | sed 's/0x//')
    DEVICE=$(cat "$SYSFS/device" 2>/dev/null | sed 's/0x//')
    if [ "$CURRENT_DRIVER" != "none" ]; then
        echo "$BDF" > "$SYSFS/driver/unbind" 2>/dev/null || true
        sleep 0.5
    fi
    echo "$VENDOR $DEVICE" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
    sleep 0.5
    CURRENT_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")
    if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
        ok "Bound to vfio-pci (inline)"
    else
        fail "Binding failed (driver=$CURRENT_DRIVER)"
    fi
fi

# ── 5. Force power state D0 ──
echo "[3/5] Forcing PCIe power D0..."
echo on > "$SYSFS/power/control" 2>/dev/null || true
ok "power/control = on"

# PM Control register at offset 0x84 — clear bits 1:0 to force D0,
# then set bit 3 (PME_En=0, No_Soft_Reset=1) = 0x0008
setpci -s "$BDF" 0x84.W=0x0008 2>/dev/null || warn "setpci PM failed"
ok "PM CSR = 0x0008 (D0)"

# ── 6. Enable Memory + BusMaster ──
echo "[4/5] Enabling Memory Space + Bus Master..."
setpci -s "$BDF" COMMAND=0x0006 2>/dev/null || warn "setpci COMMAND failed"
CMD=$(setpci -s "$BDF" COMMAND 2>/dev/null || echo "????")
ok "PCI COMMAND = 0x$CMD"

# ── 7. Verify BAR0 accessible ──
echo "[5/5] Verifying BAR0..."
if [ -f "$SYSFS/resource0" ]; then
    BAR0_SIZE=$(stat -c%s "$SYSFS/resource0" 2>/dev/null || echo "0")
    if [ "$BAR0_SIZE" -gt 0 ]; then
        ok "resource0 ready ($BAR0_SIZE bytes)"
    else
        warn "resource0 exists but size=0"
    fi
else
    fail "resource0 not found — BAR0 not mapped"
fi

# ── Summary ──
echo ""
echo -e "${GREEN}=== NVMe ready ===${NC}"
echo ""
echo "Run ntransformer with:"
echo "  sudo GPUNVME_PCI_BDF=$BDF GPUNVME_GGUF_LBA=0 \\"
echo "       ./ntransformer -m /path/to/model.gguf --streaming"
echo ""
echo "Or test gpu-nvme-direct first:"
echo "  cd ~/gpu-nvme-direct/build-hw"
echo "  sudo ./test_layer_loader $BDF"
