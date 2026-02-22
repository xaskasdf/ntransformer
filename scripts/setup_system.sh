#!/bin/bash
# =============================================================================
# ntransformer + gpu-nvme-direct: System Setup Script
# =============================================================================
#
# Configures a Linux system for GPU-initiated NVMe I/O and LLM inference
# with ntransformer. Handles all OS-level, driver, and kernel modifications.
#
# Tested on: Ubuntu 25.10 (kernel 6.17), NVIDIA 590.48.01, CUDA 13.1
#            AMD Ryzen 5800X, RTX 3090, WD SN740 NVMe
#
# *** WARNING — READ BEFORE RUNNING ***
#
# This script modifies system-level components. Each modification carries risk:
#
#   Phase 2 (GRUB):     Disables IOMMU. This removes hardware isolation between
#                        PCIe devices. Malicious/buggy devices could DMA to
#                        arbitrary host memory. Requires reboot.
#
#   Phase 3 (NVIDIA):   Patches the NVIDIA kernel module source (os-mlock.c)
#                        and rebuilds via DKMS. A bad patch can prevent the GPU
#                        driver from loading (black screen on reboot). Backup is
#                        created automatically. Requires reboot.
#
#   Phase 3b (CUDA):    Patches a CUDA toolkit header (math_functions.h) to fix
#                        a glibc 2.42 noexcept conflict. Only needed on Ubuntu
#                        25.10+ with glibc >= 2.42. Backup created automatically.
#
#   Phase 4 (VFIO):     Enables unsafe noiommu mode for VFIO. This bypasses
#                        IOMMU protections. Required for consumer GPUs (GeForce)
#                        which don't support proper P2P IOMMU passthrough.
#
#   Phase 5 (NVMe):     Unbinds NVMe from kernel driver and gives it to VFIO.
#                        The NVMe will NOT appear as /dev/nvmeX while bound.
#                        NEVER run this on your boot drive. Script checks and
#                        refuses, but double-check your BDF.
#
# ROLLBACK:
#   - Phase 2: Remove the iommu=off param from /etc/default/grub, update-grub
#   - Phase 3: cp /usr/src/nvidia-VERSION/nvidia/os-mlock.c.orig os-mlock.c
#              dkms remove nvidia/VERSION --all && dkms install nvidia/VERSION
#   - Phase 3b: cp CUDA_PATH/include/crt/math_functions.h.orig math_functions.h
#   - Phase 5: sudo ./scripts/restore_nvme.sh [BDF]
#
# BIOS settings required (must be done manually before running this script):
#   - Above 4G Decoding: ON   (required for 64-bit BAR mapping)
#   - IOMMU: OFF              (or leave on, script adds kernel param)
#   - Secure Boot: OFF        (required for unsigned kernel module loading)
#
# SPDX-License-Identifier: BSD-2-Clause
# =============================================================================

set -euo pipefail

# ── Configuration ──
NVME_BDF="${NVME_BDF:-0000:01:00.0}"      # NVMe test device (NOT boot drive!)
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVIDIA_DKMS_VER=""                         # Auto-detected
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GPUNVME_DIR="$(cd "$NT_DIR/../gpu-nvme-direct" 2>/dev/null && pwd || echo "")"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC}   $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
info() { echo -e "  ${CYAN}[INFO]${NC} $1"; }
header() { echo -e "\n${BOLD}=== $1 ===${NC}\n"; }

START_PHASE=1
CHECK_ONLY=false
NVME_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) START_PHASE="$2"; shift 2 ;;
        --check) CHECK_ONLY=true; shift ;;
        --nvme-only) NVME_ONLY=true; START_PHASE=4; shift ;;
        --nvme-bdf) NVME_BDF="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: sudo $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --phase N       Start from phase N (1-7)"
            echo "  --check         Verify system state without making changes"
            echo "  --nvme-only     Run phases 4-5 only (NVMe setup, needed per-reboot)"
            echo "  --nvme-bdf BDF  NVMe PCI address (default: $NVME_BDF)"
            echo ""
            echo "Phases:"
            echo "  1   System packages (gcc-14, cmake, kernel headers)"
            echo "  2   GRUB kernel parameters (IOMMU disable)"
            echo "  3   NVIDIA DKMS patch (cudaHostRegisterIoMemory fix)"
            echo "  3b  CUDA header patch (glibc 2.42 rsqrt noexcept conflict)"
            echo "  4   VFIO module loading (per-reboot)"
            echo "  5   NVMe device setup (per-reboot)"
            echo "  6   Build ntransformer + gpu-nvme-direct"
            echo "  7   Verification tests"
            echo ""
            echo "Run --check first to see what needs to be done without making changes."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


# =========================================================================
# Helper Functions (defined before use)
# =========================================================================

# NVIDIA DKMS Patch: os-mlock.c
#
# WHY:  Linux kernel 6.12+ removed follow_pfn(), which the NVIDIA driver uses
#       to resolve PFNs for IO memory registration (cudaHostRegisterIoMemory).
#       Without this, the GPU cannot map NVMe BAR0 via CUDA, which is required
#       for gpu-nvme-direct to write doorbells from GPU kernels.
#
# WHAT: Adds nv_follow_flavors() which tries (in order):
#       1. follow_pfnmap_start/end() — kernel 6.12+ (GPL-only symbol)
#       2. follow_pte()              — kernel 6.10+ (non-GPL, works with NVIDIA)
#       3. Returns error             — if neither available
#       The NVIDIA conftest.sh already detects which symbols are present.
#
# RISK: If the patch is wrong, the NVIDIA driver won't compile (DKMS fails)
#       or won't load (black screen on reboot). We create a .orig backup.
#       Recovery: restore the .orig file, rebuild DKMS, reboot.
apply_osmlock_patch() {
    local FILE="$1"

    python3 - "$FILE" << 'PYEOF'
import sys, re

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

flavors_func = '''
static inline int nv_follow_flavors(struct vm_area_struct *vma,
                                    unsigned long address,
                                    unsigned long *pfn)
{
#if NV_IS_EXPORT_SYMBOL_PRESENT_follow_pfnmap_start
    struct follow_pfnmap_args args = {};
    int rc;

    args.address = address;
    args.vma = vma;

    rc = follow_pfnmap_start(&args);
    if (rc)
        return rc;

    *pfn = args.pfn;

    follow_pfnmap_end(&args);

    return 0;
#elif NV_IS_EXPORT_SYMBOL_PRESENT_follow_pte
    int status = 0;
    spinlock_t *ptl;
    pte_t *ptep;

    if (!(vma->vm_flags & (VM_IO | VM_PFNMAP)))
        return status;

    //
    // The first argument of follow_pte() was changed from
    // mm_struct to vm_area_struct in kernel 6.10.
    //
#if defined(NV_FOLLOW_PTE_ARG1_VMA)
    status = follow_pte(vma, address, &ptep, &ptl);
#else
    status = follow_pte(vma->vm_mm, address, &ptep, &ptl);
#endif
    if (status)
        return status;

#if defined(NV_PTEP_GET_PRESENT)
    *pfn = pte_pfn(ptep_get(ptep));
#else
    *pfn = pte_pfn(READ_ONCE(*ptep));
#endif

    // The lock is acquired inside follow_pte()
    pte_unmap_unlock(ptep, ptl);
    return 0;
#else
    return -1;
#endif // NV_IS_EXPORT_SYMBOL_PRESENT_follow_pfnmap_start
}

'''

# Match nv_follow_pfn function and replace its body
pattern = r'(static inline int nv_follow_pfn\(struct vm_area_struct \*vma,\s*unsigned long address,\s*unsigned long \*pfn\)\s*\{)[^}]*(\})'
replacement_body = r'\1\n    return nv_follow_flavors(vma, address, pfn);\n\2'

insert_point = content.find('static inline int nv_follow_pfn')
if insert_point == -1:
    print("ERROR: Could not find nv_follow_pfn function", file=sys.stderr)
    sys.exit(1)

if 'nv_follow_flavors' in content:
    print("Already patched, skipping", file=sys.stderr)
    sys.exit(0)

new_content = content[:insert_point] + flavors_func + content[insert_point:]
new_content = re.sub(pattern, replacement_body, new_content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(new_content)

print(f"Patched {filepath}", file=sys.stderr)
PYEOF
}

# CUDA Header Patch: math_functions.h
#
# WHY:  glibc 2.42+ (Ubuntu 25.10) declares rsqrt()/rsqrtf() with the
#       C++23 `noexcept` specifier via __GLIBC_USE_IEC_60559_FUNCS_EXT_C23.
#       CUDA 13.1 declares the same functions WITHOUT noexcept. The compiler
#       sees two conflicting declarations and errors out.
#
# WHAT: Adds a preprocessor guard around rsqrt() and rsqrtf() declarations:
#       if glibc provides the IEC 60559 C23 extension, declare with noexcept;
#       otherwise, declare without. This matches what the host compiler expects.
#
# RISK: Minimal. Only affects the rsqrt/rsqrtf declarations. If CUDA is
#       updated, the vendor may include this fix and the patch becomes a no-op
#       (the guard is already present). Backup created as .orig.
apply_cuda_rsqrt_patch() {
    local FILE="$1"

    python3 - "$FILE" << 'PYEOF'
import sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    lines = f.readlines()

# Check if already patched
content = ''.join(lines)
if '__GLIBC_USE_IEC_60559_FUNCS_EXT_C23' in content:
    print("Already patched, skipping", file=sys.stderr)
    sys.exit(0)

new_lines = []
i = 0
patched = 0
while i < len(lines):
    line = lines[i]
    # Look for: extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  rsqrt(double x);
    # or:       extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   rsqrtf(float x);
    stripped = line.strip()
    if (stripped.startswith('extern __DEVICE_FUNCTIONS_DECL__') and
        ('rsqrt(double x);' in stripped or 'rsqrtf(float x);' in stripped)):
        # Get the function name for the noexcept version
        func_decl = line.rstrip().rstrip(';')
        new_lines.append('#if defined(__GLIBC__) && defined(__GLIBC_USE_IEC_60559_FUNCS_EXT_C23) && __GLIBC_USE_IEC_60559_FUNCS_EXT_C23\n')
        new_lines.append(func_decl + ' noexcept;\n')
        new_lines.append('#else\n')
        new_lines.append(line)
        new_lines.append('#endif\n')
        patched += 1
    else:
        new_lines.append(line)
    i += 1

if patched == 0:
    print("WARNING: Could not find rsqrt declarations to patch", file=sys.stderr)
    sys.exit(1)

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print(f"Patched {patched} declaration(s) in {filepath}", file=sys.stderr)
PYEOF
}


# ── Root check ──
if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}ERROR: Must run as root (sudo)${NC}"
    exit 1
fi
REAL_USER="${SUDO_USER:-$(whoami)}"

echo -e "${BOLD}ntransformer + gpu-nvme-direct System Setup${NC}"
echo "────────────────────────────────────────────"
echo "  OS:        $(lsb_release -ds 2>/dev/null || grep PRETTY /etc/os-release 2>/dev/null | cut -d= -f2 || echo 'unknown')"
echo "  Kernel:    $(uname -r)"
echo "  NVMe BDF:  $NVME_BDF"
echo "  Mode:      $(if $CHECK_ONLY; then echo 'CHECK ONLY (no changes)'; elif $NVME_ONLY; then echo 'NVMe only (per-reboot)'; else echo "Full setup (from phase $START_PHASE)"; fi)"

if ! $CHECK_ONLY && ! $NVME_ONLY; then
    echo ""
    echo -e "  ${YELLOW}This script will modify system files (GRUB, NVIDIA DKMS, CUDA headers).${NC}"
    echo -e "  ${YELLOW}Run with --check first to preview changes without applying them.${NC}"
    echo ""
    read -r -p "  Continue? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi
echo ""


# =========================================================================
# PHASE 1: System Packages
# =========================================================================
if [ "$START_PHASE" -le 1 ] && ! $NVME_ONLY; then
    header "Phase 1: System Packages"

    PKGS_NEEDED=()

    # gcc-14 (CUDA 13.1 requires exactly gcc-14; gcc-15 is incompatible)
    if command -v gcc-14 &>/dev/null; then
        ok "gcc-14 $(gcc-14 --version | head -1 | grep -oP '\d+\.\d+\.\d+' || echo '')"
    else
        warn "gcc-14 not found (CUDA 13.1 requires gcc-14, NOT gcc-15)"
        PKGS_NEEDED+=(gcc-14 g++-14)
    fi

    # cmake >= 3.24
    if command -v cmake &>/dev/null; then
        ok "cmake $(cmake --version | head -1 | grep -oP '\d+\.\d+(\.\d+)?' || echo '')"
    else
        warn "cmake not found"
        PKGS_NEEDED+=(cmake)
    fi

    # kernel headers (needed for DKMS rebuild in phase 3)
    if [ -d "/lib/modules/$(uname -r)/build" ]; then
        ok "Kernel headers for $(uname -r)"
    else
        warn "Kernel headers missing (needed for NVIDIA DKMS rebuild)"
        PKGS_NEEDED+=("linux-headers-$(uname -r)")
    fi

    # pciutils (setpci, lspci — needed for NVMe BAR0 configuration)
    if command -v setpci &>/dev/null; then
        ok "pciutils (setpci/lspci)"
    else
        PKGS_NEEDED+=(pciutils)
    fi

    # python3 (needed for patch scripts)
    if command -v python3 &>/dev/null; then
        ok "python3"
    else
        PKGS_NEEDED+=(python3)
    fi

    # CUDA toolkit
    if [ -x "$CUDA_PATH/bin/nvcc" ]; then
        ok "CUDA $($CUDA_PATH/bin/nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//')"
    else
        warn "CUDA toolkit not found at $CUDA_PATH"
        info "Install from https://developer.nvidia.com/cuda-downloads"
        info "CUDA 13.1+ required. Set CUDA_PATH env var if not at /usr/local/cuda"
    fi

    # NVIDIA driver
    if lsmod | grep -q "^nvidia "; then
        NVIDIA_DKMS_VER=$(modinfo nvidia 2>/dev/null | grep "^version:" | awk '{print $2}')
        ok "NVIDIA driver $NVIDIA_DKMS_VER (DKMS)"
    else
        warn "NVIDIA kernel module not loaded"
        info "Install: apt install nvidia-driver nvidia-dkms"
    fi

    if [ ${#PKGS_NEEDED[@]} -gt 0 ]; then
        echo ""
        info "Missing packages: ${PKGS_NEEDED[*]}"
        if ! $CHECK_ONLY; then
            echo "  Installing..."
            apt-get update -qq
            apt-get install -y -qq "${PKGS_NEEDED[@]}"
            ok "Packages installed"
        fi
    else
        ok "All packages present"
    fi
fi


# =========================================================================
# PHASE 2: GRUB Kernel Parameters (IOMMU)
#
# WHY:  IOMMU (AMD-Vi / Intel VT-d) enforces device isolation by translating
#       PCIe bus addresses through page tables. gpu-nvme-direct needs the GPU
#       to write directly to NVMe BAR0 via posted PCIe writes. With IOMMU
#       active, these writes may be blocked or require complex IOMMU group
#       configuration that consumer motherboards don't support well.
#
# RISK: Disabling IOMMU removes DMA protection. Any PCIe device can read/write
#       arbitrary host memory. This is standard practice for GPU passthrough
#       and VFIO setups, but reduces security on multi-tenant systems.
#       On a single-user workstation, the risk is minimal.
# =========================================================================
if [ "$START_PHASE" -le 2 ] && ! $NVME_ONLY; then
    header "Phase 2: Kernel Parameters (IOMMU)"

    CMDLINE=$(cat /proc/cmdline)
    NEED_REBOOT=false

    CPU_VENDOR=$(grep -m1 "vendor_id" /proc/cpuinfo | awk '{print $3}')

    if [ "$CPU_VENDOR" = "AuthenticAMD" ]; then
        IOMMU_PARAM="amd_iommu=off"
        info "AMD CPU detected — will use amd_iommu=off"
    else
        IOMMU_PARAM="intel_iommu=off"
        info "Intel CPU detected — will use intel_iommu=off"
    fi

    if echo "$CMDLINE" | grep -q "$IOMMU_PARAM"; then
        ok "IOMMU already disabled ($IOMMU_PARAM in kernel cmdline)"
    elif dmesg 2>/dev/null | grep -qi "iommu.*disabled\|AMD-Vi.*disabled"; then
        ok "IOMMU appears disabled (BIOS setting or kernel default)"
    else
        warn "IOMMU may be active — gpu-nvme-direct requires it OFF"

        if ! $CHECK_ONLY; then
            GRUB_FILE="/etc/default/grub"
            CURRENT=$(grep "^GRUB_CMDLINE_LINUX=" "$GRUB_FILE" | sed 's/GRUB_CMDLINE_LINUX="//' | sed 's/"$//')

            if ! echo "$CURRENT" | grep -q "$IOMMU_PARAM"; then
                info "Adding '$IOMMU_PARAM' to GRUB_CMDLINE_LINUX"
                if [ -z "$CURRENT" ]; then
                    NEW="$IOMMU_PARAM"
                else
                    NEW="$CURRENT $IOMMU_PARAM"
                fi
                sed -i "s|^GRUB_CMDLINE_LINUX=.*|GRUB_CMDLINE_LINUX=\"$NEW\"|" "$GRUB_FILE"
                update-grub 2>/dev/null || grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null
                ok "GRUB updated"
                info "Rollback: edit /etc/default/grub, remove '$IOMMU_PARAM', run update-grub"
                NEED_REBOOT=true
            fi
        fi
    fi

    if $NEED_REBOOT; then
        echo ""
        echo -e "  ${YELLOW}*** REBOOT REQUIRED ***${NC}"
        echo "  Run: sudo reboot"
        echo "  Then re-run: sudo $0 --phase 3"
        exit 0
    fi
fi


# =========================================================================
# PHASE 3: NVIDIA DKMS Patch (os-mlock.c)
#
# WHY:  cudaHostRegisterIoMemory() allows CUDA to map PCI BAR regions so GPU
#       kernels can perform MMIO reads/writes. Internally, the NVIDIA driver
#       calls follow_pfn() to resolve virtual addresses to physical page frame
#       numbers. Linux kernel 6.12 removed follow_pfn() (commit 233eb0bf3b94).
#       Without this patch, cudaHostRegisterIoMemory fails and the GPU cannot
#       access NVMe registers.
#
# RISK: Modifies NVIDIA DKMS source and triggers a kernel module rebuild.
#       If the patch doesn't compile, the NVIDIA driver won't load after
#       reboot (you'll get a text console or Nouveau fallback).
#       RECOVERY: Boot into recovery mode or TTY, then:
#         cp /usr/src/nvidia-VERSION/nvidia/os-mlock.c.orig os-mlock.c
#         dkms remove nvidia/VERSION --all
#         dkms install nvidia/VERSION
#         reboot
# =========================================================================
if [ "$START_PHASE" -le 3 ] && ! $NVME_ONLY; then
    header "Phase 3: NVIDIA DKMS Patch (os-mlock.c)"

    if [ -z "$NVIDIA_DKMS_VER" ]; then
        NVIDIA_DKMS_VER=$(modinfo nvidia 2>/dev/null | grep "^version:" | awk '{print $2}' || echo "")
    fi

    if [ -z "$NVIDIA_DKMS_VER" ]; then
        warn "Cannot detect NVIDIA driver version — install nvidia-dkms first"
    else
        DKMS_SRC="/usr/src/nvidia-${NVIDIA_DKMS_VER}/nvidia/os-mlock.c"

        if [ ! -f "$DKMS_SRC" ]; then
            warn "DKMS source not found at $DKMS_SRC"
            info "Check: ls /usr/src/nvidia-*/"
        else
            if grep -q "nv_follow_flavors" "$DKMS_SRC"; then
                ok "os-mlock.c already patched"
            else
                info "os-mlock.c needs patching (kernel $(uname -r) removed follow_pfn)"

                if ! $CHECK_ONLY; then
                    # Backup
                    cp "$DKMS_SRC" "${DKMS_SRC}.orig"
                    ok "Backup created: ${DKMS_SRC}.orig"

                    apply_osmlock_patch "$DKMS_SRC"
                    ok "os-mlock.c patched"

                    # Rebuild DKMS
                    info "Rebuilding NVIDIA DKMS module (1-3 minutes)..."
                    dkms remove "nvidia/${NVIDIA_DKMS_VER}" --all 2>/dev/null || true
                    dkms install "nvidia/${NVIDIA_DKMS_VER}" 2>&1 | tail -5
                    ok "DKMS module rebuilt"

                    echo ""
                    echo -e "  ${YELLOW}*** REBOOT REQUIRED ***${NC}"
                    echo "  Run: sudo reboot"
                    echo "  Then re-run: sudo $0 --phase 4"
                    echo ""
                    info "Rollback: cp ${DKMS_SRC}.orig ${DKMS_SRC}"
                    info "          dkms remove nvidia/${NVIDIA_DKMS_VER} --all"
                    info "          dkms install nvidia/${NVIDIA_DKMS_VER}"
                    exit 0
                fi
            fi
        fi
    fi


    # ── Phase 3b: CUDA Header Patch (rsqrt noexcept) ──
    #
    # Only needed on glibc >= 2.42 (Ubuntu 25.10+)
    echo ""
    info "Phase 3b: CUDA header patch (glibc rsqrt conflict)"

    CUDA_MATH_H="$CUDA_PATH/targets/x86_64-linux/include/crt/math_functions.h"

    if [ ! -f "$CUDA_MATH_H" ]; then
        warn "CUDA math_functions.h not found at $CUDA_MATH_H"
    else
        # Check glibc version
        GLIBC_VER=$(ldd --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+$' || echo "0")
        GLIBC_MAJOR=$(echo "$GLIBC_VER" | cut -d. -f1)
        GLIBC_MINOR=$(echo "$GLIBC_VER" | cut -d. -f2)

        if [ "$GLIBC_MAJOR" -ge 2 ] && [ "$GLIBC_MINOR" -ge 42 ] 2>/dev/null; then
            if grep -q "__GLIBC_USE_IEC_60559_FUNCS_EXT_C23" "$CUDA_MATH_H"; then
                ok "math_functions.h already patched (rsqrt noexcept)"
            else
                info "glibc $GLIBC_VER detected — rsqrt noexcept patch needed"

                if ! $CHECK_ONLY; then
                    cp "$CUDA_MATH_H" "${CUDA_MATH_H}.orig"
                    ok "Backup: ${CUDA_MATH_H}.orig"

                    apply_cuda_rsqrt_patch "$CUDA_MATH_H"
                    ok "math_functions.h patched (rsqrt/rsqrtf noexcept guards)"

                    info "Rollback: cp ${CUDA_MATH_H}.orig $CUDA_MATH_H"
                fi
            fi
        else
            ok "glibc $GLIBC_VER — rsqrt patch not needed (requires >= 2.42)"
        fi
    fi
fi


# =========================================================================
# PHASE 4: VFIO Module Loading (per-reboot)
#
# WHY:  VFIO (Virtual Function I/O) allows userspace programs to directly
#       access PCI device registers. gpu-nvme-direct uses VFIO to mmap the
#       NVMe BAR0 and expose it to CUDA via cudaHostRegisterIoMemory.
#       "Unsafe noiommu mode" is required because consumer motherboards
#       don't properly support IOMMU groups for PCIe passthrough.
#
# RISK: Enables userspace DMA without IOMMU protection. Combined with
#       phase 2 (IOMMU off), any process with access to /dev/vfio/* can
#       DMA to arbitrary memory. This is standard for VFIO/GPU passthrough.
#       Modules are not persisted — revert by rebooting.
# =========================================================================
if [ "$START_PHASE" -le 4 ]; then
    header "Phase 4: VFIO Modules (per-reboot)"

    if lsmod | grep -q "^vfio_pci "; then
        ok "vfio-pci already loaded"
    else
        if ! $CHECK_ONLY; then
            modprobe vfio enable_unsafe_noiommu_mode=1
            ok "vfio loaded (unsafe noiommu mode)"
            modprobe vfio-pci
            ok "vfio-pci loaded"
        else
            warn "vfio-pci not loaded"
            info "Load with: modprobe vfio enable_unsafe_noiommu_mode=1 && modprobe vfio-pci"
        fi
    fi

    if [ -f /sys/module/vfio/parameters/enable_unsafe_noiommu_mode ]; then
        NOIOMMU=$(cat /sys/module/vfio/parameters/enable_unsafe_noiommu_mode)
        if [ "$NOIOMMU" = "Y" ] || [ "$NOIOMMU" = "1" ]; then
            ok "Unsafe noiommu mode: ON"
        else
            if ! $CHECK_ONLY; then
                echo 1 > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode
                ok "Unsafe noiommu mode: enabled"
            else
                warn "Noiommu mode OFF — gpu-nvme-direct requires it"
            fi
        fi
    fi
fi


# =========================================================================
# PHASE 5: NVMe Device Setup (per-reboot)
#
# WHY:  The NVMe must be unbound from the kernel nvme driver and bound to
#       vfio-pci so userspace (gpu-nvme-direct) can access BAR0 directly.
#       After unbinding, the device enters PCIe power state D3 (sleep);
#       we force D0 and enable Memory+BusMaster in the PCI command register.
#
# RISK: The NVMe will NOT appear as /dev/nvmeX while bound to vfio-pci.
#       Any data on it is still intact but inaccessible to the OS.
#       NEVER run this on your boot drive — the script checks and refuses.
#       To restore: sudo ./scripts/restore_nvme.sh [BDF]
# =========================================================================
if [ "$START_PHASE" -le 5 ]; then
    header "Phase 5: NVMe Device ($NVME_BDF)"

    SYSFS="/sys/bus/pci/devices/$NVME_BDF"

    if [ ! -d "$SYSFS" ]; then
        fail "Device $NVME_BDF not found"
        echo "  Available NVMe devices:"
        lspci -nn | grep -i "NVMe\|Non-Volatile" | sed 's/^/    /'
        exit 1
    fi

    DEVNAME=$(lspci -s "$NVME_BDF" 2>/dev/null | cut -d: -f3- | sed 's/^ //')
    info "Found: $DEVNAME"

    # Safety: refuse boot drive
    ROOT_DEV=$(findmnt -n -o SOURCE / 2>/dev/null || echo "")
    if echo "$ROOT_DEV" | grep -q "nvme"; then
        ROOT_NVME=$(readlink -f "/sys/block/$(echo "$ROOT_DEV" | sed 's|/dev/||;s|p[0-9]*||')/device/device" 2>/dev/null || echo "")
        if echo "$ROOT_NVME" | grep -q "$NVME_BDF"; then
            echo ""
            fail "*** $NVME_BDF IS YOUR BOOT DRIVE ***"
            fail "Binding it to VFIO would make your system unbootable."
            fail "Use --nvme-bdf to specify a DIFFERENT NVMe device."
            exit 1
        fi
    fi

    CURRENT_DRV=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")

    if [ "$CURRENT_DRV" = "vfio-pci" ]; then
        ok "Already bound to vfio-pci"
    elif ! $CHECK_ONLY; then
        if [ "$CURRENT_DRV" != "none" ]; then
            info "Unbinding from $CURRENT_DRV..."
            echo "$NVME_BDF" > "$SYSFS/driver/unbind" 2>/dev/null || true
            sleep 0.5
        fi

        VENDOR=$(cat "$SYSFS/vendor" 2>/dev/null | sed 's/0x//')
        DEVICE=$(cat "$SYSFS/device" 2>/dev/null | sed 's/0x//')
        echo "$VENDOR $DEVICE" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
        echo "$NVME_BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
        sleep 0.5

        CURRENT_DRV=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")
        if [ "$CURRENT_DRV" = "vfio-pci" ]; then
            ok "Bound to vfio-pci"
        else
            fail "Binding failed (driver=$CURRENT_DRV)"
            exit 1
        fi
    else
        warn "NVMe bound to '$CURRENT_DRV' (needs vfio-pci)"
    fi

    # Force power state D0 + enable Memory/BusMaster
    if ! $CHECK_ONLY; then
        echo on > "$SYSFS/power/control" 2>/dev/null || true
        setpci -s "$NVME_BDF" 0x84.W=0x0008 2>/dev/null || warn "setpci PM control failed"
        ok "Power state: D0"

        setpci -s "$NVME_BDF" COMMAND=0x0006 2>/dev/null || warn "setpci COMMAND failed"
        CMD=$(setpci -s "$NVME_BDF" COMMAND 2>/dev/null || echo "????")
        ok "PCI COMMAND: 0x$CMD (Memory Space + Bus Master)"
    fi

    # Verify BAR0
    if [ -f "$SYSFS/resource0" ]; then
        BAR0_SIZE=$(stat -c%s "$SYSFS/resource0" 2>/dev/null || echo "0")
        if [ "$BAR0_SIZE" -gt 0 ]; then
            ok "BAR0 accessible ($BAR0_SIZE bytes)"
        else
            warn "BAR0 exists but size=0 (device may be in D3 still)"
        fi
    else
        fail "BAR0 (resource0) not found — Above 4G Decoding may be OFF in BIOS"
    fi

    info "Restore NVMe to kernel: sudo $SCRIPT_DIR/restore_nvme.sh $NVME_BDF"
fi


# =========================================================================
# PHASE 6: Build
# =========================================================================
if [ "$START_PHASE" -le 6 ] && ! $NVME_ONLY; then
    header "Phase 6: Build"

    # Detect GPU architecture for CUDA
    GPU_ARCH="86"  # Default: sm_86 (Ampere, RTX 3090)
    if command -v nvidia-smi &>/dev/null; then
        CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo "")
        CC_MINOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f2 || echo "")
        if [ -n "$CC_MAJOR" ] && [ -n "$CC_MINOR" ]; then
            GPU_ARCH="${CC_MAJOR}${CC_MINOR}"
            info "Detected GPU compute capability: ${CC_MAJOR}.${CC_MINOR} (sm_${GPU_ARCH})"
        fi
    fi

    CMAKE_OPTS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc"
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14
        -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH"
    )

    if $CHECK_ONLY; then
        info "Would build with: ${CMAKE_OPTS[*]}"
    else
        # gpu-nvme-direct
        if [ -n "$GPUNVME_DIR" ] && [ -f "$GPUNVME_DIR/CMakeLists.txt" ]; then
            info "Building gpu-nvme-direct (hardware mode)..."
            sudo -u "$REAL_USER" mkdir -p "$GPUNVME_DIR/build-hw"
            cd "$GPUNVME_DIR/build-hw"
            sudo -u "$REAL_USER" cmake .. "${CMAKE_OPTS[@]}" -DGPUNVME_USE_SIM=OFF 2>&1 | tail -3
            sudo -u "$REAL_USER" cmake --build . -j"$(nproc)" 2>&1 | tail -3
            ok "gpu-nvme-direct built"
            cd "$NT_DIR"
        else
            warn "gpu-nvme-direct not found at $GPUNVME_DIR"
            info "ntransformer will build without NVMe backend"
        fi

        # ntransformer
        info "Building ntransformer..."
        sudo -u "$REAL_USER" mkdir -p "$NT_DIR/build"
        cd "$NT_DIR/build"

        NT_CMAKE_OPTS=("${CMAKE_OPTS[@]}")
        if [ -n "$GPUNVME_DIR" ] && [ -d "$GPUNVME_DIR/build-hw" ]; then
            NT_CMAKE_OPTS+=(-DUSE_GPUNVME=ON)
            info "NVMe backend: ENABLED"
        fi

        sudo -u "$REAL_USER" cmake .. "${NT_CMAKE_OPTS[@]}" 2>&1 | tail -3
        sudo -u "$REAL_USER" cmake --build . -j"$(nproc)" 2>&1 | tail -3
        ok "ntransformer built"
    fi
fi


# =========================================================================
# PHASE 7: Verification
# =========================================================================
if [ "$START_PHASE" -le 7 ]; then
    header "Phase 7: Verification"

    TESTS_PASS=0
    TESTS_FAIL=0

    # GPU
    if nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        ok "GPU: $GPU_NAME"
        ((TESTS_PASS++))
    else
        fail "nvidia-smi failed — NVIDIA driver not loaded"
        ((TESTS_FAIL++))
    fi

    # VFIO
    if lsmod | grep -q "^vfio_pci "; then
        ok "VFIO modules loaded"
        ((TESTS_PASS++))
    else
        warn "VFIO not loaded (run with --nvme-only or phase 4)"
    fi

    # BAR0
    if [ -x "${GPUNVME_DIR}/build-hw/dump_bar0" ]; then
        CURRENT_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$NVME_BDF/driver" 2>/dev/null)" 2>/dev/null || echo "none")
        if [ "$CURRENT_DRV" = "vfio-pci" ]; then
            if "${GPUNVME_DIR}/build-hw/dump_bar0" "$NVME_BDF" 2>/dev/null | grep -q "VS:"; then
                ok "NVMe BAR0 readable"
                ((TESTS_PASS++))
            else
                fail "BAR0 unreadable (device may be in D3)"
                ((TESTS_FAIL++))
            fi
        else
            info "Skipping BAR0 test (NVMe not on vfio-pci)"
        fi
    fi

    # cudaHostRegisterIoMemory
    if [ -x "${GPUNVME_DIR}/build-hw/check_p2p" ]; then
        CURRENT_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$NVME_BDF/driver" 2>/dev/null)" 2>/dev/null || echo "none")
        if [ "$CURRENT_DRV" = "vfio-pci" ]; then
            if "${GPUNVME_DIR}/build-hw/check_p2p" "$NVME_BDF" 2>/dev/null | grep -q "SUCCESS"; then
                ok "cudaHostRegisterIoMemory: WORKS (DKMS patch OK)"
                ((TESTS_PASS++))
            else
                fail "cudaHostRegisterIoMemory FAILED — DKMS patch needed (phase 3)"
                ((TESTS_FAIL++))
            fi
        fi
    fi

    # ntransformer binary
    if [ -x "$NT_DIR/build/ntransformer" ]; then
        ok "ntransformer binary: OK"
        ((TESTS_PASS++))
    else
        warn "ntransformer not built (run phase 6)"
    fi

    echo ""
    echo "────────────────────────────────────────────"
    echo -e "  Results: ${GREEN}$TESTS_PASS passed${NC}, ${RED}$TESTS_FAIL failed${NC}"

    if [ "$TESTS_FAIL" -gt 0 ]; then
        echo ""
        echo "  Fix failed tests before running ntransformer with NVMe backend."
    fi
fi


# =========================================================================
# Summary
# =========================================================================
header "Done"

if $NVME_ONLY; then
    echo "NVMe device $NVME_BDF ready for gpu-nvme-direct."
    echo ""
    echo "Run ntransformer:"
    echo "  sudo GPUNVME_PCI_BDF=$NVME_BDF GPUNVME_GGUF_LBA=0 \\"
    echo "    ./build/ntransformer -m /path/to/model.gguf --streaming -p \"Hello\" -n 32"
else
    echo "System configured. Quick reference:"
    echo ""
    echo "  # After reboot — setup NVMe (required every boot):"
    echo "  sudo $0 --nvme-only --nvme-bdf $NVME_BDF"
    echo ""
    echo "  # Write model to NVMe (one-time):"
    echo "  sudo $SCRIPT_DIR/restore_nvme.sh $NVME_BDF   # rebind to kernel driver"
    echo "  sudo dd if=model.gguf of=/dev/nvme1n1 bs=1M status=progress"
    echo "  sudo $0 --nvme-only                           # rebind to vfio-pci"
    echo ""
    echo "  # Run inference (model fits in VRAM+RAM, no NVMe needed):"
    echo "  ./build/ntransformer -m model.gguf --streaming -p \"Hello\" -n 32"
    echo ""
    echo "  # Run with layer skip (fastest for 70B, 2.5x speedup):"
    echo "  ./build/ntransformer -m model-q4_k_m.gguf --streaming --skip-threshold 0.98 -p \"Hello\""
    echo ""
    echo "  # Run with NVMe backend (model > VRAM+RAM):"
    echo "  sudo GPUNVME_PCI_BDF=$NVME_BDF GPUNVME_GGUF_LBA=0 \\"
    echo "    ./build/ntransformer -m model.gguf --streaming -p \"Hello\" -n 32"
    echo ""
    echo "  # Restore NVMe to normal:"
    echo "  sudo $SCRIPT_DIR/restore_nvme.sh $NVME_BDF"
fi
