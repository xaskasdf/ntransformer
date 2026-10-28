#!/bin/bash
# post_reboot.sh — v2: tras el reboot (MOK enrolado o SB desactivado en BIOS)
# Uso: sudo bash ~/post_reboot.sh
set -uo pipefail

ok(){ echo -e "\033[32m[OK]\033[0m $*"; }
warn(){ echo -e "\033[33m[WARN]\033[0m $*"; }

# ---------------------------------------------------------------- 1. driver NVIDIA
modprobe nvidia 2>/dev/null; modprobe nvidia-uvm 2>/dev/null
if nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null)"
else
    dmesg | grep -iE 'nvidia|lockdown|module.*reject|integrity' | tail -5
    warn "driver NVIDIA NO cargó"; exit 1
fi

# ---------------------------------------------------------------- 2. monturas (re-hacer si faltan)
mountpoint -q /media/ubuntu/cp1 || mount -t ntfs3 -o ro /dev/nvme1n1p2 /media/ubuntu/cp1 2>/dev/null
[ -e /media/ubuntu/cp ] || ln -sfn cp1 /media/ubuntu/cp
mountpoint -q /media/ubuntu/cp1 && ok "C: ro (linux_kit en /media/ubuntu/cp/linux_kit)" || warn "C: falta"
if ! mountpoint -q /mnt/d; then
    DG=$(ldmtool scan 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)[0])' 2>/dev/null)
    [ -n "$DG" ] && ldmtool create volume "$DG" Volume1 >/dev/null 2>&1
    mkdir -p /mnt/d
    mount -t ntfs3 -o ro /dev/mapper/ldm_vol_FLYINGWHALES-Dg0_Volume1 /mnt/d 2>/dev/null
fi
mountpoint -q /mnt/d && ok "D: ro (fuentes en /mnt/d/projects)" || warn "D: falta"

# ---------------------------------------------------------------- 3. VFIO del SN5000S
BDF=$(lspci | grep -i 'SN5000' | awk '{print $1}' | head -1)
S=/sys/bus/pci/devices/0000:$BDF
CUR=$(basename $(readlink $S/driver 2>/dev/null) 2>/dev/null)
if [ "$CUR" != "vfio-pci" ]; then
    echo "0000:$BDF" > $S/driver/unbind 2>/dev/null
    modprobe vfio_pci 2>/dev/null
    echo "$(cat $S/vendor) $(cat $S/device)" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null
    echo "0000:$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null
    sleep 1
fi
CUR=$(basename $(readlink $S/driver 2>/dev/null) 2>/dev/null)
if [ "$CUR" = "vfio-pci" ]; then
    echo on > $S/power/control 2>/dev/null
    setpci -s $BDF 84.W=0x0008 2>&1 | grep -q failed && warn "setpci 84.W rechazada (lockdown?)"
    setpci -s $BDF COMMAND=0x0006 2>&1 | grep -q failed && warn "setpci COMMAND rechazada"
    ok "SN5000S vfio-pci BDF=$BDF BAR0=$(wc -c < $S/resource0 2>/dev/null)B"
else
    warn "SN5000S driver=$CUR"; exit 1
fi

echo ""
echo -e "\033[32m=== Listo para el probe: ===\033[0m"
echo "  sudo ~/probe $BDF /media/ubuntu/cp/linux_kit/expert_lba_index.json"
