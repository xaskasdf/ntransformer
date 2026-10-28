#!/bin/bash
# run_glm5.sh v2 — dos discos para los expertos (stripe por capas):
#   disco 0: SN5000S (0b:00.0, serial 25306P803002) — capas pares + raw store
#   disco 1: SN520   (01:00.0, serial 193562805988) — capas impares
# Hace: bootstrap del backbone (E: -> /dev/shm si falta) -> wake de ambos ->
#        vfio rebind de ambos -> setpci 84.W+COMMAND con read-back -> exec.
# Uso: sudo bash run_glm5.sh [args de ntransformer...]
set -uo pipefail

find_bdf() {  # por serial: nvme namespace -> controller dir -> PCI BDF
    for b in /sys/block/nvme*n1; do
        s=$(tr -d ' ' < $b/device/serial 2>/dev/null)
        [ "$s" = "$1" ] || continue
        # /sys/devices/pci.../0000:0b:00.0/nvme/nvme0/nvme0n1
        ns=$(readlink -f $b/device)
        pci_dir=$(dirname $(dirname $ns))
        basename $pci_dir
        return
    done
    return 1
}

wake_disk() {  # serial -> nvme kernel bind + dd (sacarlo de low-power)
    local serial=$1
    local bdev=$(lsblk -no NAME,SERIAL | awk -v s=$serial '$2==s{print $1; exit}')
    [ -z "$bdev" ] && return 1
    local ns=$(readlink -f /sys/class/block/$bdev/device)
    local BDF=$(basename $(dirname $(dirname $ns)))
    dd if=/dev/$bdev bs=512 count=1 status=none >/dev/null 2>&1 || true
    echo $BDF
}

SN5_SERIAL=25306P803002   # SN5000S 512G (raw store, disco 0)
SN5_SERIAL2=193562805988  # SN520 256G (disco 1)

# ---------- 1. backbone a /dev/shm (el E: va a VFIO, el archivo no accesible) ----------
mkdir -p /dev/shm/glm5-backup
if [ ! -f /dev/shm/glm5-backup/glm5_backbone.bin ]; then
    B0=$(find_bdf $SN5_SERIAL); [ -z "$B0" ] && { echo "SN5000S no está"; exit 1; }
    DRV=$(basename "$(readlink /sys/bus/pci/devices/$B0/driver 2>/dev/null)" 2>/dev/null || true)
    if [ "$DRV" != "nvme" ]; then
        echo "$B0" > /sys/bus/pci/drivers/nvme/bind 2>/dev/null || true; sleep 2
    fi
    BDEV=$(lsblk -no NAME,SERIAL | awk -v s=$SN5_SERIAL '$2==s{print $1; exit}')
    mkdir -p /mnt/e && mount -t ntfs3 -o ro /dev/${BDEV}p1 /mnt/e 2>/dev/null || true
    cp /mnt/e/glm5_backbone.bin /mnt/e/glm5_backbone_manifest.json /dev/shm/glm5-backup/ || { echo "copie del backbone falló"; exit 1; }
    umount /mnt/e 2>/dev/null
fi
H=$(sha256sum /dev/shm/glm5-backup/glm5_backbone.bin | awk '{print $1}')
[ "$H" = "6daac5cce05830935ba02a60350325e15abdfe860898bfb3ad4674cb27bc3a1d" ] || { echo "backbone sha256 MAL: $H"; exit 1; }
echo "backbone OK en /dev/shm ($H)"

# ---------- 2. wake + vfio + registros de ambos discos ----------
setup_disk() {  # serial bdf_env_name
    local serial=$1 envname=$2
    local BDF=$(wake_disk $serial)
    [ -z "$BDF" ] && { echo "disco $serial no encontrado"; return 1; }
    local CUR=$(basename "$(readlink /sys/bus/pci/devices/$BDF/driver 2>/dev/null)" 2>/dev/null || true)
    if [ "$CUR" != "vfio-pci" ]; then
        echo "$BDF" > /sys/bus/pci/devices/$BDF/driver/unbind 2>/dev/null || true
        modprobe vfio_pci 2>/dev/null || true
        echo "$(cat /sys/bus/pci/devices/$BDF/vendor) $(cat /sys/bus/pci/devices/$BDF/device)" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
        echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
        sleep 1
    fi
    echo on > /sys/bus/pci/devices/$BDF/power/control 2>/dev/null
    local SHORT=${BDF#0000:}
    setpci -s $SHORT 84.W=0x0008 2>/dev/null || true          # el SN520 puede no tenerlo
    setpci -s $SHORT COMMAND=0x0006
    local V1=$(setpci -s $SHORT COMMAND 2>/dev/null)
    [ "$V1" = "0006" ] || { echo "COMMAND no aplicó en $BDF ($V1)"; return 1; }
    export $envname=$SHORT
    echo "disco $serial: $BDF vfio-pci COMMAND=$V1"
}
setup_disk $SN5_SERIAL  GPUNVME_PCI_BDF  || exit 1
setup_disk $SN5_SERIAL2 GPUNVME_PCI_BDF1 || { echo "disco 1 falló"; exit 1; }

# ---------- 3. entorno y exec ----------
export GLM5_BACKBONE_PACK=${GLM5_BACKBONE_PACK:-/dev/shm/glm5-backup/pack}
export GLM5_GPNVME=${GLM5_GPNVME:-1}
export GLM5_LBA_BIN=${GLM5_LBA_BIN:-/home/ubuntu/glm5/glm5_lba2_two.bin}
export GLM5_GPU=${GLM5_GPU:-1}
export GLM5_CHUNK=${GLM5_CHUNK:-3}  # workaround: race del union load con chunks grandes (ver GLM5_LINUX_BENCH.md §10)

# symlink del backbone para el pack en RAM
mkdir -p /dev/shm/glm5-backup/pack
for f in /home/ubuntu/glm5/pack/*; do
    b=$(basename $f)
    [ "$b" = "glm5_backbone.bin" ] && { ln -sfn /dev/shm/glm5-backup/glm5_backbone.bin /dev/shm/glm5-backup/pack/$b; continue; }
    [ "$b" = "glm5_backbone_manifest.json" ] && { ln -sfn /dev/shm/glm5-backup/glm5_backbone_manifest.json /dev/shm/glm5-backup/pack/$b; continue; }
    ln -sfn "$f" /dev/shm/glm5-backup/pack/$b
done

BIN=${BIN:-/home/ubuntu/ntransformer-gpnvme}
MODEL=/dev/shm/glm5-backup/pack/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf
exec sudo -E "$BIN" -m "$MODEL" "$@"
