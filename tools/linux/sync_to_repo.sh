#!/bin/bash
# sync_to_repo.sh — sincroniza la integración gpnvme (~/src/ntransformer, ~/gpu-nvme-direct)
# al repo persistente en D: (montado via ldmtool). Correr DESDE LINUX, como root o con sudo.
#
# Uso: sudo bash ~/sync_to_repo.sh
#
# Requiere: ldmtool, rsync. El span de Windows D: aparece como volumen LDM.

set -euo pipefail

REPO_WIN="/mnt/winrepo/ntransformer"      # destino (montar abajo)
GPNVME_WIN="/mnt/winrepo/gpu-nvme-direct"
SRC="$HOME/src/ntransformer"              # fuentes de la sesión Linux
GPNVME_SRC="$HOME/gpu-nvme-direct"        # clon gpu-nvme-direct con el patch del stream

echo "== 1. montar D: (span LDM de Windows) =="
if [ ! -d "$REPO_WIN" ]; then
    mkdir -p "$REPO_WIN" "$GPNVME_WIN"
    # ldmtool descubre el span; el nombre del volumen puede variar — ver 'ldmtool scan'
    VOL=$(ldmtool scan all 2>/dev/null | grep -oP 'f2d76644[0-9a-f-]+' | head -1 || true)
    if [ -z "$VOL" ]; then
        echo "ldmtool no encontró el span. Correr 'ldmtool scan' manual y ajustar VOL."
        exit 1
    fi
    ldmtool create "$VOL" Volume1 || true
    # el device creado (buscar en /dev/mapper)
    DEV=$(ls /dev/mapper/ | grep -i "ldmvol" | head -1)
    mount -t ntfs3 -o rw "/dev/mapper/$DEV" /mnt/winrepo
fi

echo "== 2. respaldo del estado actual del repo =="
tar czf "$HOME/repo_backup_$(date +%Y%m%d_%H%M).tgz" -C "$REPO_WIN" src docs 2>/dev/null || true

echo "== 3. archivos de la integración gpnvme (ntransformer) =="
FILES=(
    "src/model/glm5/glm5_gpu.cu"          # fill_slice rama gpnvme + GLM5_VERIFY
    "src/model/glm5/glm5_gpu.h"
    "src/model/glm5/glm5_model.cpp"       # pack mode (GLM5_BACKBONE_PACK)
    "src/model/glm5/glm5_model.h"
    "src/model/glm5/glm5_forward.cpp"     # cambios del run_cli pack mode
    "src/model/glm5/glm5_forward.h"
    "CMakeLists.txt"                       # si cambió (USE_GPUNVME / link)
)
for f in "${FILES[@]}"; do
    if [ -f "$SRC/$f" ]; then
        if ! cmp -s "$SRC/$f" "$REPO_WIN/$f" 2>/dev/null; then
            cp -v "$SRC/$f" "$REPO_WIN/$f"
        fi
    fi
done

echo "== 4. scripts de la sesión (a tools/linux/) =="
mkdir -p "$REPO_WIN/tools/linux"
for f in "$HOME/regen_lba_index_linux.py" "$HOME/run_glm5.sh" "$HOME/post_reboot.sh" \
         "$HOME/glm5/glm5_lba.bin"; do
    [ -f "$f" ] && cp -v "$f" "$REPO_WIN/tools/linux/"
done

echo "== 5. patch del layer_loader (gpu-nvme-direct) =="
if [ -f "$GPNVME_SRC/src/host/layer_loader.cu" ]; then
    diff -u "$GPNVME_WIN/src/host/layer_loader.cu" "$GPNVME_SRC/src/host/layer_loader.cu" \
        > "$REPO_WIN/tools/linux/layer_loader_stream_fix.patch" || true
fi

echo "== 6. diff final (verificación) =="
echo "archivos cambiados vs sesión Linux:"
for f in "${FILES[@]}"; do
    [ -f "$SRC/$f" ] && ! cmp -s "$SRC/$f" "$REPO_WIN/$f" 2>/dev/null && echo "  DIFIERE: $f"
done
echo "listo. El repo en D: tiene la integración; C:\\linux_kit\\GLM5_LINUX_BENCH.md ya está documentado."

# NOTA: no desmontar D: si se va a seguir usando; umount /mnt/winrepo al final si hace falta
