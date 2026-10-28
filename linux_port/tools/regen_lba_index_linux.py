#!/usr/bin/env python3
# regen_lba_index_linux.py — regenera expert_lba_index.json desde Linux.
# FIEMAP (ioctl read-only) sobre los GGUF montados ro. NO escribe en E:.
# LBAs = absolutos del disco (fiemap physical es desde el block device).
import json, struct, ctypes, os, sys
PART_OFFSET = 2048  # nvme1n1p1 start sector (/sys/block/nvme1n1/nvme1n1p1/start)

FIEMAP_IOCTL = 0xC020660B
libc = ctypes.CDLL(None, use_errno=True)

# struct fiemap: u64 start, u64 length, u32 flags, u32 mapped, u32 count, u32 res, extents[]
FM_HDR = 32
EXT_SIZE = 56  # u64 logical, u64 physical, u64 length, u64 res64[2], u32 flags, u32 res[3]

def fiemap_call(fd, start, length, count=512):
    buf = ctypes.create_string_buffer(FM_HDR + EXT_SIZE * count)
    struct.pack_into('<QQIIII', buf, 0, start, length, 0, 0, count, 0)
    err = libc.ioctl(fd, FIEMAP_IOCTL, buf)
    if err != 0:
        raise OSError(ctypes.get_errno(), 'FIEMAP failed')
    mapped = struct.unpack_from('<QQIIII', buf, 0)[3]
    exts = []
    for i in range(mapped):
        logical, physical, length, _r1, _r2, flags, _r3, _r4 = struct.unpack_from('<4QI3I', buf, FM_HDR + i * EXT_SIZE)
        exts.append((logical, physical, length, flags))
    return exts

def fiemap_extents(fd, size):
    out, start = [], 0
    while start < size:
        exts = fiemap_call(fd, start, size - start)
        if not exts:
            break
        for logical, physical, length, flags in exts:
            out.append((logical, physical, length))
            if logical + length > start:
                start = logical + length
        if len(exts) < 1:
            break
    return out

def runs_for_range(exts, off, nbytes, sector=512):
    """runs [[lba_abs, nbytes], ...] cubriendo los sectores [floor(off/512), ceil((off+nbytes)/512))
    del disco. El slice empieza en in_off = off % 512 del primer sector."""
    # lba0 real = runs[0][0] (fisico); este valor logico no se usa
    start = (off // sector) * sector
    end = ((off + nbytes + sector - 1) // sector) * sector
    runs, pos = [], start
    for logical, physical, length in exts:
        lend = logical + length
        if lend <= pos: continue
        if logical >= end: break
        s, e = max(pos, logical), min(end, lend)
        if e > s:
            lba = (physical + (s - logical)) // sector + PART_OFFSET
            n = e - s
            if runs and runs[-1][0] + runs[-1][1] // sector == lba:
                runs[-1][1] += n
            else:
                runs.append([lba, n])
        pos = e
        if pos >= end: break
    if pos < end:
        raise Exception(f'hueco: slice [{off},{end}) no cubierto')
    in_off = off - start
    return runs[0][0], in_off, runs

def main():
    idx_path = sys.argv[1] if len(sys.argv) > 1 else '/media/ubuntu/cp/linux_kit/expert_lba_index.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/glm5/expert_lba_index.json')
    old = json.load(open(idx_path))
    base = '/mnt/e/Q8_0/'
    files = {}
    new_layers = {}
    sectors = old['bytes_per_sector']
    n_slices = n_runs = n_multi = 0
    for layer in sorted(old['layers'], key=int):
        lv = old['layers'][layer]
        nl = {}
        for key in ('gate', 'up', 'down'):
            t = lv[key]
            if t['file'] not in files:
                fd = os.open(base + t['file'], os.O_RDONLY)
                size = os.lseek(fd, 0, os.SEEK_END)
                files[t['file']] = (fd, fiemap_extents(fd, size))
                print(f'fiemap {t["file"]}: {len(files[t["file"]][1])} extents', file=sys.stderr)
            fd, exts = files[t['file']]
            experts = []
            sb = t['slice_bytes']
            for _ in t['experts']:
                e_idx = len(experts)
                off = t['tensor_offset'] + e_idx * sb
                lba0, in_off, runs = runs_for_range(exts, off, sb, sectors)
                experts.append({'lba0': lba0, 'in_off': in_off, 'runs': runs})
                n_slices += 1; n_runs += len(runs)
                if len(runs) > 1: n_multi += 1
            nl[key] = {'file': t['file'], 'tensor_offset': t['tensor_offset'],
                       'slice_bytes': sb, 'experts': experts}
        new_layers[layer] = nl
        print(f'layer {layer} ok', file=sys.stderr)
    for fd, _ in files.values(): os.close(fd)
    out = {'bytes_per_sector': sectors, 'cluster_bytes': old['cluster_bytes'],
           'partition_lba_offset': 0, 'layers': new_layers,
           'note': 'regenerado en Linux via FIEMAP; LBAs absolutos del disco'}
    json.dump(out, open(out_path, 'w'))
    print(f'slices={n_slices} runs={n_runs} multi-run={n_multi} -> {out_path}')

if __name__ == '__main__':
    main()
