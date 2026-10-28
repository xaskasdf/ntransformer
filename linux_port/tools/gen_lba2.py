#!/usr/bin/env python3
# gen_lba2.py — convierte expert_lba_index.json (regen FIEMAP) a binario LBA2
#
# Uso:
#   python3 gen_lba2.py <index.json> <out.bin> 1
#       todo al disco 0 (equivalente funcional al LBA1 actual)
#   python3 gen_lba2.py <index.json> <out.bin> 2 <layout.json>
#       experts repartidos según layout.json (produce fill_second_disk.py),
#       donde layout = {"slice_bytes": N, "tensors": {"<il>_<kind>": {"disk": 0|1}}}
#       (por TENSOR, no por experto: gate/up/down de una capa van juntos)
#
# Formato LBA2: 'LBA2' | u32 n_layers | u32 n_experts | u32 slice_bytes
#   por (layer,kind) en orden 3..44 × gate,up,down:
#     por expert: u32 in_off | u32 n_runs | runs: u64 lba, u32 nbytes, u8 disk
import json, struct, sys

def main():
    idx_path, out_path = sys.argv[1], sys.argv[2]
    n_disks = int(sys.argv[3])
    layout = None
    if n_disks == 2:
        layout = json.load(open(sys.argv[4]))
        assert layout.get('n_disks') == 2, 'layout mismatch'
    idx = json.load(open(idx_path))
    layers = sorted(idx['layers'], key=int)
    slice_bytes = idx['layers'][layers[0]]['gate']['slice_bytes']
    if layout:
        assert layout['slice_bytes'] == slice_bytes, 'slice_bytes mismatch'

    out = open(out_path, 'wb')
    out.write(struct.pack('<4sIII', b'LBA2', len(layers), 288, slice_bytes))
    for layer in layers:
        lv = idx['layers'][layer]
        for kind in ('gate', 'up', 'down'):
            t = lv[kind]
            tkey = f'{layer}_{kind}'
            lt = layout['tensors'].get(tkey, {'disk': 0}) if layout else {'disk': 0}
            for e, exp in enumerate(t['experts']):
                if lt.get('disk') == 1:
                    # tensor migrado al disco 1: el run del experto e (alineado, in_off=0)
                    r = lt['runs'][e]
                    out.write(struct.pack('<II', 0, 1))
                    out.write(struct.pack('<QIB', r[0], r[1], 1))
                else:
                    out.write(struct.pack('<II', exp['in_off'], len(exp['runs'])))
                    for r in exp['runs']:
                        out.write(struct.pack('<QIB', r[0], r[1], 0))
    out.close()
    import os
    print(f'{out_path}: {os.path.getsize(out_path)/1e6:.1f}MB, {len(layers)} capas, {n_disks} disco(s)')

if __name__ == '__main__':
    main()
