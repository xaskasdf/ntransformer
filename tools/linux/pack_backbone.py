import struct, json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\Users\xasko\AppData\Local\Temp\opencode')
from build_lba_index import parse_shard_header, nbytes, all_tensors, base, shards

# backbone = todo menos los 3 tensores de expertos enrutados por capa
def is_routed(name):
    return name.startswith('blk.') and (
        name.endswith('ffn_gate_exps.weight') or
        name.endswith('ffn_up_exps.weight') or
        name.endswith('ffn_down_exps.weight'))

out_path = r'E:\glm5_backbone.bin'
manifest = {'tensors': {}}
total = 0
skipped = 0
with open(out_path, 'wb') as out:
    for name in sorted(all_tensors.keys()):
        if is_routed(name):
            skipped += 1
            continue
        fname, off = all_tensors[name]
        with open(os.path.join(base, fname), 'rb') as fh:
            fh.seek(off)
            # tamano: leer dims del header del shard ya parseado — re-parse por tensor es caro;
            # usar el mapa de (name -> (dims, tt)) que reconstruimos
            pass
    # (se completa abajo con el segundo pase)

# segundo pase con dims: re-parsear headers una vez y guardar dims
dims_map = {}
for si, f in enumerate(shards):
    with open(os.path.join(base, f), 'rb') as fh:
        buf = fh.read(64 << 22)
    tensors, data_off = parse_shard_header(buf)
    for name, dims, tt, off in tensors:
        n = 1
        for d in dims: n *= d
        dims_map[name] = (dims, tt, nbytes(tt, n))

with open(out_path, 'wb') as out:
    pos = 0
    for name in sorted(all_tensors.keys()):
        if is_routed(name):
            continue
        fname, off = all_tensors[name]
        dims, tt, nb = dims_map[name]
        with open(os.path.join(base, fname), 'rb') as fh:
            fh.seek(off)
            data = fh.read(nb)
        out.write(data)
        manifest['tensors'][name] = {
            'offset': pos, 'bytes': nb, 'dims': dims, 'ggml_type': tt, 'shard': fname
        }
        pos += nb
        if pos % (1 << 30) < nb:
            print(f'  {pos/1e9:.1f} GB', flush=True)

manifest['total_bytes'] = pos
with open(r'E:\glm5_backbone_manifest.json', 'w') as f:
    json.dump(manifest, f)
print(f'backbone: {out_path} {pos/1e9:.2f} GB, {len(manifest["tensors"])} tensores, {skipped} enrutados excluidos')
