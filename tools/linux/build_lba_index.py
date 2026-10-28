import ctypes, struct, json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

k32 = ctypes.windll.kernel32
# bytes por cluster de E:
spc, bps = ctypes.c_uint(0), ctypes.c_uint(0)
k32.GetDiskFreeSpaceW('E:\\', ctypes.byref(spc), ctypes.byref(bps), None, None)
SPC = spc.value
BPS = bps.value
CBC = SPC * BPS
print(f'sector={BPS}B cluster={CBC}B')

with open(r'C:\Users\xasko\AppData\Local\Temp\opencode\gguf_lba_map.json') as f:
    lba_map = json.load(f)

# --- parser GGUF (reutilizable): tensor name -> (shard_file, offset_en_shard, nbytes) ---
def parse_shard_header(buf):
    i = [0]
    def u32():
        v = struct.unpack_from('<I', buf, i[0])[0]; i[0] += 4; return v
    def u64():
        v = struct.unpack_from('<Q', buf, i[0])[0]; i[0] += 8; return v
    def s():
        n = u64(); v = buf[i[0]:i[0]+n].decode(); i[0] += n; return v
    def val(t):
        if t == 8: return s()
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(t, 0)
        if t == 9:
            et = u32(); n = u64()
            for _ in range(n): val(et)
            return
        i[0] += sz if sz else 0
        if sz == 0 and t not in (8, 9): raise Exception(f't{t}')
    assert u32() == 0x46554747
    u32(); ntens = u64(); nkv = u64()
    for _ in range(nkv):
        s(); val(u32())
    tensors = []
    for _ in range(ntens):
        name = s(); nd = u32(); dims = [u64() for _ in range(nd)]
        tt = u32(); off = u64()
        tensors.append((name, dims, tt, off))
    align = 32
    data_off = (i[0] + align - 1) // align * align
    return tensors, data_off

base = r'E:\Q8_0'
shards = sorted(f for f in os.listdir(base) if f.endswith('.gguf'))

# todos los tensores: name -> (file, file_offset_abs)
# data_offset del shard: ojo — tensor offsets son relativos al data section del shard
all_tensors = {}
for si, f in enumerate(shards):
    with open(os.path.join(base, f), 'rb') as fh:
        buf = fh.read(64 << 22)
    tensors, data_off = parse_shard_header(buf)
    for name, dims, tt, off in tensors:
        all_tensors[name] = (f, data_off + off)

# nbytes por tipo GGML
def nbytes(tt, n):
    if tt == 0: return n*4
    if tt == 1: return n*2
    if tt == 24: return n*2
    if tt == 8: return (n//32)*34
    if tt in (2,): return (n//32)*18
    if tt in (12,): return (n//256)*144
    if tt in (14,): return (n//256)*210
    raise Exception(f'type {tt}')

# --- indice experto -> LBA contiguo ---
H, FF = 4096, 2048
gate_slice = FF * (H // 32) * 34      # 8,912,896
down_slice = H * (FF // 32) * 34      # 8,912,896

def file_to_phys(fname, off, ln):
    """rango de archivo -> lista de (lba, nbytes) usando runs"""
    fm = lba_map['files'][fname]
    runs = fm['runs']
    res = []
    end = off + ln
    for vcn, lcn, ncl in runs:
        rs, re_ = vcn * CBC, (vcn + ncl) * CBC
        if re_ <= off or rs >= end:
            continue
        s = max(rs, off); e = min(re_, end)
        lba = (lcn * CBC + (s - rs)) // 512
        res.append((lba, e - s))
    return res

index = {'bytes_per_sector': 512, 'cluster_bytes': CBC, 'layers': {}}
bad = 0
total_slices = 0
for il in range(3, 45):
    p = f'blk.{il}.'
    entry = {}
    for tname, key in [('ffn_gate_exps.weight', 'gate'),
                       ('ffn_up_exps.weight', 'up'),
                       ('ffn_down_exps.weight', 'down')]:
        fname, off = all_tensors[p + tname]
        total = 1
        for _ in (): pass
        # el tensor es expert-major: [.., n_expert], expert e en off + e*slice
        sl = gate_slice if key in ('gate', 'up') else down_slice
        experts = []
        for e in range(288):
            r = file_to_phys(fname, off + e * sl, sl)
            if len(r) != 1:
                bad += 1
            total_slices += 1
            experts.append(r if len(r) != 1 else [[r[0][0], sl]])
        entry[key] = {'file': fname, 'tensor_offset': off, 'slice_bytes': sl, 'experts': experts}
    index['layers'][il] = entry

print(f'capas MoE indexadas: {len(index["layers"])}')
print(f'slices: {total_slices}, no-contiguos: {bad} ({100*bad/total_slices:.2f}%)')

# sanity: leer 512B del experto 0 capa 3 via LBA crudo y comparar con el archivo
import random
random.seed(7)
for probe in range(3):
    il = random.choice(list(index['layers'].keys()))
    e = random.randrange(288)
    key = random.choice(['gate', 'up', 'down'])
    ent = index['layers'][il][key]
    lba, nb = ent['experts'][e][0]
    fname = ent['file']
    # leer del archivo los mismos bytes
    with open(os.path.join(base, fname), 'rb') as fh:
        fh.seek(ent['tensor_offset'] + e * ent['slice_bytes'])
        ref = fh.read(4096)
    print(f'probe: blk.{il}.{key} e={e} lba={lba} (verificar en Linux contra {ref[:16].hex()}...)')

out_path = r'E:\Q8_0\expert_lba_index.json'
with open(out_path, 'w') as f:
    json.dump(index, f, separators=(',', ':'))
print(f'indice: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)')
