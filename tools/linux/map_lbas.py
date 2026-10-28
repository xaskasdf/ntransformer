import ctypes, struct, json, os
from ctypes import wintypes

k32 = ctypes.windll.kernel32
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
FSCTL_GET_RETRIEVAL_POINTERS = 0x00090073   # fn28 METHOD_NEITHER

def get_runs(path):
    h = k32.CreateFileW(path, GENERIC_READ, 3, None, OPEN_EXISTING, 0, None)
    if h in (-1, 0xFFFFFFFF):
        raise RuntimeError(f'CreateFileW {path}: {ctypes.GetLastError()}')
    try:
        runs = []
        vcn = 0
        max_ext = 4096
        out = ctypes.create_string_buffer(16 + max_ext * 16)
        ret = wintypes.DWORD()
        while True:
            inp = struct.pack('<q', vcn)
            ok = k32.DeviceIoControl(h, FSCTL_GET_RETRIEVAL_POINTERS,
                                     inp, len(inp), out, len(out), ctypes.byref(ret), None)
            if not ok:
                err = ctypes.GetLastError()
                if err == 38:  # ERROR_HANDLE_EOF
                    break
                raise RuntimeError(f'DeviceIoControl: {err}')
            n, _res, _start = struct.unpack_from('<IIq', out.raw, 0)
            if n == 0:
                break
            for i in range(n):
                next_vcn, lcn = struct.unpack_from('<qq', out.raw, 16 + i * 16)
                if lcn >= 0:
                    runs.append([vcn, lcn, next_vcn - vcn])
                vcn = next_vcn
            if n < max_ext:
                break
        return runs
    finally:
        k32.CloseHandle(h)

# cluster size de E:
out = ctypes.create_string_buffer(128)
ret = wintypes.DWORD()
h = k32.CreateFileW('E:\\', GENERIC_READ, 3, None, OPEN_EXISTING, 0, None)
ok = k32.DeviceIoControl(h, 0x00090064, None, 0, out, 128, ctypes.byref(ret), None)
k32.CloseHandle(h)
# NTFS_VOLUME_DATA: BytesPerSector@0x00, BytesPerCluster@0x08, ...
bps, bpc = struct.unpack_from('<qq', out.raw, 0)
print(f'volume: sector={bps}B cluster={bpc}B')

base = r'E:\Q8_0'
files = sorted(f for f in os.listdir(base) if f.endswith('.gguf'))
result = {
    'bytes_per_sector': bps,
    'bytes_per_cluster': bpc,
    'files': {}
}
for f in files:
    p = os.path.join(base, f)
    runs = get_runs(p)
    size = os.path.getsize(p)
    total_clusters = sum(r[2] for r in runs)
    # contiguidad: suma de gaps de logitud 1 cluster (fragmentacion interna)
    print(f'{f}: {size/1e9:.1f}GB runs={len(runs)} clusters={total_clusters} '
          f'avg_run={size/max(1,len(runs))/1e6:.0f}MB')
    result['files'][f] = {'size': size, 'runs': runs}

with open(r'C:\Users\xasko\AppData\Local\Temp\opencode\gguf_lba_map.json', 'w') as fh:
    json.dump(result, fh)
print('mapa: C:\\Users\\xasko\\AppData\\Local\\Temp\\opencode\\gguf_lba_map.json')

# stats: cuanto saltaria un slice de 8.9MB entre extents en promedio
import random
random.seed(1)
f2 = result['files'][files[1]]
runs = f2['runs']
# construir mapa lineal vcn->lcn por bloques
bounds = []
v = 0
for vcn, lcn, n in runs:
    bounds.append((vcn * bpc, lcn * bpc, n * bpc))
sz = f2['size']
cross = 0
tests = 2000
for _ in range(tests):
    off = random.randrange(0, sz - 9_000_000)
    end = off + 8_912_896
    nruns = sum(1 for s, l, n in bounds if s < end and (s + n) > off)
    cross += max(0, nruns - 1)
print(f'slices 8.9MB: {cross/tests:.2f} cruces de extent promedio')
