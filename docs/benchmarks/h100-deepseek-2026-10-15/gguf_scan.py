#!/usr/bin/env python3
"""Minimal GGUF header/KV/tensor-info parser (no tensor data access)."""
import struct, sys, re
from collections import OrderedDict

GGUF_MAGIC = 0x46554747

def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def read_value(f, t):
    if t == 0: return struct.unpack("<B", f.read(1))[0]
    if t == 1: return struct.unpack("<b", f.read(1))[0]
    if t == 2: return struct.unpack("<H", f.read(2))[0]
    if t == 3: return struct.unpack("<h", f.read(2))[0]
    if t == 4: return struct.unpack("<I", f.read(4))[0]
    if t == 5: return struct.unpack("<i", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return struct.unpack("<B", f.read(1))[0] != 0
    if t == 8: return read_string(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return ("ARRAY", et, n, [read_value(f, et) for _ in range(n)])
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"bad type {t}")

TYPE_NAMES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",
 11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",
 19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",
 28:"F64",29:"IQ1_M",30:"BF16",39:"MXFP4"}
# (block_elems, block_bytes)
TYPE_BLOCKS = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),10:(256,84),
 11:(256,110),12:(256,144),13:(256,176),14:(256,210),16:(256,66),17:(256,74),18:(256,98),
 19:(256,50),20:(32,18),21:(256,110),22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),
 27:(1,8),28:(1,8),29:(256,56),30:(1,2),39:(32,17)}

def row_size(t, n):
    be, bb = TYPE_BLOCKS[t]
    return (n // be) * bb

def main(path):
    f = open(path, "rb")
    magic = struct.unpack("<I", f.read(4))[0]
    assert magic == GGUF_MAGIC, hex(magic)
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"version={version} n_tensors={n_tensors} n_kv={n_kv}")

    kv = OrderedDict()
    for _ in range(n_kv):
        k = read_string(f)
        t = struct.unpack("<I", f.read(4))[0]
        v = read_value(f, t)
        kv[k] = v

    print("\n=== KV ===")
    trunc = {"tokenizer.ggml.tokens","tokenizer.ggml.scores","tokenizer.ggml.token_type",
             "tokenizer.ggml.merges","deepseek41.engram.token_map","deepseek41.engram.multipliers",
             "deepseek41.engram.primes","deepseek41.engram.offsets"}
    for k, v in kv.items():
        if k in trunc and isinstance(v, tuple) and v[0]=="ARRAY":
            _, et, n, arr = v
            head = arr[:10]
            print(f"{k}: ARRAY<{TYPE_NAMES.get(et,et)}> n={n} head={head}")
        else:
            s = str(v)
            if len(s) > 500: s = s[:500] + "..."
            print(f"{k}: {s}")

    infos = []
    for _ in range(n_tensors):
        name = read_string(f)
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
        tt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, dims, tt, off))

    print("\n=== TENSORS (grouped) ===")
    groups = OrderedDict()
    for name, dims, tt, off in infos:
        key = re.sub(r"\.\d+\.", ".N.", name)
        g = groups.setdefault(key, {"count": 0, "types": set(), "dims": dims, "bytes": 0})
        g["count"] += 1
        g["types"].add(tt)
        g["dims"] = dims
        nelem = 1
        for d in dims: nelem *= d
        g["bytes"] += row_size(tt, nelem)

    total = 0
    for k, g in groups.items():
        total += g["bytes"]
        tn = [TYPE_NAMES.get(t, str(t)) for t in sorted(g["types"])]
        print(f"{k:48s} x{g['count']:<4d} {tuple(reversed(g['dims']))} {tn} {g['bytes']/2**20:.1f}MiB")
    print(f"\nTOTAL tensor bytes: {total/2**30:.2f} GiB, {len(infos)} tensors")

if __name__ == "__main__":
    main(sys.argv[1])
