"""Create a private range-check variant from a hash-pinned composition snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_SHA = '0454df4e72a9007b59274a0eb814b756e8b8bc90d581f4df39fee900d8d6a9f6'
OLD = "    wire [64:0] end_address={1'b0,src_req_address}+{52'b0,src_req_bytes};"
NEW = """    // The retained 4 KiB span check prevents crossing the last 16 GiB page.
    // Relational comparisons preserve X/Z propagation; positive length keeps
    // an unknown length from becoming a definite range fault in simulation.
    wire range_bad=src_req_address>=64'h400000000 && src_req_bytes>13'd0;"""


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def variant(text):
    if text.count(OLD) != 1 or text.count("end_address>65'h400000000") != 1:
        raise ValueError('Frozen predicate anchor changed')
    return text.replace(OLD, NEW).replace("end_address>65'h400000000", 'range_bad')


def prepare(donor, here, out):
    if out.exists(): raise ValueError('New private prepared directory required')
    if sha(donor/'ntrpu_axi_bridge_hbm_source.sv') != SOURCE_SHA:
        raise ValueError('Wrong source donor')
    out.mkdir(parents=True)
    inputs = {}
    for p in donor.iterdir():
        if p.is_file() and p.suffix in ('.sv', '.json'):
            shutil.copy2(p, out/p.name); inputs[p.name] = sha(p)
    for pattern in ('*hbm_source*.py', '*hbm_cdc*.py'):
        for p in here.glob(pattern):
            if p.name.startswith(('run_', 'review_')):
                shutil.copy2(p, out/p.name)
    for name in ('tb_axi_bridge_hbm_source.sv', 'tb_axi_bridge_hbm_cdc.sv',
                 'run_hbm_source_trace.tcl', 'run_hbm_cdc_trace.tcl',
                 'ntrpu_axi_bridge_hbm_mem_watch.sv', 'synth_hbm_clocked.tcl',
                 'run_hbm_clocktree.tcl', 'constrain_hbm_clocktree.tcl',
                 'insert_hbm_clocktree.tcl', 'audit_hbm_clocktree.tcl',
                 'audit_hbm_clocktree_topology.tcl', 'tb_axi_bridge_hbm_range.sv',
                 'prepare_hbm_range.py', 'prove_hbm_range.py', 'run_hbm_range.py',
                 'refine_hbm_range.tcl'):
        shutil.copy2(here/name, out/name)
    original = (donor/'ntrpu_axi_bridge_hbm_source.sv').read_text()
    (out/'reference_hbm_source.sv').write_text(original.replace('module ntrpu_axi_bridge_hbm_source (', 'module reference_hbm_source ('))
    (out/'ntrpu_axi_bridge_hbm_source.sv').write_text(variant(original))
    record = dict(status='HBM_RANGE_PREPARED', donor_files=inputs,
                  variant_sha256=sha(out/'ntrpu_axi_bridge_hbm_source.sv'),
                  change='Replace only the end-address range predicate; all state/protocol logic unchanged',
                  two_state_equivalence_pending=True, four_state_checks_pending=True)
    (out/'RANGE_VARIANT.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('donor', type=Path); ap.add_argument('output', type=Path)
    a = ap.parse_args(); print(json.dumps(prepare(a.donor, Path(__file__).resolve().parent, a.output), indent=2))
