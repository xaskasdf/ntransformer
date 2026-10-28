"""Preflight the mapped public-only bench against RTL before netlist replay."""
from pathlib import Path
from prepare_ntrpu_peer_commit import once
from prepare_ntrpu_peer_regions_routed import public_bench


def prepare(root):
    bench=public_bench(root).replace('ntrpu_peer_regions_physical dut(', 'ntrpu_peer_regions dut(')
    bench=bench.replace('tb_ntrpu_peer_shared','tb_ntrpu_peer_regions_public')
    (root/'fpga/tb/tb_ntrpu_peer_regions_public.sv').write_text(bench,encoding='utf-8',newline='\n')
    text=(root/'tools/ntrpu_peer_shared.tcl').read_text(encoding='utf-8')
    text=once(text,'ntrpu_peer_read_pkg ntrpu_peer_read_lane ntrpu_peer_read_admission ntrpu_peer_read_shared',
              'ntrpu_peer_read_pkg ntrpu_peer_region_pkg ntrpu_peer_region_lookup ntrpu_peer_region_admission '
              'ntrpu_peer_region_lane ntrpu_peer_read_domain ntrpu_peer_regions')
    text=text.replace('tb_ntrpu_peer_shared','tb_ntrpu_peer_regions_public')
    text=text.replace('SG ALL_PASS mode=full cases=6','SP ALL_PASS cases=6')
    text=text.replace('PEER_SHARED_EXPORTED','REGIONS_PUBLIC_EXPORTED')
    (root/'tools/ntrpu_peer_regions_public.tcl').write_text(text,encoding='utf-8',newline='\n')


if __name__=='__main__':
    text=Path(__file__).with_name('run_ntrpu_peer_shared.py').read_text(encoding='utf-8')
    text=once(text,"out=a.output.resolve();out.mkdir", "assert a.full;out=a.output.resolve();out.mkdir")
    text=once(text,"'tools/ntrpu_peer_shared.tcl'","'tools/ntrpu_peer_regions_public.tcl'")
    text=once(text,"experiment='peer_shared'","experiment='peer_regions_public'")
    text=once(text,"marker='SG ALL_PASS mode='+('full cases=6' if a.full else 'unit cases=15')",
              "marker='SP ALL_PASS cases=6'")
    exec(compile(text,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
