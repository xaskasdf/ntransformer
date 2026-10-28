"""Explicit oracle adaptations for retained guard channels; no packet changes."""
from pathlib import Path
from prepare_ntrpu_guarded_native import once


def core(text):
    return once(text,
        "require(stats['b_stall']>0 and stats['r_stall']>0 and stats['arm_stall']>0,'MMIO pauses exercised')",
        "# Retained channels can accept core B/R immediately; report measured stalls.\n    require(stats['arm_stall']>0,'MMIO ARM pause exercised')")


def transport(text):
    text=text.replace('check_ntrpu_pcie_managed_native_core','check_ntrpu_guarded_native_core')
    start=text.index("    require(first(reset,'lifetime')<first(reset,'transport')")
    end=text.index('    for row in hd:',start)
    text=text[:start]+'''    lifetime=first(reset,'lifetime')
    require(lifetime<min(first(reset,'transport'),first(hd,'reset')),'lifetime before guard transport release')
    require(sum(int(x['ps'])>=lifetime and x['reset']=='0' for x in hd)>=64,'host reset at least64 actual edges')
    require(sum(int(x['ps'])>=lifetime and x['transport']=='0' for x in reset)>=64,'source reset at least64 actual edges')
    # Guard reset releases occur on local rising edges, unlike the old BFM's
    # falling-edge assignments. Both sampled views are checked against VCD.
'''+text[end:]
    return once(text,
        "require(all(hc_stalls.get(k,0)>0 for k in ('AW','W','AR')) and all(sc_stalls.get(k,0)>0 for k in ('B','R')),'native request and returning acknowledgement pauses exercised')",
        "require(all(hc_stalls.get(k,0)>0 for k in ('AW','W','AR')),'native host request pauses exercised; core B/R stalls reported without a coverage claim')")


def native(text):
    text=text.replace('check_ntrpu_pcie_managed_native_transport','check_ntrpu_guarded_native_transport')
    text=text.replace('check_ntrpu_pcie_managed_native_core','check_ntrpu_guarded_native_core')
    text=once(text,"ROOT='/board/ntrpu_bypass/'", "from check_ntrpu_guarded_native_startup import EXTRA,review_startup\n\nROOT='/board/ntrpu_bypass/'")
    text=once(text,'    names={};widths={};scope=[];keep_header=[]','    extra.update(EXTRA)\n    names={};widths={};scope=[];keep_header=[]')
    text=once(text,"len(names)==70,'exact70 native pins'", "len(names)==84,'exact84 native pins'")
    text=once(text,'    frames,writes,reads=fixture();', '    frames,writes,reads=fixture();')
    text=once(text,"    return dict(status='PASS_NATIVE_NTRD_DUAL_MIG_COLD',check=checked,signals=70,", "    guard=review_startup(sim,vcd,host,source,finish)\n    return dict(status='PASS_GUARDED_NATIVE_NTRD_DUAL_MIG_COLD',check=checked,guard=guard,signals=84,")
    text=once(text,'External barriers remain trusted BFMs.', 'Automatic guard uses actual native link/reset inputs and cold peer observations. External data barriers remain trusted BFMs; runtime reset coordination is unqualified.')
    return text


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    for suffix,fn in [('_core',core),('_transport',transport),('',native)]:
        src=root/('check_ntrpu_pcie_managed_native'+suffix+'.py')
        dst=root/('check_ntrpu_guarded_native'+suffix+'.py')
        with dst.open('x',newline='\n') as f:f.write(fn(src.read_text()))
