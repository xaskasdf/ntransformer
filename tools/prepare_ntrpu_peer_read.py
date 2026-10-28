"""Prepare a private two-frame F1 + bounded peer ledger composition."""
import argparse
import ast
import json
from pathlib import Path
import shutil

from prepare_ntrpu_read_frames import prepare as frames
from prepare_ntrpu_read_window import sha
from run_ntrpu_peer_read import NEW


def prepare(root, output, binding, stub, return_cut=False):
    if return_cut:
        from prepare_ntrpu_read_cut import prepare as cut
        cut(root, output, 2, binding, stub, speculative=True)
    else:
        frames(root, output, 2, binding, stub)
    inherited = json.loads((output/'VARIANT.json').read_text())
    extra = NEW+[
        'tools/prepare_ntrpu_peer_read.py', 'tools/run_ntrpu_peer_read_physical.py',
        'tools/prepare_ntrpu_peer_read_routed.py', 'tools/run_ntrpu_peer_read_routed.py',
        'tools/check_ntrpu_peer_read_routed.py',
        'tools/run_ntrpu_peer_read_resume.py', 'tools/run_ntrpu_peer_read_refine.py',
        'tools/run_ntrpu_peer_read_drc.py', 'tools/check_ntrpu_peer_read_drc.py',
        'tools/check_ntrpu_peer_read_guard.py', 'tools/ntrpu_hbm_drc_audit.tcl',
        'tools/ntrpu_peer_read_physical.tcl', 'tools/check_ntrpu_peer_read_physical.py',
        'tools/check_ntrpu_peer_read_oracle.py', 'tools/ntrpu_transport_model.py',
        'tools/check_ntrpu_dual_hbm.py', 'tools/check_ntrpu_hbm_audit.py',
        'fpga/constraints/ntrpu_hbm_startup_cdc.xdc']
    if return_cut:
        extra += ['tools/prepare_ntrpu_read_cut.py', 'tools/check_ntrpu_read_cut.py',
                  'tools/check_ntrpu_peer_read_cut.py',
                  'fpga/experimental/ntrpu_hbm_read_return_cut.sv']
    for name in extra:
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    pending = [n for n in extra if n.endswith('.py')]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((output/name).read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [v.name for v in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                if not module or '.' in module:
                    continue
                dependency = 'tools/'+module+'.py'
                if not (root/dependency).exists():
                    continue
                if not (output/dependency).exists():
                    shutil.copyfile(root/dependency, output/dependency)
                    extra.append(dependency)
                pending.append(dependency)
    if return_cut:
        bench = output/'fpga/tb/tb_ntrpu_peer_read.sv'
        marker = '    // FULL=0\'s BFM must share the reader clock.'
        monitor = '''    if(PAIR && FULL)begin: return_cut_monitor
        for(genvar rc=0;rc<2;rc++)begin
            always @(posedge mem_clk)if(mem_rst_n)begin
                $display("RC STATE %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,rc,
                    pair.dut.reader.channels[rc].adapter.return_cut.count,
                    pair.dut.reader.channels[rc].adapter.return_cut.read_slot,pair.dut.reader.channels[rc].adapter.return_cut.write_slot,
                    pair.dut.reader.channels[rc].adapter.raw_rvalid,pair.dut.reader.channels[rc].adapter.raw_rready,
                    pair.dut.reader.channels[rc].adapter.s_rvalid,pair.dut.reader.channels[rc].adapter.s_rready,
                    pair.dut.reader.channels[rc].adapter.pending);
                if(pair.dut.reader.channels[rc].adapter.raw_rvalid && pair.dut.reader.channels[rc].adapter.raw_rready)
                    $display("RC IN %0d %0d %0d %h",$rtoi($realtime*1000),epoch,rc,
                        {pair.dut.reader.channels[rc].adapter.raw_rid,pair.dut.reader.channels[rc].adapter.raw_rresp,
                         pair.dut.reader.channels[rc].adapter.raw_rlast,pair.dut.reader.channels[rc].adapter.raw_rdata});
                if(pair.dut.reader.channels[rc].adapter.s_rvalid)
                    $display("RC WORD %0d %0d %0d %h",$rtoi($realtime*1000),epoch,rc,
                        {pair.dut.reader.channels[rc].adapter.s_rid,pair.dut.reader.channels[rc].adapter.s_rresp,
                         pair.dut.reader.channels[rc].adapter.s_rlast,pair.dut.reader.channels[rc].adapter.s_rdata});
            end
        end
    end
'''
        text = bench.read_text()
        assert text.count(marker)==1
        bench.write_text(text.replace(marker, monitor+marker))
    result = dict(variant='peer_read_single_span', frames_per_channel=2,
                  contexts_per_channel=2, native_fragment_credits=8,
                  source_nodes=[0, 1], shared_source_scheduler=False,
                  return_cut=return_cut, return_cut_revision=2 if return_cut else None,
                  inherited=inherited, originals={n: sha(root/n) for n in extra})
    result['prepared'] = {p.relative_to(output).as_posix(): sha(p)
                          for p in output.rglob('*') if p.is_file() and p.name!='VARIANT.json'}
    (output/'VARIANT.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--binding',type=Path,required=True)
    ap.add_argument('--stub',type=Path,required=True)
    ap.add_argument('--return-cut',action='store_true',help='Experimental two-slot speculative return queue')
    a=ap.parse_args()
    result=prepare(Path(__file__).resolve().parents[1],a.output,a.binding,a.stub,a.return_cut)
    print(json.dumps(dict(variant=result['variant'],files=len(result['prepared']))))
