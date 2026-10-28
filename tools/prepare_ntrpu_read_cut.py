"""Prepare an isolated F1 variant with a registered width-to-CDC return boundary."""
import argparse
import json
from pathlib import Path
from prepare_ntrpu_read_frames import prepare as frames
from prepare_ntrpu_read_window import sha, replace_once


def prepare(root, out, depth, binding, stub, legacy=False, speculative=False):
    assert depth == 2
    frames(root, out, depth, binding, stub, legacy)
    inherited = json.loads((out/'VARIANT.json').read_text())
    path = out/'fpga/experimental/ntrpu_hbm_read_adapter.sv'
    text = path.read_text()
    text = replace_once(text, 'ntrpu_hbm_read_tracker #(.MAX_PENDING(8)) tracker(.*);',
                        'ntrpu_hbm_read_tracker #(.MAX_PENDING(8)) tracker(.pending(tracker_pending),.*);')
    marker = '    wire [255:0] validated_data;'
    boundary = '''    wire tracker_pending,return_empty;
    wire raw_rvalid,raw_rready,raw_rlast;
    wire [511:0] raw_rdata;
    wire [3:0] raw_rid;
    wire [1:0] raw_rresp;
    assign pending=tracker_pending||!return_empty;
    // The frame still owns the payload. Stop/fault cannot discard this queue.
    // Two registered slots isolate the width converter from the CDC write enable.
    ntrpu_hbm_read_response_queue #(.WIDTH(519)) return_cut (
        .clk(clk),.rst_n(rst_n),.s_valid(raw_rvalid),.s_ready(raw_rready),
        .s_data({raw_rid,raw_rresp,raw_rlast,raw_rdata}),
        .m_valid(s_rvalid),.m_ready(s_rready),
        .m_data({s_rid,s_rresp,s_rlast,s_rdata}),.empty(return_empty));
'''
    text = replace_once(text, marker, boundary+marker)
    for signal in ['rdata','rid','rresp','rlast','rvalid','rready']:
        text = replace_once(text, '.s_axi_'+signal+'(s_'+signal+')',
                            '.s_axi_'+signal+'(raw_'+signal+')')
    if speculative:
        text=replace_once(text,'ntrpu_hbm_read_response_queue #(.WIDTH(519)) return_cut',
                          'ntrpu_hbm_read_return_cut #(.WIDTH(519)) return_cut')
        text+='\n'+(root/'fpga/experimental/ntrpu_hbm_read_return_cut.sv').read_text()
    path.write_text(text)
    if not legacy:
        path=out/'fpga/tb/tb_ntrpu_read_return.sv'
        monitor='''
    for(genvar rc=0;rc<2;rc++)begin: return_cut_monitor
        always @(posedge mem_clk)if(mem_rst_n)begin
            $display("RC STATE %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",stamp(),epoch,rc,
                dut.channels[rc].adapter.return_cut.count,
                dut.channels[rc].adapter.return_cut.read_slot,dut.channels[rc].adapter.return_cut.write_slot,
                dut.channels[rc].adapter.raw_rvalid,dut.channels[rc].adapter.raw_rready,
                dut.channels[rc].adapter.s_rvalid,dut.channels[rc].adapter.s_rready,adapter_pending[rc]);
            if(dut.channels[rc].adapter.raw_rvalid&&dut.channels[rc].adapter.raw_rready)
                $display("RC IN %0d %0d %0d %h",stamp(),epoch,rc,
                    {dut.channels[rc].adapter.raw_rid,dut.channels[rc].adapter.raw_rresp,
                     dut.channels[rc].adapter.raw_rlast,dut.channels[rc].adapter.raw_rdata});
            if(dut.channels[rc].adapter.s_rvalid)
                $display("RC WORD %0d %0d %0d %h",stamp(),epoch,rc,
                    {dut.channels[rc].adapter.s_rid,dut.channels[rc].adapter.s_rresp,
                     dut.channels[rc].adapter.s_rlast,dut.channels[rc].adapter.s_rdata});
        end
    end
'''
        path.write_text(replace_once(path.read_text(),'    initial begin #200000;',monitor+'    initial begin #200000;'))
    originals={}
    extra=['tools/prepare_ntrpu_read_cut.py','tools/check_ntrpu_read_cut.py']
    if speculative:extra.append('fpga/experimental/ntrpu_hbm_read_return_cut.sv')
    for name in extra:
        src=root/name;dst=out/name;dst.write_bytes(src.read_bytes());originals[name]=sha(src)
    result=dict(variant='read_cut',revision=2 if speculative else 1,frames=2,credits=8,cut_slots=2,cut_width=519,
                speculative_free_slot=speculative,
                legacy=legacy,inherited_variant=inherited,originals=originals,
                prepared={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
                          if p.is_file() and p.name!='VARIANT.json'})
    (out/'VARIANT.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--stub',type=Path,required=True)
    ap.add_argument('--legacy',action='store_true');ap.add_argument('--speculative',action='store_true');a=ap.parse_args()
    result=prepare(Path(__file__).resolve().parents[1],a.output,2,a.binding,a.stub,a.legacy,a.speculative)
    print(json.dumps(dict(variant=result['variant'],files=len(result['prepared']))))
