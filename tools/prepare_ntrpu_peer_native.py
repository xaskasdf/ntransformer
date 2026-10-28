"""Build a native-HBM bench around an immutable, reviewed peer RTL candidate."""
import argparse
import json
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save
from prepare_ntrpu_read_return import prepare as binding_include


def once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def prepare(selected, review, output, ip_origin=None):
    selected = selected.resolve()
    physical = json.loads(review.read_text())
    assert physical['status'] == 'PEER_READ_ROUTED_INTERNAL_TIMING_PASS'
    assert physical['result_sha256'] == sha(selected/'output/result.json')
    result = json.loads((selected/'output/result.json').read_text())
    assert result['sources'] == result['sources_after']
    ip_origin = ip_origin or selected
    if ip_origin.resolve()!=selected:
        assert sha(ip_origin/'output/result.json')==sha(selected/'output/DONOR_RESULT.json')
    for name, digest in result['sources'].items():
        assert sha(selected/'repo'/name) == digest, name
    shutil.copytree(selected/'repo', output)
    root = Path(__file__).resolve().parents[1]
    for name in ('prepare_ntrpu_peer_native.py', 'run_ntrpu_peer_native.py',
                 'check_ntrpu_peer_native.py', 'ntrpu_peer_native.tcl',
                 'ntrpu_peer_native_run.tcl'):
        shutil.copyfile(root/'tools'/name, output/'tools'/name)
    text = (output/'fpga/tb/tb_ntrpu_peer_read.sv').read_text()
    text = once(text, 'module tb_ntrpu_peer_read;', 'module tb_ntrpu_peer_native;')
    text = once(text, 'FULL=0,PAIR=0;', 'FULL=1,PAIR=1,NATIVE=0;')
    begin = text.index('        assign arready[l]=')
    end = text.index('    if(PAIR)begin: pair', begin)
    text = text[:begin]+'''        always @(posedge mem_clk)begin
            if(!mem_rst_n)begin nr[l]<=0;nw[l]<=0;rb[l]<=0;end
            else begin
                if(arvalid[l] && arready[l])begin
                    nw[l]<=nw[l]+1;
                    $display("PN AR %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,l,araddr[l],int'(arlen[l])+1);
                end
                if(rvalid[l] && rready[l])begin
                    nr[l]<=nr[l]+1;
                    $display("PN NR %0d %0d %0d %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,raw_rid[l*6+:6],rresp[l],rlast[l],rdata[l]);
                end
            end
        end
    end
'''+text[end:]
    text = once(text, '        wire [7:0] h_arlen;\n        wire [67:0] h_araddr;\n        ntrpu_peer_read_pair', '        ntrpu_peer_read_pair')
    text = text.replace('.init_complete_async(2\'b11)', '.init_complete_async(init_complete_async)')
    text = once(text, '.h_rvalid(rvalid),.h_rready(rready),.h_rdata({rdata[1],rdata[0]}),.h_rid(12\'b0),.h_rresp',
                '.h_arid(h_arid),.h_arsize(h_arsize),.h_arburst(h_arburst),\n            .h_rvalid(rvalid),.h_rready(rready),.h_rdata({rdata[1],rdata[0]}),.h_rid(raw_rid),.h_rresp')
    original = (output/'fpga/tb/tb_ntrpu_read_return.sv').read_text()
    writer = original[original.index('    reg [1:0] wcmd_valid'):original.index('    reg [1:0] ar_open')]
    extra = '''    wire [1:0] init_complete_async={apb_complete_1,apb_complete_0};
    wire [1:0] h_arvalid=arvalid,h_rready=rready;
    wire [67:0] h_araddr;
    wire [7:0] h_arlen;
    wire [11:0] h_arid,raw_rid;
    wire [5:0] h_arsize;
    wire [3:0] h_arburst,raw_rresp;
    wire [1:0] raw_arready,raw_rvalid,raw_rlast;
    wire [511:0] raw_rdata;
    wire [1:0] stray=0;
    assign arready=raw_arready & ar_open;
    assign rvalid=raw_rvalid & r_open;
    assign {rdata[1],rdata[0]}=raw_rdata;
    assign {rresp[1],rresp[0]}=raw_rresp;
    assign rlast=raw_rlast;
    `include "ntrpu_read_return_binding.svh"
'''+writer
    text = once(text, '    integer epoch=0,', extra+'    integer epoch=0,')
    text = once(text, 'epoch=epoch+1;tick(10);cold_reset_n=1;while(!started)tick();tick(5);',
                'epoch=epoch+1;apb_rst_n=0;tick(NATIVE?940:10);apb_rst_n=1;cold_reset_n=1;while(!started || !wstarted)tick();tick(5);')
    text = once(text, '        configure();\n        $display("PR BOOT', '        $display("PR BOOT')
    start = text.index('    request_t v;\n    event_t e;\n    initial begin')
    body = text[text.index('        scenario=2;boot();command_open=0;', start):text.index('        scenario=3;boot();', start)]
    body = once(body, '        scenario=2;boot();command_open=0;', '        scenario=2;command_open=0;')
    # Keep the qualified nominal ledger scenario, including exact oracle mapping.
    stop = text[text.index('        scenario=21;boot();ar_open=0;', start):text.index('        $display("PR ALL_PASS', start)]
    stop = once(stop, 'scenario=21;boot();ar_open=0;', 'scenario=21;ar_open=0;result_ready=0;dma_ready=0;')
    stop = once(stop, 'submit(l,request_for(l,1))', 'submit(l,request_for(l,11))')
    text = text[:start]+(root/'fpga/tb/ntrpu_peer_native_body.svh').read_text()+body+stop+'''
        $display("PN ALL_PASS cases=2 native=%0d",NATIVE);$finish;
    end
    initial begin #150000;$fatal(1,"PN_WATCHDOG case=%0d",scenario);end
endmodule
'''
    (output/'fpga/tb/tb_ntrpu_peer_native.sv').write_text(text)
    shutil.copyfile(root/'fpga/tb/ntrpu_peer_native_body.svh', output/'fpga/tb/ntrpu_peer_native_body.svh')
    binding_include(output/'fixtures/native.sv', output/'fixtures/ntrpu_read_return_binding.svh')
    rtl = {n: h for n,h in result['sources'].items()
           if n.endswith('.sv') and n.startswith(('fpga/rtl/', 'fpga/experimental/'))}
    assert rtl and all(sha(output/n)==h for n,h in rtl.items())
    save(output/'NATIVE_PROFILE.json', dict(selected_result_sha256=sha(selected/'output/result.json'),
         selected_checkpoint_sha256=sha(selected/'output/suite/routed.dcp'), selected_rtl=rtl,
         selected_read_configs={n:dict(s.split('\t',1) for s in (ip_origin/'output/suite'/n).read_text().splitlines()
                                      if s.startswith('CONFIG.')) for n in
             ('hbm_read_cdc_config.tsv','hbm_read_width_config.tsv','hbm_read_protocol_config.tsv')},
         scope='Native HBM writes/reads, peer nominal reuse and global stop; DMA completion and ACK are test stimuli'))
    shutil.copyfile(review, output/'SELECTED_PHYSICAL_REVIEW.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('selected',type=Path);ap.add_argument('review',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--ip-origin',type=Path)
    a=ap.parse_args();prepare(a.selected,a.review,a.output,a.ip_origin)
