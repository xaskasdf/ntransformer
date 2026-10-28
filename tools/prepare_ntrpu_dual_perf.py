"""Generate a finite R4 throughput bank, retaining the qualified protocol monitors."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


READBACK = r'''
    for(genvar ch=0;ch<2;ch++)begin: read_port
        task automatic readback(input integer first);
            for(integer q=first;q<counts[ch];q++)begin
                integer tag;tag=queues[ch][q];
                @(negedge ui_clk[ch]);araddr[ch*37+:37]=addresses[tag][36:0];
                arlen[ch*8+:8]=lengths[tag]-1;arvalid[ch]=1;
                do @(posedge ui_clk[ch]);while(!arready[ch]);
                @(negedge ui_clk[ch]);arvalid[ch]=0;rready[ch]=1;
                for(integer beat=0;beat<lengths[tag];beat++)begin
                    do @(posedge ui_clk[ch]);while(!m_rvalid[ch]);
                    need(rid[ch*4+:4]==0 && rresp[ch*2+:2]==0 &&
                         rlast[ch]==(beat==lengths[tag]-1) &&
                         rdata[ch*512+:512]===pattern(tag,beat),"PERF_READBACK_BYTES");
                    $display("PERF RD %0d %0d %0d %0d %0d %h %b",ps(),scenario,ch,tag,beat,rdata[ch*512+:512],rlast[ch]);
                end
                @(negedge ui_clk[ch]);rready[ch]=0;
            end
        endtask
    end
'''

STIMULUS = r'''
    integer scenario=-1;
    reg measuring=0;
    always @(posedge source_clk)if(measuring)begin
        need(admit && running && !fault && !stop,"PERF_HEALTHY");
        $display("PERF S %0d %0d %02h %03h %b %b %b %b %b %b",ps(),scenario,
            banks_used,pending_counts,cmd_valid,cmd_ready,data_valid,data_ready,cpl_valid,cpl_ready);
    end
    initial begin
        for(integer k=0;k<256;k++)begin aw_time[k]=0;w_time[k]=0;b_time[k]=0;finished[k]=0;end
        #100;$display("DUAL CASE cold_barrier");send(0,0,1);send(1,0,1);drain;
        fork read_port[0].readback(0);read_port[1].readback(0);join
        for(integer s=0;s<6;s++)begin
            integer q0,q1,beats,ch;
            scenario=s;q0=counts[0];q1=counts[1];beats=(s==0)?1:(s==1)?8:64;
            @(negedge ui_clk[0]);hold_w[0]=(s==4);hold_b[0]=(s==5);
            @(negedge source_clk);measuring=1;
            $display("PERF BEGIN %0d %0d first=%0d jobs=16 beats=%0d w_hold=%b b_hold=%b",ps(),s,accepted+1,beats,hold_w,hold_b);
            fork
                begin
                    for(integer k=0;k<16;k++)begin
                        ch=(s==3)?((k%4)==3):k%2;
                        send(ch,64'h100000000+s*64'h100000+k*4096,beats);
                    end
                end
                begin
                    if(s>=4)begin
                        wait_source(800);@(negedge ui_clk[0]);hold_w[0]=0;hold_b[0]=0;
                        $display("PERF RELEASE %0d %0d port=0",ps(),s);
                    end
                end
            join
            drain;@(negedge source_clk);measuring=0;
            $display("PERF END %0d %0d accepted=%0d completed=%0d",ps(),s,accepted,completed);
            fork read_port[0].readback(q0);read_port[1].readback(q1);join
            $display("PERF VERIFIED %0d %0d",ps(),s);
        end
        need(accepted==98 && completed==98 && !fault && !restart_required && drained,"PERF_FINAL");
        fork mem[0].model.verify_periodic_complete();mem[1].model.verify_periodic_complete();join
        $display("DUAL PERF PASS commands=%0d completions=%0d writes0=%0d writes1=%0d reads0=%0d reads1=%0d scenarios=6 ps=%0d",
                 accepted,completed,writes[0],writes[1],reads[0],reads[1],ps());
        $finish;
    end
    initial begin #300000;$fatal(1,"PERF_TIMEOUT commands=%0d completions=%0d",accepted,completed);end
endmodule
'''


def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    source = root/'fpga/tb/tb_ntrpu_dual_multibuffer_mig.sv'
    original = source.read_text()
    text = original.split('    initial begin\n        for(integer k=0;k<64;k++)')[0]
    assert text != original
    text = text.replace('[0:63]', '[0:255]')
    text = replace(text, 'ntrpu_dual_multibuffer_mig dut(.*);', 'ntrpu_dual_routed_core dut(.*);')
    text = replace(text, 'ntrpu_dual_mig_fixture #', 'ntrpu_dual_mig_perf_fixture #')
    text = replace(text, 'reg [73:0] araddr=0;', 'reg [73:0] araddr=0;\n    reg [15:0] arlen=0;')
    text = replace(text, '.m_axi_araddr(araddr[g*37+:37]),', '.m_axi_arlen(arlen[g*8+:8]),.m_axi_araddr(araddr[g*37+:37]),')
    text = replace(text, 'admit && !hold_cpl[ch] && cycle%7!=ch', 'admit && !hold_cpl[ch]')
    text = replace(text, '$display("DUAL AR %0d %0d %h",ps(),g,araddr[g*37+:37]);',
                   '$display("PERF AR %0d %0d %0d %h %0d",ps(),scenario,g,araddr[g*37+:37],arlen[g*8+:8]+1);')
    a = text.index('                    address=addresses[tag]+w_beat[g]*64;')
    z = text.index('                    $display("DUAL W ', a)
    text = text[:a]+'                    writes[g]++;\n'+text[z:]
    a = text.index('    // Static per-port tasks')
    z = text.index('    task automatic release_b;', a)
    text = text[:a]+READBACK+text[z:]
    text = replace(text, '@(negedge source_clk);cmd_channel=ch;',
                   '@(negedge source_clk);$display("PERF OFFER %0d %0d %0d %0d",ps(),scenario,tag,ch);cmd_channel=ch;')
    text += STIMULUS
    target = args.output/source.name
    target.write_text(text)
    fixture = root/'fpga/tb/ntrpu_dual_mig_fixture.sv'
    f = fixture.read_text()
    f = replace(f, 'module ntrpu_dual_mig_fixture #', 'module ntrpu_dual_mig_perf_fixture #')
    f = replace(f, 'input wire [36:0] m_axi_araddr,', 'input wire [7:0] m_axi_arlen,\n    input wire [36:0] m_axi_araddr,')
    f = replace(f, '    wire [7:0] m_axi_arlen=0;\n', '')
    (args.output/'ntrpu_dual_mig_perf_fixture.sv').write_text(f)
    unit = root/'fpga/tb/tb_ntrpu_dual_mig_cold_start.sv'
    (args.output/unit.name).write_bytes(unit.read_bytes())
    manifest = {}
    for src, dst in [(source,target),(fixture,args.output/'ntrpu_dual_mig_perf_fixture.sv'),(unit,args.output/unit.name)]:
        manifest[src.name] = dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                                 generated_sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
        (args.output/(src.name+'.diff')).write_text(''.join(difflib.unified_diff(
            src.read_text().splitlines(True), dst.read_text().splitlines(True), fromfile=src.name, tofile=dst.name)))
    (args.output/'transformation.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
