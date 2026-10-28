"""Compose frozen return-cut transport and two unchanged HBM write adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PARENT = dict(awvalid=1,awready=1,awaddr=64,awlen=8,awid=4,awcache=4,
              awsize=3,awburst=2,wvalid=1,wready=1,wlast=1,wdata=512,wstrb=64,
              bready=1,bvalid=1,bid=4,bresp=2)
CHILD = dict(awvalid=1,awready=1,awaddr=34,awlen=4,awid=6,awsize=3,awburst=2,
             wvalid=1,wready=1,wlast=1,wdata=256,wstrb=32,bready=1,bvalid=1,bid=6,bresp=2)
GENERATED = ['ntrpu_candidate_transport','ntrpu_destination_candidate',
    'ntrpu_peer_multibuffer_offer','ntrpu_peer_multibuffer_reserved',
    'ntrpu_candidate_dispatch','ntrpu_buffer_pool_allocator_return_cut']

def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def bench(text):
    text=one(text,'module tb_ntrpu_candidate_transport;','module tb_ntrpu_hbm_transport;')
    text=text.replace('dut.','dut.transport.')
    text=one(text,'ntrpu_candidate_transport dut(.*);','ntrpu_hbm_transport dut(.*);')
    text=one(text,'reg [1:0] cpl_ready=0,m_awready=0,m_wready=0,m_bvalid=0;',
             'reg [1:0] cpl_ready=0; wire [1:0] m_awready,m_wready,m_bvalid;')
    text=one(text,'reg [7:0] m_bid=0;','wire [7:0] m_bid;')
    text=one(text,'reg [3:0] m_bresp=0;','wire [3:0] m_bresp;')
    declarations=[]
    for name,width in PARENT.items():
        declarations.append(f'    assign m_{name}=dut.t_{name};')
    incoming={'awready','wready','bvalid','bid','bresp'}
    for name,width in CHILD.items():
        declarations.append(f'    {"reg" if name in incoming else "wire"} [{2*width-1}:0] h_{name}'+('=0;' if name in incoming else ';'))
    declarations.append('    wire [1:0] adapter_idle,adapter_fault;')
    text=one(text,'    integer cycle=0,','\n'.join(declarations)+'\n    integer cycle=0,')
    # Parent response and acceptance now come from real IP, never the testbench.
    text=one(text,'            m_awready[ch]=rst_n&&!hold_aw[ch]&&cycle%5!=0;','')
    text=one(text,'            m_wready[ch]=rst_n&&!hold_w[ch]&&cycle%7!=0;','')
    begin=text.index('            if(!m_bvalid[ch]||b_sent[ch]||!rst_n)begin')
    end=text.index('\n            end',begin)+len('\n            end')
    text=text[:begin]+text[end:]
    text=text.replace('    task ','    task automatic ')
    text=one(text,'step;submit_valid[ch]=0;disable waiting;end step;end',
             'step;submit_valid[ch]=0;return;end step;end')
    text=text.replace("64'h100000000+64'(tag)*4096",'destination(tag)')
    text=text.replace("64'h100000000+64'(t)*4096",'destination(t)')
    text=one(text,'    function [231:0] descriptor',
        "    function [63:0] destination(input integer tag);\n        destination=scenario==15?64'h400000000:64'h100000000+64'(tag)*4096;\n    endfunction\n    function [231:0] descriptor")
    text=one(text,'repeat(4)step;scenario=number;rst_n=1;',
             'repeat(32)step;scenario=number;rst_n=1;')
    assert '    initial begin\n        #150;\n        reset_case(1);' in text
    # The adapter absorbs writes before its target stalls: do not reuse the
    # standalone transport\'s exact bank occupancy or parent-AW expectations.
    text=one(text,'need(fetched[0]==6&&queued[3:0]==2,"BANK_LIMIT");',
             'need(fetched[0]<=8&&cpls[0]==0&&!drained,"BLOCKED_HBM_RETAINS_WORK");')
    text=one(text,'need(ws[0]==2&&aws[0]==0&&!drained,"W_BEFORE_AW");',
             'need(hw[0]>0&&ha[0]==0&&!drained,"W_BEFORE_HBM_AW");')
    text=one(text,'        $display("TE SUITE_PASS cases=13 cycle=%0d",cycle);$finish;', '''        reset_case(14);hold_b=1;
        for(integer j=0;j<8;j++)push_one(0,970+j,64);
        repeat(1200)step;
        need(aws[0]==4&&bs[0]==0&&ha[0]>=24&&!drained,"FOUR_PARENTS_BLOCK_FIFTH");
        $display("HT SATURATED %0d %0d %0d",cycle,aws[0],ha[0]);
        stop=1;step;stop=0;hold_b=0;drain_all;
        need(submit_ready==0,"STOP_STAYS_CLOSED");pass_case;

        reset_case(15);push_one(0,960,9);wait_cpl(0,1);repeat(10)step;
        need(fault&&status[960]==3&&ha[0]==0&&hw[0]==0&&drained,"LOCAL_REJECT_NO_HBM_WRITE");pass_case;

        for(integer bad=16;bad<=17;bad++)begin
            reset_case(bad);push_one(0,990,1);repeat(150)step;
            need(fault&&adapter_fault[0]&&!drained&&submit_ready==0&&ha[0]==1&&hw[0]==1&&hb[0]==0&&bs[0]==0&&cpls[0]==0,"FOREIGN_B_QUARANTINE");
            need(h_bvalid[0]&&!h_bready[0],"BAD_B_RETAINED");
            $display("HT QUARANTINE %0d %0d %0d %0d",cycle,scenario,h_bid[5:0],h_bresp[1:0]);pass_case;
        end
        $display("HT ALL_PASS cases=17 cycle=%0d",cycle);$finish;''')
    target=(ROOT/'fpga/tb/ntrpu_hbm_transport_target.svh').read_text()
    return one(text,'endmodule',target+'\nendmodule')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--reference',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False);ref=a.reference
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files=json.loads((ref/'FILES.json').read_text())
    cut=json.loads((ref/'suite/generated/RETURN_CUT.json').read_text())
    record=json.loads((ref/'result.json').read_text())
    assert record['experiment']=='registered_return_axi' and record['stage']=='rtl'
    assert record['status']=='EXPORTED_PENDING_REVIEW' and record['sources']==record['sources_after']
    assert sha(ref/'suite/generated/RETURN_CUT.json')==files['suite/generated/RETURN_CUT.json']
    assert all(sha(ROOT/n)==h for n,h in record['sources'].items())
    for name,h in cut['logic_sha256'].items():
        source=ref/'suite/generated'/(name+'.sv') if name in GENERATED else ROOT/'fpga/rtl'/(name+'.sv')
        assert sha(source)==h,name
        (out/(name+'.sv')).write_bytes(source.read_bytes())
    source=ref/'suite/generated/tb_ntrpu_candidate_transport.sv'
    assert sha(source)==files['suite/generated/tb_ntrpu_candidate_transport.sv']
    (out/'tb_ntrpu_hbm_transport.sv').write_text(bench(source.read_text()))
    (out/'COMPOSITION.json').write_text(json.dumps(dict(reference_result_sha256=sha(ref/'result.json'),
        frozen_logic=cut['logic_sha256'],sources={p.name:sha(p) for p in out.glob('*.sv')},
        adapters=2,admit='rst_n; stop only closes transport ingress',clock_domains=1,
        scope='HBM-compatible behavioral target; no hard HBM controller'),indent=2)+'\n')

if __name__=='__main__':main()
