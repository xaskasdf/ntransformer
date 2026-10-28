"""Prepare a registered cancellation boundary and a full-port protocol comparison."""
import hashlib
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
def put(path,text):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def build():
    original=(ROOT/'fpga/rtl/ntrpu_post_ffn_handoff_r2.sv').read_text()
    assert hashlib.sha256((ROOT/'fpga/rtl/ntrpu_post_ffn_handoff_r2.sv').read_bytes()).hexdigest()=='dc1f8525d7277095178040e5023cca7a70bf0f661b1a4c06148a4dc9c86d8ce5'
    old='    assign child_cancel_valid=child_owned && (cancel_valid || epoch_bad || source_bad || fault!=OK || revoked);'
    assert original.count(old)==1
    new='''    // Downstream cancellation crosses this ownership boundary only from retained state.
    // A sampled fault reaches the child after the next edge, and stays asserted until
    // retirement. Public result_valid still masks live revocation immediately.
    assign child_cancel_valid=child_owned && (fault!=OK || revoked || cancel_seen);'''
    candidate=original.replace('module ntrpu_post_ffn_handoff_r2','module ntrpu_post_ffn_handoff_r3').replace(old,new)
    put(ROOT/'fpga/rtl/ntrpu_post_ffn_handoff_r3.sv',candidate)
    bank=(ROOT/'fpga/tb/tb_ntrpu_post_ffn_handoff_r4.sv').read_text()
    bank=bank.replace('tb_ntrpu_post_ffn_handoff_r4','tb_ntrpu_post_ffn_cancel_r1').replace('ntrpu_post_ffn_handoff_r2 dut','ntrpu_post_ffn_handoff_r3 dut')
    # Direct assertions avoid the previously observed XSim conditional task/function issue.
    bank,count=re.subn(r'    task automatic require\(.*?    endtask\n','',bank,count=1,flags=re.S);assert count==1
    bank=bank.replace('require(', '`PF_ASSERT(')
    bank=bank.replace('`default_nettype none','`default_nettype none\n`define PF_ASSERT(C,M) begin if((C)!==1\'b1) $fatal(1,"PFHAND_FAIL cycle=%0d job=%0d mode=%0d: %s",cycle,jobs,mode_now,M); end',1)
    bank=bank.replace('`default_nettype wire','`undef PF_ASSERT\n`default_nettype wire')
    header=original[original.index(' (')+2:original.index('\n);')]
    ports={}
    for line in header.splitlines():
        if not line.strip():continue
        match=re.fullmatch(r'\s*(input|output)\s+(?:wire|logic)\s*(\[[^]]+\])?\s*(.*?)\s*,?\s*',line)
        assert match,line
        direction,width,names=match.groups()
        for name in names.rstrip(',').split(','):
            match=re.fullmatch(r'\s*(\w+)\s*(\[[^]]+\])?\s*',name);assert match,name
            n,array=match.groups();assert n not in ports;ports[n]=(direction,width or '',array or '')
    outputs={n:(w,a) for n,(d,w,a) in ports.items() if d=='output'}
    lines=['    // R2 and R3 see identical peers. Every public/control output except the','    // deliberately delayed cancel must match before and after each rising edge.']
    for n,(w,a) in outputs.items():lines.append(f'    wire {w} baseline_{n} {a};')
    lines+=['    ntrpu_post_ffn_handoff_r2 baseline (',',\n'.join(f"        .{n}({'baseline_'+n if d=='output' else n})" for n,(d,_,_) in ports.items()),'    );']
    lines+=['    integer cancellation_delays=0,comparison_edges=0;bit sampled_cancel=0;','    task compare_outputs;begin']
    for n,(w,a) in outputs.items():
        if n=='child_cancel_valid':continue
        if a:
            bounds=re.fullmatch(r'\[(\d+):(\d+)\]',a);start,end=map(int,bounds.groups())
            lines.append(f'        for(integer i={start};i<={end};i++)`PF_ASSERT({n}[i]===baseline_{n}[i],"R2/R3 mismatch: {n}")')
        else:lines.append(f'        `PF_ASSERT({n}===baseline_{n},"R2/R3 mismatch: {n}")')
    lines+=['        `PF_ASSERT(!child_cancel_valid || baseline_child_cancel_valid,"Spurious registered cancellation")',
        '    end endtask',
        '    always @(negedge clk)begin #0.2;if(rst_n)begin',
        '        compare_outputs();if(baseline_child_cancel_valid && !child_cancel_valid)cancellation_delays++;',
        '    end end',
        '    always @(posedge clk)begin sampled_cancel=baseline_child_cancel_valid;#0.2;if(rst_n)begin',
        '        compare_outputs();comparison_edges++;',
        '        if(sampled_cancel && ffn_bus_owned)`PF_ASSERT(child_cancel_valid,"Cancellation not visible after sampling edge")',
        '    end end']
    marker='    always #2.5 clk=~clk;';assert bank.count(marker)==1
    bank=bank.replace(marker,'\n'.join(lines)+'\n'+marker)
    marker='        $finish;';assert bank.count(marker)==1
    bank=bank.replace(marker,'        `PF_ASSERT(cancellation_delays>0 && comparison_edges>1000,"Comparison coverage")\n        $display("PFCANCEL_PASS seed=%0d jobs=%0d delayed_samples=%0d compared_edges=%0d",seed,jobs,cancellation_delays,comparison_edges);\n'+marker)
    put(ROOT/'fpga/tb/tb_ntrpu_post_ffn_cancel_r1.sv',bank)
    return dict(ports=len(ports),compared_outputs=len(outputs)-1,stimulus_jobs=522,registered_cancel_only=True)

if __name__=='__main__':print(__import__('json').dumps(build()))
