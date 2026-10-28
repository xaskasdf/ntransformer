"""Freeze a registered BREADY candidate and complete-loader equivalence bank."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Non-unique source anchor: '+old)
    return text.replace(old,new,1)


def candidate(text):
    text=once(text,'module ntrpu_hbm_weight_loader (','module ntrpu_hbm_bready_loader (')
    return once(text,'    assign h_bready=(mem_cold_rst_n&&ms==M_RESPONSE)?(lane?2\'b10:2\'b01):2\'b00;', '''    wire [1:0] bready_next;
    ntrpu_hbm_bready_predictor bready_predictor(
        .offer(ms==M_OFFER),.response(ms==M_RESPONSE),.lane(lane),
        .aw_pending(aw_pending),.w_pending(w_pending),
        .awready(h_awready),.wready(h_wready),.bvalid(h_bvalid),.next_ready(bready_next));
    reg [1:0] bready_reg=0;
    always @(posedge mem_clk or negedge mem_cold_rst_n)begin
        if(!mem_cold_rst_n)bready_reg<=0;
        else bready_reg<=bready_next;
    end
    assign h_bready=bready_reg;''')


def prepare(donor,bench,output):
    if output.exists():raise ValueError('Fresh output required')
    source=(donor/'ntrpu_hbm_weight_loader.sv').read_text()
    ref=once(source,'module ntrpu_hbm_weight_loader (','module ntrpu_hbm_weight_loader_reference (')
    tb=bench.read_text();outputs=[];declarations=[]
    header=source.split(');',1)[0]
    for kind,width,names in re.findall(r'output\s+(wire|reg)\s*(\[[^\]]+\])?\s*([^\n]+)',header):
        for n in names.rstrip(',').split(','):
            n=n.strip().split('=')[0]
            if not re.fullmatch(r'\w+',n):raise ValueError('Unparsed output '+n)
            outputs.append(n);declarations.append(f'    wire {width or ""} ref_{n};')
    if len(outputs)!=len(set(outputs)) or 'h_bready' not in outputs:raise ValueError('Output inventory')
    binding='\n'.join(declarations)+'\n    ntrpu_hbm_weight_loader_reference reference('+','.join('.'+n+'(ref_'+n+')' for n in outputs)+',.*);'
    compare=[n for n in outputs if n!='h_bready']
    monitors='''
    integer equivalence_checks=0;
    realtime last_memory_edge=0;
    always @(posedge mem_clk)last_memory_edge=$realtime;
    always @(h_bready)if(mem_cold_rst_n&&$realtime!=last_memory_edge)
        $fatal(1,"BREADY changes between memory edges");
    always @(posedge clk or posedge mem_clk or negedge cold_rst_n or negedge mem_cold_rst_n or negedge bus_rst_n)begin
        #0.001;
        if(!mem_cold_rst_n)begin
            if(h_bready!==0)$fatal(1,"asynchronous BREADY clear");
        end else if(h_bready!==ref_h_bready)$fatal(1,"BREADY reference controls");
        if({OUTPUTS}!=={REFERENCE})$fatal(1,"complete loader output equivalence");
        equivalence_checks++;
    end
'''.replace('OUTPUTS',','.join(compare)).replace('REFERENCE',','.join('ref_'+n for n in compare))
    tb=once(tb,'    ntrpu_hbm_weight_loader dut(.*);','    ntrpu_hbm_bready_loader dut(.*);\n'+binding+monitors)
    tb=once(tb,'    reg bad_id=0;','    reg bad_id=0;\n    reg early_b=0;')
    tb=once(tb,'if(delay_b[i]>3+seed)','if(early_b||delay_b[i]>3+seed)')
    tb=once(tb,'hold_b=0;hold_aw=0;hold_w=0;response=0;bad_id=0;mem_enable=1;',
            'hold_b=0;hold_aw=0;hold_w=0;response=0;bad_id=0;early_b=0;mem_enable=1;')
    extra='''        reset_all;early_b=1;
        for(integer ch=0;ch<2;ch++)for(integer order=0;order<3;order++)begin
            stage((64'(ch)<<33)+64'h2000+64'(order*32),64'(100+ch*3+order),10+ch*3+order);
            hold_aw=order==1;hold_w=order==2;write_reg('h80c,1,15,0);ticks(8);
            hold_aw=0;hold_w=0;completion_wait(64'(100+ch*3+order),0);
            require(received_data[ch]===pattern(10+ch*3+order),"earliest response payload");
            write_reg('h810,1,15,0);
        end
        group_pass("earliest_B_both_lanes_all_AW_W_orders");
        reset_all;stage('h3000,200,20);hold_b=1;write_reg('h80c,1,15,0);
        wait(h_bready!=0);mem_enable=0;ticks(4);
        bus_rst_n=0;ticks(3);require(h_bready!=0,"BREADY retained through transport reset with stopped clock");
        cold_rst_n=0;mem_cold_rst_n=0;#0.01;
        require(h_bready===0,"asynchronous BREADY clear with stopped clock");
        reset_all;group_pass("BREADY_stopped_clock_cold_clear");
        $display("BREADY_EQUIVALENCE PASS seed=%0d checks=%0d outputs=OUTPUT_COUNT",seed,equivalence_checks);
'''.replace('OUTPUT_COUNT',str(len(outputs)))
    tb=once(tb,'        $display("WL PASS seed=',extra+'        $display("WL PASS seed=')
    tb=once(tb,'#200000;','#400000;')
    output.mkdir(parents=True)
    for name,text in {'reference_loader.sv':ref,'candidate_loader.sv':candidate(source),'tb_hbm_weight_loader.sv':tb}.items():
        (output/name).write_text(text,newline='\n')
    here=Path(__file__).resolve().parent
    helpers=['prepare_hbm_reset_bready.py','ntrpu_hbm_bready_predictor.sv','tb_hbm_bready_predictor.sv',
             'synthesize_hbm_reset_bready.tcl','run_hbm_reset_bready.py','review_hbm_reset_bready.py']
    for name in helpers:shutil.copy2(here/name,output/name)
    for name in ['ntrpu_control_snapshot.sv','ntrpu_hbm_control_plane.sv']:shutil.copy2(donor/name,output/name)
    record=dict(status='RESET_BREADY_UNIT_PREPARED',donor_files={n:sha(donor/n) for n in ['ntrpu_hbm_weight_loader.sv','ntrpu_control_snapshot.sv','ntrpu_hbm_control_plane.sv']},
                bench_sha256=sha(bench),compared_outputs=outputs,files={p.name:sha(p) for p in output.iterdir()},
                full_board_integrated=False,physical_timing_qualified=False)
    (output/'PREPARED.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['donor','bench','output']:p.add_argument(n,type=Path)
    a=p.parse_args();prepare(a.donor,a.bench,a.output)
