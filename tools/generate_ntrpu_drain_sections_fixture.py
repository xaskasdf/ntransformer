"""Derive integration fixtures from sealed public-port section benches.

Only testbench orchestration is replaced. Original drivers, payload patterns,
AMD IP wiring and transaction logs are retained; no force/release statements.
"""
import hashlib
import json
import pathlib
import re

ROOT=pathlib.Path(__file__).resolve().parents[1]
TB=ROOT/'fpga/tb'
manifest={}

def save(name,text,source):
    assert not re.search(r'\b(force|release)\s+\w+\s*=',text)
    p=TB/name
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    manifest[name]=dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       generated_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

src=TB/'tb_ntrpu_axil_section_post_r2.sv';original=src.read_text()
s=original.split('    task ask;')[0]
header='''module ntrpu_drain_axil_fixture #(parameter integer DEVICE_PERIOD_PS=5000,MODE=0)(
    input wire mgmt_clk,section_request,release_forward,release_local,host_enable,
    input wire [63:0] generation,tag,
    output reg ready_to_close=0,forward_held=0,local_held=0,drained=0,
    output wire host_clk_o,cold_o,section_ack_o,local_idle_o,request_present_o,reply_pending_o,
    output wire [1:0] source_faults_o,output wire section_fault_o
);'''
s=re.sub(r'module ntrpu_axil_section_post_r2_case[^\n]*',header,s,count=1)
s=s.replace('reg mgmt_clk=0,host_clk=0,device_clk=0,host_run=1,device_run=1;',
            'reg host_clk=0,device_clk=0;wire host_run=host_enable,device_run=1;')
s=s.replace('    always #5 mgmt_clk=~mgmt_clk;\n','')
s=s.replace('    reg request=0;','    wire request=section_request;')
s=s.replace("    reg [63:0] tag=64'h100000000,generation=64'h100000001;\n",'')
for task in ['take_aw','take_w','take_b','take_ar','take_r','start_hold','release_reset']:
    match=re.search(r'    task '+task+r'\b.*?endtask',original,re.S);assert match
    s+=match[0]+'\n'
begin=original.index('    always @(posedge host_clk)if(manager_reset_n)begin')
end=original.index('    always @(posedge mgmt_clk)if(cold && !fault_test_active)',begin)
s+=original[begin:end]
s+='''
    assign host_clk_o=host_clk;assign cold_o=cold;
    assign section_ack_o=pair_ack;assign local_idle_o=local_idle;
    assign request_present_o=u_awvalid || u_wvalid || u_arvalid;
    assign reply_pending_o=u_bvalid || u_rvalid;
    assign source_faults_o=gate_fault_sources;assign section_fault_o=|fault_mgmt;
    function [31:0] expected_word(input integer n);
        reg [31:0] data,result;
        begin data=32'ha1b2c3d4+n;result=32'h13579bdf ^ ((32'h1c00+n*4)*32'h01020408);
            for(integer b=0;b<4;b=b+1)if(((n+1)>>b)&1)result[b*8+:8]=data[b*8+:8];
            expected_word=result;
        end
    endfunction
    initial begin
        #300;actuator_cold=1;mtick(3);start_hold();release_reset();cold=1;htick(64);
        for(integer n=0;n<4;n=n+1)begin
            take_aw(32'h7000+n*16);take_w(32'ha1b2c3d4+n,n+1);take_b(0);
            take_ar(32'h7000+n*16,0);take_r(expected_word(n),0);
        end
        take_aw(32'h7400);take_w(32'h89abcdef,10);take_ar(32'h7000,0);
        ready_to_close=1;wait(section_request);wait(u_bvalid && u_rvalid);#0.001;
        check(!pair_ack,"held forwarded B/R prevent section certificate");forward_held=1;
        wait(release_forward);
        fork take_b(0);take_r(expected_word(0),0);join
        wait(pair_ack);htick(2);
        take_aw(32'h9000);take_w(32'haabbccdd,5);take_ar(32'h9000,2);
        wait(u_bvalid && u_rvalid);#0.001;
        check(pair_ack && !local_idle,"actual local replies excluded from section certificate");local_held=1;
        wait(release_local);fork take_b(2);take_r(0,2);join
        wait(local_idle);htick(4);
        check(uaw==6 && uwc==6 && ubc==6 && uar==6 && urc==6,"AXIL upstream transfer counts");
        check(daw==5 && dwc==5 && dbc==5 && dar==5 && drc==5,"AXIL forwarded transfer counts");
        check(!gate_fault && fault_mgmt==0,"AXIL healthy throughout integration");drained=1;
    end
endmodule
'''
save('ntrpu_drain_axil_fixture.sv',s,src)

src=TB/'tb_ntrpu_axi4_section_r2.sv';original=src.read_text()
start=original.index('    initial begin\n        for(integer id=0;id<16;id=id+1)')
s=original[:start]
header='''module ntrpu_drain_axi4_fixture #(parameter integer DEVICE_PERIOD_PS=5000)(
    input wire mgmt_clk,section_request,release_forward,release_local,host_enable,late_offer,
    input wire [63:0] generation,tag,
    output reg ready_to_close=0,forward_held=0,local_held=0,drained=0,late_accepted=0,
    output wire host_clk_o,cold_o,section_ack_o,local_idle_o,request_present_o,reply_pending_o,
    output wire [6:0] source_faults_o,output wire section_fault_o
);'''
s=s.replace('module tb_ntrpu_axi4_section_r2;',header)
s=s.replace('    parameter integer DEVICE_PERIOD_PS=5000;\n','')
s=s.replace('reg mgmt_clk=0,host_clk=0,device_clk=0,run_host=1,run_device=1;',
            'reg host_clk=0,device_clk=0,run_device=1;wire run_host=host_enable;')
s=s.replace('    always #5000 mgmt_clk=~mgmt_clk;\n','')
s=s.replace('reg req_valid=0,release_responses=0,inject_orphan=0;',
            'wire req_valid=section_request;reg release_responses=0,inject_orphan=0;')
s=s.replace("    reg [63:0] generation=64'h100000001,tag=64'h200000001;\n",'')
for name in ['ntrpu_axi4_retained_admission','ntrpu_axi4_section_certificate']:
    s=re.sub(r'\b'+name+r'\b',name+'_r2',s)
cut=original.index('        @(negedge mgmt_clk);req_valid=1;event_log("PARTIAL_GAP_REQUEST");',start)
s+='''    assign host_clk_o=host_clk;assign cold_o=cold_rst_n;
    assign section_ack_o=section_ack;assign local_idle_o=local_idle;
    assign request_present_o=s_awvalid || s_wvalid || s_arvalid;
    assign reply_pending_o=s_bvalid || s_rvalid;
    assign source_faults_o=admission_fault_sources;assign section_fault_o=|fault_mgmt;
'''
s+=original[start:cut]
s+='''        ready_to_close=1;wait(section_request);event_log("INTEGRATED_REQUEST");
        wait(state_debug==1);#1;
        must(w_left==1 && aw_left==0 && admission_w_partial==2 && !s_wvalid,"finite partial-W gap prefix");
        host_w(23,2);host_w(23,3);
        wait(s_bvalid && s_rvalid && host_proof_request);#1;
        must(!section_ack,"held forwarded replies prevent proof");forward_held=1;
        wait(release_forward);release_responses=1;wait(section_ack);#1;
        must(h_bcount==24 && h_rcount==1045 && local_idle,"normal replies precede section proof");
        @(negedge host_clk);release_responses=0;phase=1;
        fork host_aw(24);begin for(integer beat=0;beat<lens[24%7];beat=beat+1)host_w(24,beat);end host_ar(24);join
        wait(s_bvalid && s_rvalid);#1;
        must(!local_idle && section_ack,"local replies excluded from section proof");local_held=1;
        wait(release_local);release_responses=1;
        wait(h_bcount==25 && h_rcount==1061 && local_idle);
        repeat(4)@(posedge host_clk);#1;
        must(fault_mgmt==0 && h_awcount==25 && h_wcount==25 && h_arcount==25,"AXI4 exact integrated transfers");drained=1;
        wait(late_offer);host_aw(25);#1;late_accepted=1;
    end
endmodule
'''
save('ntrpu_drain_axi4_fixture.sv',s,src)
with (TB/'ntrpu_drain_sections_fixture_sources.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps(manifest))
