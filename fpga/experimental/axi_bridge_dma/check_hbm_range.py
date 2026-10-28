"""Replay solver counterexamples against HDL and reject corrupted transaction evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from review_hbm_range import review, sha

def literal(value,mask,width):return str(width)+"'b"+''.join(str((value>>i)&1) if (mask>>i)&1 else 'x' for i in reversed(range(width)))

def check(root,proof,out,toolchain):
    if out.exists():raise ValueError('New negative-control output required')
    baseline=review(root);out.mkdir(parents=True);fixture=out/'fixture';fixture.mkdir()
    manifest=json.loads((root/'FILES.json').read_text())
    for name in [*manifest,'FILES.json']:
        target=fixture/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    variants=[
        ('missing_predicate_verdict','range-predicate.log',lambda t:t.replace('HR PASS','HR OMITTED',1)),
        ('missing_success_result','source-case-00-seed-0.log',lambda t:re.sub(r'^HS RESULT[^\n]*\n','',t,count=1,flags=re.M)),
        ('corrupted_payload','cdc-case-00-seed-0.log',lambda t:re.sub(r'(HS W[^\n]*data=)([0-9a-fA-F])',lambda m:m[1]+format(int(m[2],16)^1,'x'),t,count=1)),
        ('lost_fault_ownership','source-case-01-seed-0.log',lambda t:t.replace('owned=1','owned=0',1)),
        ('modified_range_gate','source/ntrpu_axi_bridge_hbm_source.sv',lambda t:t.replace('src_req_bytes>13\'d0','src_req_bytes>13\'d64',1)),
    ]
    records=[];original_manifest=(fixture/'FILES.json').read_bytes()
    for name,path,mutate in variants:
        p=fixture/path;old=p.read_bytes();text=p.read_text();changed=mutate(text)
        if changed==text:raise ValueError('Mutation not applied: '+name)
        p.write_text(changed);modified=dict(manifest);modified[path]=sha(p)
        (fixture/'FILES.json').write_text(json.dumps(modified,indent=2)+'\n')
        case=out/name;case.mkdir();(case/p.name).write_bytes(p.read_bytes())
        try:review(fixture)
        except ValueError as exc:records.append(dict(name=name,path=path,rejected=True,reason=str(exc),sha256=sha(p)))
        else:raise ValueError('Corrupted evidence accepted: '+name)
        finally:p.write_bytes(old);(fixture/'FILES.json').write_bytes(original_manifest)
    if review(fixture)!=baseline:raise ValueError('Fixture not restored')
    queries=json.loads((proof/'RESULT.json').read_text())['queries']
    witnesses=[q for q in queries if q['result']=='sat']
    if len(witnesses)!=5:raise ValueError('Expected five solver counterexamples')
    statements=[]
    for i,row in enumerate(witnesses):
        c=row['counterexample'];a=literal(c['address'],c.get('address_known_mask',(1<<64)-1),64);n=literal(c['length'],c.get('length_known_mask',8191),13)
        statements.append(f"address={a};bytes={n};#1;if(candidate.bad_request!==reference_dut.bad_request || mutant[{i}]===reference_dut.bad_request)$fatal(1,\"WITNESS {i} ineffective\");$display(\"WITNESS {i} old=%b mutant=%b\",reference_dut.bad_request,mutant[{i}]);")
    bench='''`timescale 1ns/1ps
module tb_witness;
reg [63:0] address=0;reg [12:0] bytes=0;
ntrpu_axi_bridge_hbm_source candidate(.clk(1'b0),.cold_rst_n(1'b0),.src_req_address(address),.src_req_bytes(bytes));
reference_hbm_source reference_dut(.clk(1'b0),.cold_rst_n(1'b0),.src_req_address(address),.src_req_bytes(bytes));
wire prefix_bad=bytes==0||bytes>4096|| |address[5:0]|| |bytes[5:0];
wire page_bad=({1'b0,address[11:0]}+bytes)>4096;
wire [64:0] end_address={1'b0,address}+{52'b0,bytes};
wire [4:0] mutant;
assign mutant[0]=prefix_bad||(address>=64'h400000000);
assign mutant[1]=prefix_bad||page_bad||(address>=64'h200000000);
assign mutant[2]=prefix_bad||page_bad||(end_address>=65'h400000000);
assign mutant[3]=prefix_bad||page_bad|| |address[63:34];
assign mutant[4]=prefix_bad||page_bad||(address>=64'h400000000);
initial begin
'''+ '\n'.join(statements)+ '\n$display("WITNESSES PASS count=5");$finish;end\nendmodule\n'
    (out/'tb_witness.sv').write_text(bench);bins=toolchain/'mingw64/bin';env=dict(os.environ);env['PATH']=str(bins)+os.pathsep+env.get('PATH','')
    commands=[('compile',[str(bins/'iverilog.exe'),'-g2012','-s','tb_witness','-o',str(out/'witness.vvp'),str(out/'tb_witness.sv'),str(fixture/'source/ntrpu_axi_bridge_hbm_source.sv'),str(fixture/'source/reference_hbm_source.sv')]),
              ('run',[str(bins/'vvp.exe'),str(out/'witness.vvp')])]
    for name,args in commands:
        r=subprocess.run(args,env=env,text=True,capture_output=True,timeout=60);(out/(name+'.log')).write_text(r.stdout+r.stderr)
        if r.returncode:raise ValueError('Counterexample replay failed: '+name)
    if 'WITNESSES PASS count=5' not in (out/'run.log').read_text():raise ValueError('Missing HDL witness verdict')
    result=dict(status='HBM_RANGE_NEGATIVES_REJECTED',evidence_controls=records,hdl_counterexamples=5)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);ap.add_argument('proof',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--toolchain',type=Path,required=True)
    a=ap.parse_args();print(json.dumps(check(a.root,a.proof,a.output,a.toolchain),indent=2))
