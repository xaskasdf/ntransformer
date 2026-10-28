"""Freeze and validate the admission guard locally; never launch remote jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from audit_host_boundary import audit, read_inputs
from review_guard import review, require

HERE=Path(__file__).resolve().parent
FILES=['ntrpu_axi_bridge_guard.sv','ntrpu_axi_bridge_writer.sv','tb_ntrpu_axi_bridge_guard.sv',
       'run_guard_local.py','review_guard.py','audit_host_boundary.py','GUARD.md']
CLOCKS=[(2,5,1),(4,7,3),(7,2,2),(2,2,5)]
WRITER_SHA='885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def run(output,toolchain,example):
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source';source.mkdir()
    hashes={n:sha(HERE/n) for n in FILES}
    require(hashes['ntrpu_axi_bridge_writer.sv']==WRITER_SHA,'Unchanged validated writer')
    for name in FILES:shutil.copy2(HERE/name,source/name)
    top,app,xci=read_inputs(example)
    boundary=output/'boundary';(boundary/'imports').mkdir(parents=True)
    for path,content in [('imports/xilinx_axi_pcie4_ep.v',top),('imports/xilinx_axi_pcie_ep_app.v',app),('ntrpu_bridge.xci',xci)]:
        (boundary/path).write_bytes(content)
    binaries=toolchain/'mingw64/bin'
    tools={p.relative_to(toolchain).as_posix():sha(p) for p in sorted(toolchain.rglob('*')) if p.is_file()}
    env=dict(os.environ);env['PATH']=str(binaries)+os.pathsep+env.get('PATH','')
    result=dict(status='FAILED',sources=hashes,toolchain_sha256=tools,runs=[],rtl_negative_controls=[],
                audit_negative_controls=[],evidence_negative_controls=[],native_ip_simulated=False,
                synthesis_or_physical_cdc_validated=False)
    started=time.monotonic()

    def execute(command,cwd,log_name,success=True):
        job=subprocess.run(list(map(str,command)),cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
        log=job.stdout+job.stderr;(cwd/log_name).write_text(log,encoding='utf-8')
        if success:require(job.returncode==0,f'{log_name}: exit {job.returncode}')
        return job.returncode,log

    def compile_at(cwd,guard):
        _,log=execute([binaries/'iverilog.exe','-g2012','-Wall','-s','tb_ntrpu_axi_bridge_guard','-o',cwd/'guard.vvp',
                       guard,source/'ntrpu_axi_bridge_writer.sv',source/'tb_ntrpu_axi_bridge_guard.sv'],cwd,'compile.log')
        require(not log.strip(),'Unexpected compilation diagnostics')

    try:
        result['host_boundary']=audit(top,app,xci)
        save(boundary/'REVIEW.json',result['host_boundary'])
        variants=[('control_address_from_host',top.replace(b'board.probe.m_axil_awaddr',b'm_axi_awaddr'),app,xci,'Private control binding'),
                  ('control_data_from_host',top.replace(b'board.probe.m_axil_wdata',b'm_axi_wdata'),app,xci,'Private control binding'),
                  ('target_rewired_to_control',top.replace(b'.s_axi_awaddr         (awaddr)',b'.s_axi_awaddr         (board.probe.m_axil_awaddr)'),app,xci,'Incoming target binding')]
        altered=json.loads(xci)
        altered['ip_inst']['parameters']['component_parameters']['axilite_master_en'][0]['value']='true'
        variants.append(('extra_host_axil_master',top,app,json.dumps(altered).encode(),'Reviewed IP profile'))
        altered=json.loads(xci)
        altered['ip_inst']['parameters']['component_parameters']['axisten_freq'][0]['value']='125'
        variants.append(('unreviewed_xci_change',top,app,json.dumps(altered).encode(),'Reviewed complete XCI identity'))
        for name,t,a,x,reason in variants:
            require((t,a,x)!=(top,app,xci),'Nontrivial audit mutation')
            try:audit(t,a,x)
            except ValueError as error:
                require(reason in str(error),'Specific audit rejection')
                result['audit_negative_controls'].append(dict(name=name,rejected=True,reason=str(error)))
            else:raise ValueError('Audit mutation escaped: '+name)
        execute([binaries/'iverilog.exe','-V'],output,'compiler.log')
        execute([binaries/'vvp.exe','-V'],output,'simulator.log')
        compile_at(output,source/'ntrpu_axi_bridge_guard.sv')
        for index,clocks in enumerate(CLOCKS):
            for case in range(17):
                out=output/f'clock-{index}/case-{case:02}';out.mkdir(parents=True)
                _,log=execute([binaries/'vvp.exe',output/'guard.vvp',f'+case={case}',f'+axi_half={clocks[0]}',
                               f'+mgmt_half={clocks[1]}',f'+phase={clocks[2]}'],out,'simulation.log')
                checked=review(log,(out/'guard.vcd').read_text(),case,clocks)
                checked.update(log_sha256=sha(out/'simulation.log'),vcd_sha256=sha(out/'guard.vcd'))
                save(out/'REVIEW.json',checked);result['runs'].append(checked)
        original=(source/'ntrpu_axi_bridge_guard.sv').read_text()
        mutations=[
            ('disable_watchdog', 'if(age==CLOCK_TIMEOUT-1)clock_alive<=0;', 'if(age==CLOCK_TIMEOUT-1)clock_alive<=1;',3,'operational fault closes admission'),
            ('ignore_perst', ' && perst_n;', ';',5,'operational fault closes admission'),
            ('auto_clear_fault', 'fault_code<=fault_code | {!clock_alive,!healthy_m[2]};\n            end',
             'fault_code<=fault_code | {!clock_alive,!healthy_m[2]};\n            end else begin fault<=0;fault_code<=0;end',3,'fault sticky after clock and link recovery'),
            ('fault_resets_context', 'assign client_cold_rst_n=cold_a[2];', 'assign client_cold_rst_n=cold_a[2] && !fault;',9,'operational event never asserts cold reset')]
        for name,before,after,case,reason in mutations:
            require(original.count(before)==1,'Unique RTL mutation '+name)
            out=output/'mutations'/name;out.mkdir(parents=True)
            mutated=out/'guard.sv';mutated.write_text(original.replace(before,after),encoding='utf-8')
            compile_at(out,mutated)
            code,log=execute([binaries/'vvp.exe',out/'guard.vvp',f'+case={case}'],out,'simulation.log',False)
            require(code!=0 and reason in log and 'BG TIMEOUT' not in log,'Semantic RTL rejection: '+name)
            result['rtl_negative_controls'].append(dict(name=name,rejected=True,reason=reason))
        out=output/'clock-0/case-09'
        log=(out/'simulation.log').read_text();vcd=(out/'guard.vcd').read_text()
        code=next(line.split()[3] for line in vcd.splitlines() if line.startswith('$var ') and line.split()[4]=='m_wdata')
        lines=vcd.splitlines()
        target=next(i for i,line in enumerate(lines) if line.startswith('b') and line.endswith(' '+code) and 'x' not in line and len(line)>400)
        bits,identifier=lines[target].split();lines[target]='b'+('1' if bits[1]=='0' else '0')+bits[2:]+' '+identifier
        damaged_vcd='\n'.join(lines)+'\n'
        damaged_log=log.replace('BG CASE_PASS case=9 fault=1 code=2','BG CASE_PASS case=9 fault=1 code=1')
        for name,bad_log,bad_vcd in [('vcd_payload',log,damaged_vcd),('log_fault_code',damaged_log,vcd)]:
            require((bad_log,bad_vcd)!=(log,vcd),'Nontrivial evidence mutation')
            case_dir=output/'evidence-mutations'/name;case_dir.mkdir(parents=True)
            (case_dir/'simulation.log').write_text(bad_log);(case_dir/'guard.vcd').write_text(bad_vcd)
            try:review(bad_log,bad_vcd,9,CLOCKS[0])
            except ValueError as error:result['evidence_negative_controls'].append(dict(name=name,rejected=True,reason=str(error)))
            else:raise ValueError('Evidence mutation escaped: '+name)
        result['status']='BRIDGE_GUARD_LOCAL_AND_BOUNDARY_AUDIT_PASS'
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:
        result['error']=str(error)
    result['seconds']=round(time.monotonic()-started,6)
    result['sources_unchanged']=hashes=={n:sha(HERE/n) for n in FILES}
    result['toolchain_unchanged']=tools=={p.relative_to(toolchain).as_posix():sha(p) for p in sorted(toolchain.rglob('*')) if p.is_file()}
    if not result['sources_unchanged'] or not result['toolchain_unchanged']:
        result.update(status='FAILED',error='Inputs changed during validation')
    save(output/'RESULT.json',result)
    save(output/'FILES.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);parser.add_argument('--toolchain',type=Path,required=True)
    parser.add_argument('--example',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.output.resolve(),args.toolchain.resolve(),args.example.resolve())
    print(json.dumps({k:result[k] for k in ('status','seconds','error') if k in result}))
    raise SystemExit(0 if result['status']=='BRIDGE_GUARD_LOCAL_AND_BOUNDARY_AUDIT_PASS' else 1)
