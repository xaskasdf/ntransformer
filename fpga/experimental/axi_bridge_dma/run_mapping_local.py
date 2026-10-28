"""Local RTL mapping/client regression, with frozen dependencies and public-bus review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
FILES = ['ntrpu_axi_bridge_writer.sv', 'ntrpu_axi_bridge_mapper.sv',
         'ntrpu_axi_bridge_endpoint_client.sv', 'tb_ntrpu_axi_bridge_mapping.sv',
         'run_mapping_local.py', 'MAPPING.md']
MASTER_SHA = '5318eb1381e87224f0dafe7d31aec4067df800b3229a1fad6f6b4614d685d3fc'
WRITER_SHA = '885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6'
EXPECTED = {0:(4,2,0),1:(4,2,0),2:(1,0,4),3:(2,1,2),4:(3,2,4),
            5:(4,2,4),6:(1,0,2),7:(2,1,16),8:(1,0,16),9:(2,1,8),10:(0,0,1)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def review(log, seed):
    require('FATAL' not in log and 'ERROR' not in log, 'Simulator diagnostics')
    require(f'MAPPING_LOCAL_PASS seed={seed} cases=11' in log, 'Complete marker')
    cases = re.findall(r'MAP CASE PASS id=(\d+) reads=(\d+) writes=(\d+) fault=([0-9a-f]+) aw=(\d+) bytes=(\d+) results=(\d+)', log)
    require([int(p[0]) for p in cases] == list(range(11)), 'Case coverage/order')
    events = {c:[] for c in range(11)}
    for line in log.splitlines():
        if line.startswith('MC '):
            _, kind, *fields = line.split()
            item = dict(field.split('=',1) for field in fields)
            events[int(item.pop('case'))].append((kind,item))
    checked=[]
    for row in cases:
        case,reads,writes,fault,aw,count,results = [int(row[0]),int(row[1]),int(row[2]),int(row[3],16),*map(int,row[4:])]
        er,ew,ef = EXPECTED[case]
        require((reads,writes)==(er,ew), f'Control counts {case}')
        require(fault==0 if ef==0 else fault & ef == ef, f'Fault classification {case}')
        require((aw,count,results)==((1,64 if case==0 else 128,1) if case<2 else (0,0,0)), f'Data counts {case}')
        outstanding_aw=outstanding_w=outstanding_ar=None
        transactions=[];payload=bytearray();data_addresses=[];config=[];reports=[];acks=[]
        kinds=[]
        pcie=0x9abcdef000002000 if case==1 else 0x1234567800001000
        for kind,e in events[case]:
            kinds.append(kind)
            if kind=='CFG':config.append(e)
            elif kind=='CFG_R':reports.append(e)
            elif kind=='AW':
                require(outstanding_aw is None,'Overlapping AW');outstanding_aw=int(e['address'],16)
            elif kind=='W':
                require(outstanding_w is None,'Overlapping W');outstanding_w=int(e['data'],16)
            elif kind=='B':
                require(outstanding_aw is not None and outstanding_w is not None,'B before AW/W')
                transactions.append(('W',outstanding_aw,outstanding_w,int(e['response'])))
                outstanding_aw=outstanding_w=None
            elif kind=='AR':
                require(outstanding_ar is None,'Overlapping AR');outstanding_ar=int(e['address'],16)
            elif kind=='R':
                require(outstanding_ar is not None,'R before AR')
                transactions.append(('R',outstanding_ar,int(e['data'],16),int(e['response'])))
                outstanding_ar=None
            elif kind=='DMA_AW':data_addresses.append(e)
            elif kind=='DMA_W':
                require(int(e['offset'])==len(payload),'Data offset')
                payload.extend(int(e['data'],16).to_bytes(64,'little'))
                require(int(e['last'])==int(len(payload)==count),'Data last')
            elif kind=='ACK':acks.append(e)
        require(outstanding_aw is None and outstanding_w is None and outstanding_ar is None,'Undrained reviewed control trace')
        require(len(config)==len(reports)==1, f'Configuration count {case}')
        require(int(config[0]['session'],16)==9 and int(config[0]['generation'],16)==(2 if case==1 else 1), 'Configuration identity')
        require(int(config[0]['pcie'],16)==pcie and int(config[0]['axi'],16)==(0x20000000 if case==10 else 0x10000000), 'Configuration image')
        require(int(reports[0]['error'])==int(case>=2),'Configuration verdict')
        canonical=[('R',0x130),('R',0x204),('W',0x208),('W',0x20c),('R',0x208),('R',0x20c)]
        require([(k,a) for k,a,_,_ in transactions]==canonical[:er+ew],f'Exact register sequence {case}')
        for kind,address,data,response in transactions:
            require(response==(2 if (case==3 and kind=='W') or (case==6 and kind=='R') else 0),'AXI response')
            expected={0x130:0 if case==2 else 9,0x204:0x03800000,0x208:pcie>>32,0x20c:pcie&0xffffffff}[address]
            if kind=='R' and case==4 and address==0x208:expected^=1
            if kind=='R' and case==5 and address==0x20c:expected^=0x1000
            require(data==expected,f'CSR data {case}:{address:x}')
        require(bytes(payload)==bytes(((case*13+i*37)^(i>>3))&255 for i in range(count)),f'Payload {case}')
        require(len(data_addresses)==aw,'Observed data AW')
        if case<2:
            require(int(data_addresses[0]['translated'],16)==pcie and int(data_addresses[0]['axi'],16)==0x10000000,'Modeled translated address')
            require(len(acks)==1 and int(acks[0]['tag'],16)==(9<<64)|(case+1) and int(acks[0]['generation'],16)==case+1,'ACK identity')
            require(kinds.index('DMA_AW')>max(i for i,k in enumerate(kinds) if k in ('R','B')),'DMA before control verification')
            require(kinds.index('RESULT')>kinds.index('ACK')>kinds.index('DMA_B'),'Completion causality')
        checked.append(dict(case=case,reads=reads,writes=writes,bytes=count))
    coverage=re.search(r'MAP COVER aw=(\d+) w=(\d+) ar=(\d+)',log)
    require(coverage is not None and all(int(x)>0 for x in coverage.groups()),'Actual stalled control offers')
    return dict(cases=checked,scenario_count=11,payload_bytes=192,native_ip=False)


def run(output, toolchain, seeds):
    output.mkdir(parents=True,exist_ok=False)
    frozen=output/'source';frozen.mkdir()
    master=HERE/'support/ntrpu_xdma_local_master.sv'
    if not master.exists():master=HERE.parents[1]/'rtl/ntrpu_xdma_local_master.sv'
    require(sha(master)==MASTER_SHA,'Frozen local master changed')
    require(sha(HERE/FILES[0])==WRITER_SHA,'Frozen writer changed')
    sources={n:HERE/n for n in FILES};sources['support/ntrpu_xdma_local_master.sv']=master
    hashes={n:sha(p) for n,p in sources.items()}
    for n,p in sources.items():
        q=frozen/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    bins=toolchain/'mingw64/bin'
    tools={p.relative_to(toolchain).as_posix():sha(p) for p in toolchain.rglob('*') if p.is_file()}
    env=dict(os.environ);windows=Path(env.get('SystemRoot','C:/Windows'))
    env['PATH']=os.pathsep.join(map(str,[bins,windows/'System32',windows]))
    result=dict(status='FAILED',sources=hashes,toolchain=tools,runs=[],rtl_negative_controls=[],
                native_ip=False,hardware_exclusion_validated=False,profile_native_verified=False,synthesized=False)
    start=time.monotonic()
    rtl=[frozen/n for n in sources if n.endswith('.sv')]
    def execute(command,cwd,name,expect_success=True):
        job=subprocess.run(list(map(str,command)),cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
        log=job.stdout+job.stderr;(cwd/name).write_text(log,encoding='utf-8')
        if expect_success:require(job.returncode==0,f'{name}: exit {job.returncode}')
        return job.returncode,log
    def compile_at(cwd,files):
        _,log=execute([bins/'iverilog.exe','-g2012','-Wall','-s','tb_ntrpu_axi_bridge_mapping','-o',cwd/'mapping.vvp',*files],cwd,'compile.log')
        require(not log.strip(),'Unexpected compilation diagnostics')
    try:
        execute([bins/'iverilog.exe','-V'],output,'compiler.log')
        execute([bins/'vvp.exe','-V'],output,'simulator.log')
        compile_at(output,rtl)
        for seed in range(seeds):
            case=output/f'seed-{seed:02}';case.mkdir()
            _,log=execute([bins/'vvp.exe',output/'mapping.vvp',f'+seed={seed}'],case,'simulation.log')
            checked=review(log,seed)
            # An evidence mutation must fail the independent public-trace review.
            target=next(line for line in log.splitlines() if line.startswith('MC R case=0 data=12345678'))
            damaged=log.replace(target,target.replace('12345678','12345679'),1)
            try:review(damaged,seed)
            except ValueError:checked['readback_corruption_rejected']=True
            else:raise ValueError('Readback evidence mutation escaped')
            (case/'review.json').write_text(json.dumps(checked,indent=2)+'\n',encoding='utf-8')
            result['runs'].append(dict(seed=seed,**checked))
        original=(frozen/'ntrpu_axi_bridge_mapper.sv').read_text(encoding='utf-8')
        mutations={
            'wrong_upper_word':('req_data=rsp_pcie_base[63:32];','req_data=rsp_pcie_base[31:0];','mapping verdict'),
            'skip_readback_comparison':('(!req_write && (rsp_data & mask)!=(expected & mask));',"1'b0;",'mapping verdict'),
            'publish_after_writes':('else if(step==5)begin','else if(step==3)begin','complete read-back sequence'),
        }
        for name,(before,after,diagnostic) in mutations.items():
            require(original.count(before)==1,'Unique mutation: '+name)
            case=output/'mutations'/name;case.mkdir(parents=True)
            mutated=case/'mapper.sv';mutated.write_text(original.replace(before,after),encoding='utf-8')
            compile_at(case,[mutated if p.name=='ntrpu_axi_bridge_mapper.sv' else p for p in rtl])
            code,log=execute([bins/'vvp.exe',case/'mapping.vvp','+seed=0'],case,'simulation.log',False)
            require(code!=0 and diagnostic in log and 'TIMEOUT' not in log,'Semantic mutation rejection: '+name)
            result['rtl_negative_controls'].append(dict(name=name,diagnostic=diagnostic,returncode=code))
        result['status']='AXI_BRIDGE_MAPPING_CLIENT_LOCAL_PASS'
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:result['error']=str(error)
    result['seconds']=round(time.monotonic()-start,6)
    result['sources_after']={n:sha(p) for n,p in sources.items()}
    result['toolchain_unchanged']=tools=={p.relative_to(toolchain).as_posix():sha(p) for p in toolchain.rglob('*') if p.is_file()}
    if hashes!=result['sources_after'] or not result['toolchain_unchanged']:result.update(status='FAILED',error='Inputs changed')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    manifest={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'FILES.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);ap.add_argument('--toolchain',type=Path,required=True)
    ap.add_argument('--seeds',type=int,default=4)
    a=ap.parse_args()
    if not 1<=a.seeds<=16:ap.error('--seeds must be 1..16')
    result=run(a.output.resolve(),a.toolchain.resolve(),a.seeds)
    print(json.dumps({k:result[k] for k in ('status','seconds','error') if k in result}))
    raise SystemExit(0 if result['status']=='AXI_BRIDGE_MAPPING_CLIENT_LOCAL_PASS' else 1)
