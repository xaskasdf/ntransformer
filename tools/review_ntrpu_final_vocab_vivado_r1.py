"""Bind actual final-norm/head synthesis to sources and independently review protocol traces."""
import hashlib,json,re,zipfile
from pathlib import Path
import check_ntrpu_final_vocab_handoff_r1 as checker
import check_ntrpu_vivado_design_identity_r1 as identity

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def review(run,source_sha):
    run=Path(run).resolve()
    result=dict(run=run.name,source_sha256=source_sha,errors=[],board_power_qualified=False,
        exact_dimm_qualified=False,arithmetic_execution_qualified=False,physical_timing_qualified=False)
    try:
        inv=json.loads((run/'invocation.json').read_text())
        assert 'finished_utc' in inv and inv['exit_code']==0 and inv['source_sha256']==source_sha
        assert inv['frozen_source_unchanged'] and inv['cpu_affinity']==[0,1] and inv['timeout_seconds']==3600
        assert sha(run/'source.zip')==source_sha
        source=Path(inv['source'])
        with zipfile.ZipFile(run/'source.zip') as z:
            hashes=json.loads(z.read('SNAPSHOT_FILES.json'));plan=json.loads(z.read('NATIVE_PLAN.json'))
            assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(hashes)|{'SNAPSHOT_FILES.json'}
        assert all(sha(source/n)==h for n,h in hashes.items())
        console=(run/'console.log').read_text(errors='replace')
        assert console.splitlines().count('NTRPU_FINAL_VOCAB_R1_EXPORTED')==1
        assert not re.search(r'(?im)^\s*(Fatal:|ERROR:)',console)
        files={str(p.relative_to(run)):p.read_text(errors='replace') for p in (run/'output').glob('*')
               if p.is_file() and p.suffix in ('.txt','.log','.rpt','.xdc','.tsv')}
        inputs=files['output/synthesis_inputs.txt'].splitlines()
        assert not any('stub' in n or '/tb/' in n or 'protocol_model' in n for n in inputs)
        copied={}
        for name in plan['compiled_sources']:
            path=run/'output'/(name.split('/')[1]+'_rtl')/Path(name).name if name.startswith('generated/') else source/name
            assert str(path) in inputs and sha(path)==hashes[name],name
            copied[name]=sha(path)
        assert len(copied)==58
        cells=[line.split('\t') for line in files['output/arithmetic_cells.tsv'].splitlines()]
        counts={core:sum(ref==core or re.fullmatch(re.escape(core)+r'_[0-9]+',ref) is not None for name,ref in cells)
                for core in plan['expected_core_counts']}
        assert counts==plan['expected_core_counts'],counts
        compiled={}
        for project in (run/'output/protocol_project').rglob('*.prj'):
            for name in re.findall(r'"([^\\"]+)"',project.read_text()):
                path=(project.parent/name).resolve()
                if path.is_file():compiled[str(path)]=sha(path)
        expected={str(source/n):hashes[n] for n in plan['protocol_sources']}
        assert all(compiled.get(n)==h for n,h in expected.items())
        extra={n:h for n,h in compiled.items() if n not in expected}
        assert len(extra)==1 and Path(next(iter(extra))).name=='glbl.v'
        assert Path(next(iter(extra))).is_relative_to(run/'output/protocol_project')
        suites=[]
        for seed in (1,7,42):
            text=files['output/seed-'+str(seed)+'.log']
            suites.append(dict(metrics=checker.review(text,seed),rejected=checker.controls(text,seed)))
        assert sum(s['metrics']['jobs'] for s in suites)==540 and sum(len(s['rejected']) for s in suites)==174
        evidence=dict(console=console,files=files)
        design=identity.review(evidence,'ntrpu_final_vocab_real_r1')
        design_controls=identity.controls(evidence,'ntrpu_final_vocab_real_r1');assert len(design_controls)==7
        primitive=dict(line.split('\t') for line in files['output/primitive_counts.tsv'].splitlines())
        assert primitive['blackboxes']=='0' and primitive['LDCE']==primitive['LDPE']=='0' and float(primitive['clock_period'])==5
        resources={}
        for label in ('CLB LUTs','CLB Registers','Block RAM Tile','DSPs'):
            match=re.search(r'(?m)^\|\s*'+re.escape(label)+r'\*?\s*\|\s*([0-9.,]+)\s*\|',files['output/utilization.rpt'])
            assert match,label
            resources[label]=float(match[1].replace(',',''));assert resources[label]>0
        assert resources['CLB Registers']==sum(int(primitive[k]) for k in ('FDRE','FDSE','FDCE','FDPE'))
        timing=files['output/timing.rpt'];checks={n:int(v) for n,v in re.findall(r'checking (\w+) \((\d+)\)',timing)}
        for name in ('no_clock','constant_clock','unconstrained_internal_endpoints','multiple_clock','loops','latch_loops'):
            assert checks.get(name)==0,name
        match=re.search(r'^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)',timing,re.M)
        assert match and len(match[1].split())==12
        values=list(map(float,match[1].split()))
        dcp=run/'output/final_vocab_synth.dcp';assert dcp.is_file() and dcp.stat().st_size>0
        assert 'create_clock -name clk -period 5.000 [get_ports clk]' in files['output/clock.xdc']
        result.update(design_identity=design,identity_controls=design_controls,resources=resources,
            setup_wns_ns=values[0],hold_whs_ns=values[4],timing_checks=checks,core_counts=counts,
            copied_source_hashes=copied,protocol_compiled_sources=compiled,suites=suites,
            checkpoint_sha256=sha(dcp),invocation_sha256=sha(run/'invocation.json'),
            scope='Actual HLS normalization plus original dual-MAC vocabulary synthesis. Independent protocol simulation; no joint arithmetic, routed timing or full model claim.')
    except Exception as exc:result['errors'].append(repr(exc))
    result['status']='PASS_FINAL_VOCAB_R1_SYNTH_PROTOCOL' if not result['errors'] else 'DIAGNOSTIC_FINAL_VOCAB_R1'
    return result
