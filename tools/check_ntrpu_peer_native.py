"""Audit HBM preload/readback, peer lifetime and return-queue conservation."""
import argparse
from collections import deque
import json
from pathlib import Path
import re
from run_ntrpu_peer_read import sha
from check_ntrpu_peer_read_oracle import replay as oracle
from check_ntrpu_read_cut import boundary, negative_controls as cut_controls


def replay(text, native):
    assert f'PN ALL_PASS cases=2 native={native}' in text
    assert not re.search(r'(?im)^\s*(Fatal:|ERROR:)|PR_FAIL|PN_WATCHDOG|while executing',text)
    assert 'PR CASE 2 PASS' in text and 'PR CASE 21 PASS' in text and 'PR GLOBAL_STOP_PASS' in text
    memory={};writes={};readq=[deque(),deque()];jobs={};completed=set();published=False
    native_bytes=payload_bytes=0;boot=None;failures=[];cases=set();rejected=0
    for line in text.splitlines():
        p=line.split()
        if p[:2]==['PN','BOOT']:
            assert boot is None and p[3:5]==['1','1'];boot=int(p[2])
        elif p[:2]==['PN','PUBLISHED']:
            assert not published and len(completed)==20;published=True
        elif p[:2]==['PN','WCMD']:
            _,_,ts,epoch,lane,seq,address,beats=p
            key=(int(lane),int(seq));assert key not in writes and not published
            writes[key]=dict(address=int(address),length=int(beats)*64,received=0)
        elif p[:2]==['PN','WI']:
            lane,seq,beat=map(int,p[4:7]);job=writes[lane,seq];raw=bytes.fromhex(p[7])[::-1]
            assert len(raw)==64 and beat*64==job['received'] and job['received']+64<=job['length']
            for offset,value in enumerate(raw):
                address=job['address']+job['received']+offset
                assert value==((address^(address>>8)^(address>>16)^(lane*91))&255)
                assert address not in memory;memory[address]=value
            job['received']+=64
        elif p[:2]==['PN','WCPL']:
            key=tuple(map(int,p[4:6]));assert key not in completed
            assert writes[key]['received']==writes[key]['length'];completed.add(key)
        elif p[:2]==['PN','AR']:
            lane,address,beats=map(int,p[4:7]);assert published and 1<=beats<=16 and address%32==0
            assert all(address+i in memory for i in range(beats*32))
            readq[lane].append([address,beats])
        elif p[:2]==['PN','NR']:
            lane,rid,resp,last=map(int,p[4:8]);assert rid==resp==0 and readq[lane]
            address,beats=readq[lane][0];raw=bytes.fromhex(p[8])[::-1]
            assert len(raw)==32 and raw==bytes(memory[address+i] for i in range(32)) and last==int(beats==1)
            native_bytes+=32
            if beats==1:readq[lane].popleft()
            else:readq[lane][0]=[address+32,beats-1]
        if p[:2]==['PR','CASE']:cases.add(int(p[2]))
        if p[:2]==['PR','REJECT']:rejected+=1;assert int(p[2])==2
        if len(p)<6 or p[0]!='PR' or p[1] not in ('CMD','DATA','LOCAL','DMA','ACK','RESULT'):continue
        kind=p[1];case,epoch,lane,seq=map(int,p[2:6]);key=(lane,seq)
        if kind=='CMD':
            assert published and key not in jobs and case in (2,21)
            jobs[key]=dict(address=int(p[6]),length=int(p[7])*64,received=0,local=False,dma=False,ack=False,result=False)
        elif kind=='RESULT' and case==21:
            assert int(p[6])!=0 and int(p[7])==0 and seq==11
            assert lane not in failures;failures.append(lane)
        else:
            assert case==2 and key in jobs;job=jobs[key]
            if kind=='DATA':
                beat,last=map(int,p[6:8]);raw=bytes.fromhex(p[8])[::-1]
                assert len(raw)==64 and beat*64==job['received']+64 and last==int(beat*64==job['length'])
                assert raw==bytes(memory[job['address']+job['received']+i] for i in range(64))
                job['received']+=64;payload_bytes+=64
            elif kind in ('LOCAL','DMA','ACK'):
                assert int(p[6])==0 and int(p[7])==job['length']==job['received']
                assert not job[kind.lower()];job[kind.lower()]=True
            elif kind=='RESULT':
                assert int(p[6])==0 and int(p[7])==job['length']
                assert job['local'] and job['dma'] and job['ack'] and not job['result'];job['result']=True
    assert cases=={2,21} and rejected==1 and sorted(failures)==[0,1]
    assert boot is not None and (not native or boot>40_000_000)
    assert len(memory)==37376 and len(completed)==20 and payload_bytes==native_bytes==37376
    assert not any(readq) and sum(j['result'] for j in jobs.values())==20
    assert 'PR QUARANTINE 21 1 2' in text and 'PR REUSE_PASS transfers=16' in text
    return dict(boot_ps=boot,preload_transfers=20,native_read_bytes=native_bytes,
                payload_bytes=payload_bytes,successful_results=20,retained_failed_contexts=2,
                oracle=oracle(text))


def controls(text,native):
    trials={name:'\n'.join(s for s in text.splitlines() if not s.startswith(prefix)) for name,prefix in
            [('missing_ack','PR ACK 2 '),('missing_dma','PR DMA 2 '),('missing_local','PR LOCAL 2 '),
             ('missing_preload_completion','PN WCPL '),('missing_native_read','PN NR '),
             ('missing_quarantine','PR QUARANTINE ')]}
    for name,prefix in [('bad_preload','PN WI '),('bad_native_data','PN NR '),('bad_peer_data','PR DATA 2 ')]:
        line=next(s for s in text.splitlines() if s.startswith(prefix));p=line.split()
        p[-1]=format(int(p[-1],16)^1,'0'+str(len(p[-1]))+'x');trials[name]=text.replace(line,' '.join(p),1)
    for name,changed in trials.items():
        try:replay(changed,native)
        except (AssertionError,KeyError,ValueError,IndexError):continue
        raise AssertionError(name+' accepted')
    return list(trials)


def review(out):
    r=json.loads((out/'result.json').read_text())
    assert r['experiment']=='peer_read_native' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['returncode']==0 and not r['timed_out'] and r['sources']==r['sources_after'] and r['donor_unchanged']
    assert r['nice']>=10 and len(set(r['cpus']))==r['threads']
    for n,h in json.loads((out/'FILES.json').read_text()).items():
        assert (out/n).resolve().is_relative_to(out.resolve()) and sha(out/n)==h,n
    profile=json.loads((out/'NATIVE_PROFILE.json').read_text())
    assert r['sources']['NATIVE_PROFILE.json']==sha(out/'NATIVE_PROFILE.json')
    assert all(r['sources'][n]==h for n,h in profile['selected_rtl'].items())
    result=dict(result_sha256=sha(out/'result.json'),seconds=r['seconds'],threads=r['threads'],
                selected_checkpoint_sha256=profile['selected_checkpoint_sha256'],selected_rtl_files=len(profile['selected_rtl']),
                native_hbm_controller=r['stage']=='traffic',pcie_integrated=False,board_qualified=False)
    if r['stage']=='compile':
        inputs=json.loads((out/'SNAPSHOT_INPUTS.json').read_text())
        assert len(inputs)>=10 and sum(v['bytes'] for v in inputs.values())>100_000_000
        assert any('tb_ntrpu_peer_native_behav/xsimk' in n for n in inputs)
        pr=json.loads((out/'PREFLIGHT_REVIEW.json').read_text());assert pr['status']=='PEER_NATIVE_PREFLIGHT_REVIEWED'
        script=(out/'compile.sh').read_text();assert 'stub.sv' not in script and '-generic_top NATIVE=1' in script
        assert re.findall(r'--mt\s+(\d+)',script)==[str(r['threads'])]
        for name,values in profile['selected_read_configs'].items():
            assert dict(s.split('\t',1) for s in (out/'suite'/name).read_text().splitlines() if s.startswith('CONFIG.'))==values
        log=(out/'console.log').read_text();assert not re.search(r'(?im)^\s*(ERROR:|Fatal:)',log)
        assert 'tb_ntrpu_peer_native' in log and 'ntrpu_peer_read_pair' in log
        result.update(status='PEER_NATIVE_SNAPSHOT_REVIEWED',simulation_executed=False,
                      snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'),runtime_files=len(inputs),
                      runtime_bytes=sum(v['bytes'] for v in inputs.values()))
    else:
        native=int(r['stage']=='traffic')
        text=(out/('evidence/simulate.log' if native else 'suite/simulation.log')).read_text()
        result.update(status='PEER_NATIVE_TRAFFIC_REVIEWED' if native else 'PEER_NATIVE_PREFLIGHT_REVIEWED',
                      simulation_executed=True,**replay(text,native),negative_controls=controls(text,native),
                      return_cut=boundary(text.splitlines()),return_cut_negative_controls=cut_controls(text.splitlines()))
        assert result['return_cut']['remaining_at_epoch_end']==0
        if native:
            assert r['runtime_inputs_unchanged'] and 'NTRPU_PEER_NATIVE_WINDOW_COMPLETE' in text
            prior=json.loads((out/'PRIOR_REVIEW.json').read_text())
            assert prior['status']=='PEER_NATIVE_SNAPSHOT_REVIEWED' and prior['result_sha256']==r['prior_result_sha256']
            assert prior['snapshot_manifest_sha256']==sha(out/'SNAPSHOT_INPUTS.json')
            rows=(out/'evidence/progress.tsv').read_text().splitlines();header=rows[0].split('\t')
            assert all(len(s.split('\t'))==len(header) for s in rows[1:])
            values=[dict(zip(header,s.split('\t'))) for s in rows[1:]];ready=[v for v in values if v['started']=='1']
            assert ready and all(v['apb_complete_0']==v['apb_complete_1']==v['mem_rst_n']==v['apb_rst_n']=='1' and v['reader_fault']==v['wfault']=='0' for v in ready)
            result['thermal_observed']=sorted({(v['DRAM_0_STAT_CATTRIP'],v['DRAM_1_STAT_CATTRIP']) for v in ready})
            result['thermal_qualified']=False
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);a=ap.parse_args()
    print(json.dumps(review(a.output),indent=2))
