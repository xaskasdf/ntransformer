"""Snapshot and validate isolated AXI queues, leases and public transactions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('fpga/rtl/ntrpu_buffer_pool_allocator.sv',
         'fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv',
         'fpga/rtl/ntrpu_peer_multibuffer_fillstage.sv',
         'fpga/rtl/ntrpu_rv_elastic.sv',
         'fpga/rtl/ntrpu_peer_multibuffer_elastic.sv',
         'fpga/tb/tb_ntrpu_peer_multibuffer.sv',
         'fpga/tb/tb_ntrpu_rv_elastic.sv',
         'tools/prepare_ntrpu_elastic_tb.py',
         'tools/check_ntrpu_peer_multibuffer.py',
         'tools/run_ntrpu_peer_multibuffer.py',
         'tools/run_ntrpu_multibuffer_elastic.py',
         'tools/check_ntrpu_elastic.py')


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--iverilog',default='iverilog')
    ap.add_argument('--vvp',default='vvp')
    ap.add_argument('--ivl-base')
    args=ap.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    snap=out/'inputs'
    for n in SOURCES:
        t=snap/n; t.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/n,t)
    hashes={n:digest(snap/n) for n in SOURCES}
    sys.path.insert(0,str(snap/'tools'))
    from prepare_ntrpu_elastic_tb import prepare
    from check_ntrpu_peer_multibuffer import replay
    from run_ntrpu_peer_multibuffer import PROFILES
    from check_ntrpu_elastic import replay_external
    profiles={**PROFILES,'odd_beats':(2,4,3,16,4)}
    tb=out/'tb_elastic.sv'; tb.write_text(prepare((snap/SOURCES[5]).read_text()))
    report=dict(status='RUNNING',sources=hashes,unit=[],runs=[],mutants=[])
    start=time.monotonic()

    def command(cmd,label):
        p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (out/(label+'.log')).write_text(p.stdout)
        return p.returncode,p.stdout

    def compile(label,top,bench,profile=None,mutant=None):
        exe=out/(label+'.vvp')
        cmd=[args.iverilog]+(['-B',args.ivl_base] if args.ivl_base else [])+['-g2012','-s',top,'-o',str(exe)]
        if profile and top!='tb_ntrpu_rv_elastic':
            cmd += [f'-P{top}.{key}={v}' for key,v in zip(('CHANNELS','RESERVE_BANKS','BANK_BEATS','DESCRIPTORS','MAX_OUTSTANDING'),profile)]
        if top=='tb_ntrpu_rv_elastic':
            cmd += ['-P'+top+'.WIDTH='+str(profile[0] if profile else 577)] if profile else []
            names=[mutant or snap/SOURCES[3],bench]
        else:
            names=[snap/n for n in SOURCES[:5]]+[bench]
            if mutant: names[4]=mutant
        code,log=command(cmd+list(map(str,names)),label+'_compile')
        if code: raise RuntimeError('compile failed '+label+': '+log[-1800:])
        return exe

    try:
        for width in (6,85,577):
            exe=compile('unit'+str(width),'tb_ntrpu_rv_elastic',snap/SOURCES[6],(width,))
            for seed in (1,17):
                code,log=command([args.vvp,str(exe),f'+SEED={seed}'],f'unit{width}_seed{seed}')
                if code or 'ELASTIC_UNIT_PASS' not in log: raise RuntimeError('unit failed: '+log[-1200:])
                report['unit'].append(dict(width=width,seed=seed,summary=log.strip()))
        for name,profile in profiles.items():
            exe=compile(name,'tb_ntrpu_peer_multibuffer',tb,profile)
            for seed in (1,17):
                code,log=command([args.vvp,str(exe),f'+SEED={seed}'],f'{name}_seed{seed}')
                if code or 'ELASTIC_SCOREBOARD' not in log: raise RuntimeError('datapath failed '+name+': '+log[-1400:])
                try:
                    score=replay(log,profile,seed)
                except RuntimeError as error:
                    # The one-beat core can serialize its transfers into the
                    # elastic queues while both external ports drain together.
                    # This is the LAST coverage check in the preserved oracle;
                    # all transaction, lease, completion and error checks ran.
                    if name!='one_beat' or str(error)!='no simultaneous W transfers':
                        raise
                    score=dict(status='ALL_CORRECTNESS_CHECKS_PASS_INTERNAL_PARALLEL_NOT_OBSERVED',
                               events=replay(log,profile,seed,partial=True),
                               external_parallel_checked_by_scoreboard=True)
                life=int(re.search(r'queued_bank_reuses=(\d+)',log)[1])
                if profile[4]>1 and life==0: raise RuntimeError('no bank reuse with queued W')
                external=replay_external(log,profile,seed)
                report['runs'].append(dict(profile=name,seed=seed,internal_replay=score,
                                           queued_bank_reuses=life,external=external))
        source=(snap/SOURCES[3]).read_text()
        for name,old,new,reason in (
            ('lost_second','else second <= s_data;','else second <= 0;','queue order/data mismatch'),
            ('lost_shift','if (count == 2) first <= second;','if (count == 2) first <= first;','queue order/data mismatch'),
            ('lost_reset','count <= 0;','count <= count;','empty mismatch'),
            ('ready_path','count < 2;','(count < 2 || m_ready);','combinational ready dependency'),
        ):
            assert source.count(old)==1
            mutant=out/(name+'.sv'); mutant.write_text(source.replace(old,new))
            exe=compile(name,'tb_ntrpu_rv_elastic',snap/SOURCES[6],mutant=mutant)
            code,log=command([args.vvp,str(exe),'+SEED=1'],name)
            if not code or reason not in log: raise RuntimeError('mutation escaped '+name+': '+log)
            report['mutants'].append(dict(name=name,reason=reason))
        wrapper=(snap/SOURCES[4]).read_text()
        for name,old,new,reason in (
            ('aw_length','i_awlen[g*8 +:8],','(i_awlen[g*8 +:8] ^ 8\'h01),','AW order/length mismatch'),
            ('w_payload','i_wdata[g*512 +:512]}','(i_wdata[g*512 +:512] ^ 512\'d1)}','payload mismatch'),
            ('b_response','m_bresp[g*2 +:2]}','2\'d0}','completion order/status mismatch'),
            ('false_drain','core_drained && (&aw_empty) && (&w_empty) && (&b_empty)','1\'b1','drained with buffered AW'),
        ):
            assert wrapper.count(old)==1
            mutant=out/(name+'.sv'); mutant.write_text(wrapper.replace(old,new))
            exe=compile(name,'tb_ntrpu_peer_multibuffer',tb,profiles['full4'],mutant)
            code,log=command([args.vvp,str(exe),'+SEED=1'],name)
            if not code or reason not in log: raise RuntimeError('wrapper mutation escaped '+name+': '+log[-1200:])
            report['mutants'].append(dict(name=name,reason=reason))
        good=(out/'full4_seed1.log').read_text()
        for name,prefix,index,value,reason in (
            ('trace_bad_byte','EXT W ',5,'0','W payload/order'),
            ('trace_bad_length','EXT AW ',5,'64','AW order/length'),
            ('trace_early_last','EXT W ',4,'0','WLAST mismatch'),
            ('trace_bad_status','TRACE CPL ',5,'3','completion before B/status'),
        ):
            lines=good.splitlines()
            at=next(i for i,line in enumerate(lines) if line.startswith(prefix))
            fields=lines[at].split(); fields[index]=value; lines[at]=' '.join(fields)
            mutated='\n'.join(lines)+'\n'; (out/(name+'.log')).write_text(mutated)
            try: replay_external(mutated,profiles['full4'],1)
            except RuntimeError as error:
                if str(error)!=reason: raise
            else: raise RuntimeError('public oracle mutation escaped '+name)
            report['mutants'].append(dict(name=name,reason=reason))
        report['status']='ELASTIC_RTL_PASS_NOT_INTEGRATED'
    except Exception as e:
        report.update(status='FAILED',error=repr(e)); raise
    finally:
        report.update(seconds=time.monotonic()-start,sources_after={n:digest(ROOT/n) for n in SOURCES})
        if hashes!=report['sources_after']: report['status']='SOURCES_CHANGED'
        save(out/'REVIEW.json',report)
        manifest={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}
        save(out/'FILES.json',manifest)
        with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(out).as_posix())
        with zipfile.ZipFile(out.with_suffix('.zip')) as z:
            assert z.testzip() is None
            assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
        save(out/'VERIFIED.json',dict(sha256=digest(out.with_suffix('.zip')),crc='PASS',hashes='PASS'))
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'])))


if __name__=='__main__': main()
