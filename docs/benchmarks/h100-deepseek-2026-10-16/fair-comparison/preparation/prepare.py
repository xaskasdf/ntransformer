import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import common as b

ROOT=b.ROOT
CACHE=b.HOST/'ds41-cache-20261015'
LLAMA=b.HOST/'llama-bench-20261015'

def main():
    shutil.copytree(CACHE/'source', ROOT/'nt-source')
    for name in ['benchmark_ds41.cpp','benchmark_resources.h']:
        shutil.copy2(ROOT/name, ROOT/'nt-source/tools'/name)
    shutil.copy2(CACHE/'environment.json', ROOT/'nt-engine-environment.json')
    shutil.copy2(CACHE/'ctest.log', ROOT/'nt-engine-ctest.log')
    shutil.copy2(CACHE/'cpu-validation.json', ROOT/'nt-engine-cpu-validation.json')
    shutil.copy2(CACHE/'build/libntransformer_lib.a', ROOT/'libntransformer_lib.a')
    shutil.copy2(LLAMA/'llama-source.tar', ROOT/'llama-source.tar')
    shutil.copytree(LLAMA/'build/bin', ROOT/'llama-lib', symlinks=True,
        ignore=lambda path,names:[name for name in names if '.so' not in name])
    with tarfile.open(ROOT/'llama-source.tar') as archive:
        archive.extractall(ROOT/'llama-source', filter='data')
    compiler=b.PREFIX/'bin/x86_64-conda-linux-gnu-g++'
    cuda=b.PREFIX/'targets/x86_64-linux'
    b.command('compile-nt', [compiler,'-std=c++20','-O3','-march=native',
        '-I'+str(ROOT/'nt-source/src'),'-I'+str(ROOT/'nt-source/include'),ROOT/'benchmark_ds41.cpp',ROOT/'libntransformer_lib.a',
        '-L'+str(b.PREFIX/'lib'),'-L'+str(cuda/'lib'),'-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread','-o',ROOT/'bench-nt'])
    source=ROOT/'llama-source'
    b.command('compile-llama', [compiler,'-std=c++20','-O3','-march=native',
        '-I'+str(source/'include'),'-I'+str(source/'ggml/include'),'-I'+str(source/'vendor'),
        ROOT/'benchmark_llama_ds41.cpp','-L'+str(ROOT/'llama-lib'),'-lllama','-lggml','-lggml-base','-lggml-cpu',
        '-L'+str(cuda/'lib/stubs'),'-lcuda','-lpthread','-o',ROOT/'bench-llama'])
    placement={}
    for count in [20,22]:
        actual=json.loads(b.capture([ROOT/'bench-llama','--placement',count,2]))
        assert len(actual)==count and len(set(actual))==count
        resident=[i for i in range(40) if i not in actual]
        assert [sum(i*2//41==owner for i in resident) for owner in [0,1]]==[(40-count)//2]*2
        placement[str(count)]=actual
    assert json.loads(b.capture([ROOT/'bench-llama','--placement',31,1]))==list(range(31))
    for args in [[21,2],[-1,2],[20,3],[0,2]]:
        result=subprocess.run([str(ROOT/'bench-llama'),'--placement',*map(str,args)],env=b.ENV,capture_output=True,text=True)
        assert result.returncode!=0,(args,result.stdout)
    b.save('cpu-validation.json',dict(balanced_placements=placement, legacy_single_gpu_preserved=True, invalid_placements_rejected=True))
    for corpus in ['corpus.txt','corpus-heldout.txt']:shutil.copy2(CACHE/corpus, ROOT/corpus)
    for workload in ['p32','p128','p512','heldout-p512']:
        case=CACHE/('slru-'+workload)
        rows=[json.loads(line) for line in (case/'measurements.jsonl').read_text().splitlines()]
        prompt=json.loads((case/'prompt-ids.json').read_text())
        targets=json.loads((case/'target-ids.json').read_text())
        expected=next(r['predicted_ids'] for r in rows if r['kind']=='completion' and r['round']==0)
        value=dict(prompt_ids=prompt,continuation_ids=targets[:-1],target_ids=targets,
            expected_prediction_ids=expected,nt_reference_logits_sha256=b.sha(case/'all-logits.f32'),
            corpus='corpus-heldout.txt' if workload.startswith('heldout') else 'corpus.txt')
        b.save('workload-'+workload+'.json',value)
    pilot=json.loads((ROOT/'workload-p128.json').read_text())
    pilot['continuation_ids']=pilot['continuation_ids'][:16]
    pilot['target_ids']=pilot['target_ids'][:17]
    pilot['expected_prediction_ids']=pilot['expected_prediction_ids'][:17]
    pilot.pop('nt_reference_logits_sha256')
    b.save('workload-pilot.json',pilot)
    b.save('model-identity.json',b.model_identity())
    b.save('environment.json',dict(
        gpu=b.capture(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv']),
        topology=b.capture(['nvidia-smi','topo','-m']),cpu=b.capture(['lscpu']),memory=b.capture(['free','-b']),
        compiler=b.capture([compiler,'--version']),toolkit=b.capture([b.PREFIX/'bin/nvcc','--version']),
        llama_commit='5210c7c5ed61dddaee6ed476623abf4b63093d16',
        binaries={name:b.sha(ROOT/name) for name in ['bench-nt','bench-llama']},
        nt_library_sha256=b.sha(ROOT/'libntransformer_lib.a'),
        llama_libraries={p.name:b.sha(p) for p in (ROOT/'llama-lib').glob('*.so*') if p.is_file() and not p.is_symlink()},
        llama_ldd=b.capture(['ldd',ROOT/'bench-llama']),nt_ldd=b.capture(['ldd',ROOT/'bench-nt']),
        model_identity_scope='Shard sizes/mtime, not a new full hash of 473 GiB',
        nt_source_sha256={p.relative_to(ROOT/'nt-source').as_posix():b.sha(p) for p in (ROOT/'nt-source').rglob('*') if p.is_file()}))
    assert b.sha(ROOT/'libntransformer_lib.a')==json.loads((ROOT/'nt-engine-environment.json').read_text())['library_sha256']
    environment=json.loads((ROOT/'environment.json').read_text())
    external={}
    for line in environment['llama_ldd'].splitlines():
        if '=>' not in line:continue
        path=Path(line.split('=>',1)[1].strip().split()[0])
        if 'libggml' in path.name and not path.is_relative_to(ROOT):external[str(path)]=b.sha(path)
    environment['external_runtime_sha256']=external
    b.save('environment.json',environment)
    b.status('prepared')
    print('Prepared both harnesses and validated CPU placement; no GPU inference launched.',flush=True)

if __name__=='__main__':main()
