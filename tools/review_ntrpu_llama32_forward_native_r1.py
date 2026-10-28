"""Check complete native forward evidence; never qualifies RTL or a board."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_log(text,reference,negative=False):
    stages=['input-fp32','input-abi','q-fp64','q-fp32','k-fp64','k-fp32','v-fp64','v-fp32',
            'qk-rope-fp32','attention-score-fp64','attention-fp32','o-abi','o-fp64','o-fp32',
            'post-residual-fp32','ffn-abi','gate-fp64','up-fp64','swiglu-fp32','swiglu-abi',
            'down-fp64','down-fp32','output-fp32']
    names=['embedding-fp32']+['layer-%02d-%s'%(i,s) for i in range(16) for s in stages]
    names+=['output-abi','logits-fp64']+([] if negative else ['logits-fp32'])
    expected=[(n,str((reference/(n+'.bin')).stat().st_size)) for n in names]
    actual=re.findall(r'^FORWARD_COMPARE name=(\S+) bytes=(\d+) exact=1$',text,re.M)
    assert actual==expected,'Incomplete/duplicate/reordered arithmetic comparisons'
    assert re.findall(r'^FORWARD_LAYER layer=(\d+) complete=1$',text,re.M)==[str(i) for i in range(16)]
    heads=re.findall(r'^FORWARD_HEAD first=(\d+) rows=(\d+)$',text,re.M)
    assert heads==[(str(i*8192),str(min(8192,128256-i*8192))) for i in range(16)],'Output chunk coverage'
    passes=re.findall(r'^FORWARD_PASS layers=16 token=0 position=0 logits=128256 argmax=475 compared_bytes=6700032 mac_blocks=38617088 cpu_seconds=([0-9.e+-]+)$',text,re.M)
    if negative:
        assert not passes and 'FORWARD_PASS' not in text
        failures=re.findall(r'^FORWARD_FAIL logits-fp32 byte=513023 actual=(\d+) expected=(\d+)$',text,re.M)
        assert len(failures)==1 and int(failures[0][0])^int(failures[0][1])==1,'Tail corruption not detected'
    else:
        assert len(passes)==1 and float(passes[0])>0 and 'FORWARD_FAIL' not in text
        assert sum(int(size) for _,size in expected)==6700032
    lines=text.splitlines()
    assert len(lines)==len(expected)+16+16+1,'Unexpected/unparsed diagnostic'
    return dict(comparisons=len(expected),layers=16,head_chunks=16,logits=128256,
                compared_bytes=sum(int(s) for _,s in expected),mac_blocks=38617088,
                cpu_seconds=None if negative else float(passes[0]),negative_tail=negative)

def restored(path):
    receipt=json.loads((path/'VERIFIED.json').read_text())
    assert receipt['local_and_nfs_verified'] and receipt['restoration_verified']
    assert sha(path/'results.zip')==receipt['sha256']
    root=path/'restored';files=json.loads((root/'RESULT_FILES.json').read_text())
    assert len(files)==receipt['files'] and all(sha(root/n)==h for n,h in files.items())
    return root

def check_chunk(text,reference,first,end,terminal):
    stages=['input-fp32','input-abi','q-fp64','q-fp32','k-fp64','k-fp32','v-fp64','v-fp32',
            'qk-rope-fp32','attention-score-fp64','attention-fp32','o-abi','o-fp64','o-fp32',
            'post-residual-fp32','ffn-abi','gate-fp64','up-fp64','swiglu-fp32','swiglu-abi',
            'down-fp64','down-fp32','output-fp32']
    names=(['embedding-fp32'] if first==0 else [])+['layer-%02d-%s'%(i,s) for i in range(first,end) for s in stages]
    if end==16:names+=['output-abi','logits-fp64']+([] if terminal=='negative' else ['logits-fp32'])
    expected=[(n,str((reference/(n+'.bin')).stat().st_size)) for n in names]
    assert re.findall(r'^FORWARD_COMPARE name=(\S+) bytes=(\d+) exact=1$',text,re.M)==expected
    assert re.findall(r'^FORWARD_LAYER layer=(\d+) complete=1$',text,re.M)==[str(i) for i in range(first,end)]
    heads=re.findall(r'^FORWARD_HEAD first=(\d+) rows=(\d+)$',text,re.M)
    assert heads==([(str(i*8192),str(min(8192,128256-i*8192))) for i in range(16)] if end==16 else [])
    compared=sum(int(size) for _,size in expected);mac=(end-first)*1900544+(8208384 if end==16 else 0)
    prefix='FORWARD_CHUNK_PASS first=%d end=%d compared_bytes=%d mac_blocks=%d'%(first,end,compared,mac)
    if terminal=='chunk':assert text.splitlines()[-1]==prefix
    elif terminal=='head':
        assert re.fullmatch(r'FORWARD_HEAD_PASS first=%d end=16 token=0 position=0 logits=128256 argmax=475 compared_bytes=%d mac_blocks=%d cpu_seconds=[0-9.e+-]+'%(first,compared,mac),text.splitlines()[-1])
    elif terminal=='negative':
        match=re.fullmatch(r'FORWARD_FAIL logits-fp32 byte=513023 actual=(\d+) expected=(\d+)',text.splitlines()[-1])
        assert match and int(match[1])^int(match[2])==1
    else:
        assert terminal=='timeout' and first==0 and end==3
        assert text.splitlines()[-1]=='FORWARD_LAYER layer=2 complete=1'
    assert len(text.splitlines())==len(expected)+(end-first)+len(heads)+int(terminal!='timeout')
    assert ('FORWARD_FAIL' in text)==(terminal=='negative')
    return dict(first=first,end=end,comparisons=len(expected),compared_bytes=compared,
                mac_blocks=mac,head_chunks=len(heads),terminal=terminal)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stdlib',type=Path,required=True);p.add_argument('--amd',type=Path,required=True)
    p.add_argument('--chunks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    roots={'stdlib':restored(a.stdlib),'amd':restored(a.amd),'chunks':restored(a.chunks)};results={};logs={}
    manifests=[json.loads((r/'source/reference/manifest.json').read_text()) for r in roots.values()]
    assert manifests[0]==manifests[1]==manifests[2];m=manifests[0]
    sources=[json.loads((r/'source/MANIFEST.json').read_text()) for r in roots.values()]
    kernels=[{n:h for n,h in s['files'].items() if n.startswith('fpga/hls/')} for s in sources]
    assert kernels[0]==kernels[1]==kernels[2] and len(kernels[0])==22
    assert m['model_sha256']=='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
    assert len(m['tensor_sources'])==147 and m['argmax_token']==475
    for i,layer in enumerate(m['layers']):
        assert layer['layer']==i
        assert layer['input_sha256']==(m['files']['embedding-fp32.bin'] if i==0 else m['layers'][i-1]['output_sha256'])
        assert layer['output_sha256']==m['files']['layer-%02d-output-fp32.bin'%i]
    modes=[('run-stdlib','stdlib',0,16,'full'),('run-amd','amd',0,3,'timeout')]
    modes += [('chunk-%02d'%i,'chunks',i,i+1,'head' if i==15 else 'chunk') for i in range(3,16)]
    modes += [('negative-amd','chunks',16,16,'negative')]
    for mode,variant,first,end,terminal in modes:
        negative=terminal=='negative';math='stdlib' if variant=='stdlib' else 'amd'
        root=roots[variant];record=json.loads((root/(mode+'.json')).read_text())
        compile_record=json.loads((root/('compile-'+math+'.json')).read_text())
        assert compile_record['exit_code']==0 and compile_record['source_unchanged']
        assert record['exit_code']==(124 if terminal=='timeout' else int(negative)) and record['model_verified'] and record['source_unchanged']
        assert record['cpu_affinity']==[2,3] and record['vivado_launched'] is False
        assert record['model_sha256']==m['model_sha256']
        assert sha(root/('forward-'+math))==record['binary_sha256']==compile_record['binary_sha256']
        if math=='amd':
            assert record['runtime_libraries']==compile_record['runtime_libraries']
            assert record['header_dependencies']==compile_record['header_dependencies']
            assert any(n.endswith('/hls_math.h') for n in record['header_dependencies'])
            assert any('/libhlsm-GCC46.so' in n for n in record['runtime_libraries'])
            assert any('/libhlsmc++-GCC46.so' in n for n in record['runtime_libraries'])
        log=root/(mode+'.log');assert sha(log)==record['log_sha256'];text=log.read_text()
        if variant=='chunks':
            assert record['first_layer']==first and record['end_layer']==end
            assert record['input_sha256']==m['files']['layer-%02d-output-fp32.bin'%(first-1)]
            previous=roots['amd'] if first==3 else roots['chunks']
            prevmode='run-amd' if first==3 else 'chunk-%02d'%(first-1)
            prevrecord=json.loads((previous/(prevmode+'.json')).read_text())
            native_path=prevrecord['command'][4]+'/layer-%02d-output-fp32.bin'%(first-1)
            assert record['input_path']==native_path,'Expected data used instead of native predecessor'
        logs[mode]=text
        results[mode]=check_log(text,root/'source/reference') if terminal=='full' else check_chunk(text,root/'source/reference',first,end,terminal)
        expected_outputs=['layer-%02d-output-fp32.bin'%i for i in range(first,end)]+(['logits-fp32.bin'] if end==16 and not negative else [])
        assert set(record['outputs'])==set(expected_outputs)
        for n in expected_outputs:
            assert sha(root/(mode+'-output')/n)==record['outputs'][n]==m['files'][n]
    # Mutations attack coverage and terminal acceptance; no numerical tolerances.
    positive=logs['run-stdlib'];negative=logs['negative-amd'];ref=roots['amd']/'source/reference'
    mutants=[('missing_layer',positive.replace('FORWARD_LAYER layer=15 complete=1\n',''),False),
             ('duplicate_stage',positive.replace('FORWARD_COMPARE name=embedding-fp32 bytes=8192 exact=1\n','FORWARD_COMPARE name=embedding-fp32 bytes=8192 exact=1\n'*2),False),
             ('missing_tail_chunk',positive.replace('FORWARD_HEAD first=122880 rows=5376\n',''),False),
             ('short_tail_chunk',positive.replace('rows=5376','rows=5344'),False),
             ('wrong_mac_count',positive.replace('mac_blocks=38617088','mac_blocks=38617087'),False),
             ('unparsed_failure',positive+'unexpected error\n',False)]
    rejected=[]
    for name,content,is_negative in mutants:
        try:check_log(content,ref,is_negative)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Mutant accepted: '+name)
    chunk_mutants=[('early_negative',negative.replace('byte=513023','byte=0'),16,16,'negative'),
                   ('negative_false_pass',negative+'FORWARD_PASS\n',16,16,'negative'),
                   ('missing_chunk_stage',logs['chunk-03'].replace('FORWARD_LAYER layer=3 complete=1\n',''),3,4,'chunk'),
                   ('wrong_chunk_end',logs['chunk-03'].replace('end=4','end=5'),3,4,'chunk'),
                   ('missing_chunk_tail',logs['chunk-15'].replace('FORWARD_HEAD first=122880 rows=5376\n',''),15,16,'head')]
    for name,content,first,end,terminal in chunk_mutants:
        try:check_chunk(content,ref,first,end,terminal)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Mutant accepted: '+name)
    positives=[results['run-amd']]+[results['chunk-%02d'%i] for i in range(3,16)]
    assert sum(x['comparisons'] for x in positives)==372
    assert sum(x['compared_bytes'] for x in positives)==6700032
    assert sum(x['mac_blocks'] for x in positives)==38617088
    report=dict(status='PASS_LLAMA32_FULL_FORWARD_NATIVE_R1',results=results,rejected_controls=rejected,
        model_sha256=m['model_sha256'],reference_manifest_sha256=sha(roots['amd']/'source/reference/manifest.json'),
        outputs={n:m['files'][n] for n in ['layer-00-output-fp32.bin','layer-15-output-fp32.bin','logits-fp32.bin']},
        argmax_token=m['argmax_token'],top10=m['top10'],input_token=0,position=0,
        stdlib_archive_sha256=sha(a.stdlib/'results.zip'),amd_archive_sha256=sha(a.amd/'results.zip'),
        chunks_archive_sha256=sha(a.chunks/'results.zip'),amd_continuation_from_native_outputs=True,
        cpu_native_only=True,whole_model_rtl_qualified=False,board_qualified=False,multi_token_qualified=False,
        scope='One token at position zero; NTRPU Q8 activations / ordered FP64 arithmetic. Native C++ with standard and AMD math. No Vitis csim/cosim, RTL latency, GPU/BF16 equivalence or text generation.')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))

if __name__=='__main__':main()
