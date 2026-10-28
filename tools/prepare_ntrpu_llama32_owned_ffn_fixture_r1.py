"""Prepare exact post-O ABI1 sample 1 for full, unchanged layer-0 FFN RTL."""
import argparse,hashlib,json,os
from pathlib import Path
from check_ntrpu_ffn_memory_fixture import decode

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','native-review','packed-parent','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    assert digest(a.reference/'manifest.json')=='f2e3edda6ce1e3b5a6fca72886501e118584cd6004412772c279b7c615df5d09'
    assert digest(a.native_review)=='39b27795d66ffb183023e6257345c7daa1719691c113b9a2a128b547fb641fec'
    native=json.loads(a.native_review.read_text());assert native['status']=='PASS_POST_FFN_NATIVE_R1_INDEPENDENT_REVIEW'
    assert len(native['suites'])==2 and all(s['samples']==4 for s in native['suites']) and len(native['rejected_executions'])==6
    ref=json.loads((a.reference/'manifest.json').read_text())
    for n,h in ref['files'].items():assert digest(a.reference/n)==h,n
    assert digest(a.packed_parent/'manifest.json')=='cf5e8e07c88038ccd0ceb53abcb24a0378870196f4a411b18a1b815b4a133470'
    old=json.loads((a.packed_parent/'manifest.json').read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    weights=[]
    for ident,(rows,blocks) in enumerate(((8192,64),(8192,64),(2048,256)),1):
        name=f'weight-{ident}.hex';source=a.packed_parent/name
        assert digest(source)==old['files'][name]
        image=bytes.fromhex(source.read_text());dense=(a.reference/f'weight-{ident}-dense.bin').read_bytes()
        assert decode(image,rows,blocks,32)==dense and len(dense)==17825792
        # Same immutable D: volume: share file contents without another 160 MB copy.
        os.link(source,a.output/name);assert digest(a.output/name)==old['files'][name]
        weights.append(dict(old['weights'][ident-1],dense_sha256=hashlib.sha256(dense).hexdigest()))
    def write_hex(name,raw):
        with (a.output/name).open('x',encoding='ascii',newline='\n') as f:f.write(raw.hex('\n')+'\n')
        assert bytes.fromhex((a.output/name).read_text())==raw
    mapping={'input':'input-abi','buffer-4':'gate-fp64','buffer-5':'up-fp64','buffer-6':'activation-abi',
        'buffer-7':'down-fp64','activation-fp32':'activation-fp32','final-fp32':'final-fp32',
        'residual-fp32':'residual-fp32','layer-output-fp32':'layer-output-fp32'}
    for target,source in mapping.items():
        raw=(a.reference/f'job-001-{source}.bin').read_bytes()
        if target=='input':raw=b'\xa5'*4032+raw
        write_hex('job-001-'+target+'.hex',raw)
    with (a.output/'jobs.txt').open('x',encoding='ascii',newline='\n') as f:f.write('NTRFFNMEM01 1 2048 8192\n')
    manifest=dict(format='NTRFFNMEM01',workload='NTRLLAMA32_OWNED_FFN_RTL01',layout=2,dim=2048,hidden=8192,samples=1,
        source_sample_id=1,source_token_id=0,post_sample=0,input_abi_requantized=False,tile_k=[32,32,32],
        model='unsloth/Llama-3.2-1B-Instruct-GGUF',revision=ref['revision'],weights_sha256=ref['model_sha256'],
        reference_manifest_sha256=digest(a.reference/'manifest.json'),packed_parent_manifest_sha256=digest(a.packed_parent/'manifest.json'),
        native_crosscheck_sha256=digest(a.native_review),native_crosscheck_run=native['run'],native_crosscheck_archive_sha256=native['archive_sha256'],
        weights=weights,expected_write_bytes=155648,expected_read_bytes=56754176,expected_down_values=2048,full_projection_rows=[8192,8192,2048],
        source_input_sha256=ref['samples'][0]['input_sha256'],source_residual_sha256=ref['samples'][0]['residual_sha256'],
        files={q.name:digest(q) for q in sorted(a.output.iterdir())},generator_sha256=digest(__file__),
        scope='Exact post-O reference ABI1 and original complete Q8_0 matrices. Native AMD/stdlib reference already qualified; new owned FFN RTL and live post-O integration are not qualified by fixture preparation.')
    with (a.output/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PREPARED_OWNED_FFN_FIXTURE_R1',path=str(a.output),manifest_sha256=digest(a.output/'manifest.json'),sample=1,token=0,files=len(manifest['files']))))

if __name__=='__main__':main()
