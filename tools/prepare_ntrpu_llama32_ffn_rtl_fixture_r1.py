"""Pack complete layer-0 gate/up/down after the native AMD-math cross-check.

Select native sample 2 (all 2048 inputs, H=8192), without reducing matrix
dimensions or modifying any Q8_0 block. This is a fixture, not a RTL PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path

from check_ntrpu_ffn_memory_fixture import decode
from ntrpu_matrix_layout import pack_matrix


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--native-review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    reference=json.loads((args.reference/'manifest.json').read_text())
    native=json.loads(args.native_review.read_text())
    if digest(args.reference/'manifest.json')!='ea56187ad501816c9eb03614f402813310124cb661e3ba01e1d27ec32fd5c310':
        raise ValueError('Full reference identity')
    if native['status']!='PASS_LLAMA32_FFN_HLS_MATH_NATIVE_R3' or not native['local_and_nfs_verified']:
        raise ValueError('Expected backed native AMD-math cross-check')
    for name,sha in reference['files'].items():
        if digest(args.reference/name)!=sha:raise ValueError('Reference file '+name)
    args.output.mkdir(parents=True,exist_ok=False)
    def write_hex(name,raw):
        with (args.output/name).open('x',encoding='ascii',newline='\n') as stream:
            for offset in range(0,len(raw),65536):stream.write(raw[offset:offset+65536].hex('\n')+'\n')
    weights=[]
    for ident,(rows,blocks) in enumerate(((8192,64),(8192,64),(2048,256)),1):
        raw=(args.reference/f'weight-{ident}-dense.bin').read_bytes()
        image,tiles=pack_matrix(raw,rows,blocks,32)
        if decode(image,rows,blocks,32)!=raw:raise ValueError('Independent full-matrix byte layout')
        write_hex(f'weight-{ident}.hex',image)
        if bytes.fromhex((args.output/f'weight-{ident}.hex').read_text())!=image:
            raise ValueError('Serialized simulator image')
        weights.append(dict(id=ident,rows=rows,k_blocks=blocks,tile_k=32,tiles=len(tiles),
                            source_tensor=reference['weights'][ident-1],
                            dense_sha256=hashlib.sha256(raw).hexdigest(),image_bytes=len(image),
                            image_sha256=hashlib.sha256(image).hexdigest()))
    mapping={'input-fp32':'input-fp32','input':'input-abi','buffer-4':'gate-fp64',
             'buffer-5':'up-fp64','buffer-6':'activation-abi','buffer-7':'down-fp64',
             'activation-fp32':'activation-fp32','final-fp32':'final-fp32'}
    for target,source in mapping.items():
        raw=(args.reference/f'job-002-{source}.bin').read_bytes()
        if target=='input':raw=b'\xa5'*4032+raw
        write_hex('job-001-'+target+'.hex',raw)
    (args.output/'jobs.txt').write_text('NTRFFNMEM01 1 2048 8192\n',encoding='ascii',newline='\n')
    manifest=dict(format='NTRFFNMEM01',workload='NTRLLAMA32_FULL_FFN_RTL01',layout=2,
                  dim=2048,hidden=8192,samples=1,source_sample_id=2,tile_k=[32,32,32],
                  model=reference['model'],revision=reference['revision'],weights_sha256=reference['model_sha256'],
                  reference_manifest_sha256=digest(args.reference/'manifest.json'),
                  native_crosscheck_sha256=digest(args.native_review),native_crosscheck_run=native['run'],
                  native_crosscheck_archive_sha256=native['archive_sha256'],weights=weights,
                  expected_write_bytes=155648,expected_read_bytes=56754176,
                  expected_down_values=2048,full_projection_rows=[8192,8192,2048],
                  scope='Complete layer0 FFN with original Unsloth Q8_0 blocks and native sample2 synthetic input. Packed and independently decoded on CPU; RTL not run yet.',
                  tools={name:digest(Path(__file__).with_name(name)) for name in
                         ('prepare_ntrpu_llama32_ffn_rtl_fixture_r1.py','ntrpu_matrix_layout.py','ntrpu_gemv_layout.py','check_ntrpu_ffn_memory_fixture.py')},
                  files={p.name:digest(p) for p in sorted(args.output.iterdir())})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PREPARED_LLAMA32_FULL_FFN_RTL_FIXTURE_R1',
                         manifest_sha256=digest(args.output/'manifest.json'),weights=weights,
                         native_crosscheck_run=native['run'])))


if __name__=='__main__':main()
