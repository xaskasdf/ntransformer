"""Relocate generated products of a private JSON XCI without changing parameters."""
import argparse
import hashlib
import json
from pathlib import Path


def relocate(path):
    before=path.read_bytes(); obj=json.loads(before)
    original=obj['ip_inst']['gen_directory']
    original_output=obj['ip_inst']['parameters']['runtime_parameters']['OUTPUTDIR']
    obj['ip_inst']['gen_directory']='generated'
    obj['ip_inst']['parameters']['runtime_parameters']['OUTPUTDIR']=[{'value':'generated'}]
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')
    after=json.loads(path.read_bytes())
    after['ip_inst']['gen_directory']=original
    after['ip_inst']['parameters']['runtime_parameters']['OUTPUTDIR']=original_output
    assert after==json.loads(before)
    path.with_suffix('.relocation.json').write_text(json.dumps(dict(
        before=hashlib.sha256(before).hexdigest(), after=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_directory=original, original_output=original_output,
        selected_directory='generated', component_and_model_parameters_unchanged=True), indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('xci',type=Path)
    relocate(ap.parse_args().xci)
