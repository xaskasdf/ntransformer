"""Verify restored physical CDC evidence, provenance and selection boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from review_hbm_cdc_physical import manifest, review, sha, require, table, SYNTH_SHA
from check_hbm_cdc_physical import check

FUNCTIONAL_SHA='70f9e19d407e8186033142cb28f1d19a87f3e6dde838923c63fe8c8092eeaecf'
CDC_SHA='87a2deca7c6d3c7bedeb4bab640784d77b052b6e3d08958207657917ea0d9f9f'
FIFO_SHA='be809fe998385e69a52eb2fa8687b5b50e52433a5b1a42694331efc07f1f0634'


def bundle(root):
    count=manifest(root)
    selection=json.loads((root/'SELECTION.json').read_text())
    paths={key:(root/value).resolve() for key,value in selection['phases'].items()}
    require(set(paths)=={'route','reopen'} and all(path.is_relative_to(root.resolve()) for path in paths.values()),'Unexpected selected phases')
    checks={key:review(path) for key,path in paths.items()}
    require(not checks['route']['reopened_readonly'] and checks['reopen']['reopened_readonly'],'Wrong phase roles')
    require(sha(paths['route']/'output/routed.dcp')==sha(paths['reopen']/'source/donor.dcp')==selection['routed_sha256'],'Reopened checkpoint is not the selected route')
    for filename,key in [('paths.tsv','to'),('interstage.tsv','to')]:
        a=sorted(table(paths['route']/'output/audit'/filename),key=lambda r:json.dumps(r,sort_keys=True))
        b=sorted(table(paths['reopen']/'output/audit'/filename),key=lambda r:json.dumps(r,sort_keys=True))
        require(a==b,'Path evidence changed after reopen: '+filename)
    for path in paths.values():
        result=json.loads((path/'RESULT.json').read_text())
        for name,digest in result['sources'].items():
            require(sha(root/'source'/name)==digest,'Published source differs from selected tool run: '+name)
    checks['negative']=check(paths['reopen'])
    require(checks['negative']['rejected']==21,'Missing negative controls')
    attempts=selection['unselected_attempts']
    require(len(attempts)==1,'Missing failed audit history')
    failed=(root/attempts[0]).resolve()
    require(failed.is_relative_to(root.resolve()),'Attempt escapes bundle')
    manifest(failed)
    require(json.loads((failed/'RESULT.json').read_text())['status']=='FAILED','Failed audit promoted')
    parent=root/'parent/20261011-hbm-cdc.zip'
    require(sha(parent)==FUNCTIONAL_SHA,'Functional parent archive mismatch')
    with zipfile.ZipFile(parent) as archive:
        parent_selection=json.loads(archive.read('SELECTION.json'))
        synth=parent_selection['phases']['synthesis']+'/output/synth.dcp'
        require(hashlib.sha256(archive.read(synth)).hexdigest()==SYNTH_SHA,'Synthesis donor is not from the functional parent')
        require(hashlib.sha256(archive.read('source/ntrpu_axi_bridge_hbm_cdc.sv')).hexdigest()==CDC_SHA,'CDC RTL parent mismatch')
    require(sha(root/'source/ntrpu_axi_bridge_hbm_cdc.sv')==CDC_SHA and sha(root/'source/ntrpu_bridge_async_fifo.sv')==FIFO_SHA,'Frozen RTL changed')
    processes=json.loads((root/'PROCESS_COMPLETION.json').read_text())
    require(len(processes['phases'])==3 and not any(p['active'] for phase in processes['phases'] for p in phase['processes']),'Private work remains active')
    return dict(status='HBM_CDC_PHYSICAL_BUNDLE_REVIEWED',files=count,checks=checks,
                cdc_data_paths_meet_bounds_in_ooc_context=True,parent_clocks_implemented=False,
                native_hbm_pcie_simulation=False,integrated_cdc_qualified=False,board_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bundle',type=Path)
    print(json.dumps(bundle(parser.parse_args().bundle),indent=2))
