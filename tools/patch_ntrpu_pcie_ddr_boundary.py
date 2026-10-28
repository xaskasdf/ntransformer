"""Select transport-only boundary tasks in a fresh copy of the AMD example."""
import argparse
import hashlib
import json
from pathlib import Path
from patch_ntrpu_pcie_ddr_ffn import patch


def patch_boundary(example):
    patch(example)
    path=example/'NTRPU_FFN_PATCH.json'
    manifest=json.loads(path.read_text());baseline=json.loads(path.read_text())
    tx=example/'imports/usp_pci_exp_usrapp_tx.v';text=tx.read_text()
    assert text.count('endmodule')==1
    tasks=(Path(__file__).parents[1]/'fpga/tb/ntrpu_pcie_ddr_boundary_tasks.vh').read_text()
    tx.write_text(text.replace('endmodule',tasks+'\nendmodule'),encoding='utf-8',newline='\n')
    tests=example/'imports/sample_tests.vh';text=tests.read_text()
    assert text.count('    NTRPU_FFN_RUN;\n    $finish;')==1
    tests.write_text(text.replace('    NTRPU_FFN_RUN;\n    $finish;',
                    '    NTRPU_BOUNDARY_RUN;\n    $finish;'),encoding='utf-8',newline='\n')
    manifest.update(computed_jobs=0,invalid_jobs=0,dma_boundary_cases=11,baseline_patch=baseline)
    for entry in manifest['files']:
        entry['after_sha256']=hashlib.sha256((example/entry['path']).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('example',type=Path)
    patch_boundary(p.parse_args().example)
