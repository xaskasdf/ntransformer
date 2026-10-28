"""Extend a newly copied and normally patched AMD example with DMA tests."""
import argparse
import hashlib
import json
from pathlib import Path

from patch_ntrpu_pcie_ddr_ffn import patch


def patch_errors(example):
    patch(example)
    manifest_path=example/'NTRPU_FFN_PATCH.json'
    manifest=json.loads(manifest_path.read_text())
    baseline=json.loads(manifest_path.read_text())
    tx=example/'imports/usp_pci_exp_usrapp_tx.v'
    text=tx.read_text();assert text.count('endmodule')==1
    tasks=(Path(__file__).parents[1]/'fpga/tb/ntrpu_pcie_ddr_error_tasks.vh').read_text()
    tx.write_text(text.replace('endmodule',tasks+'\nendmodule'),encoding='utf-8',newline='\n')
    tests=example/'imports/sample_tests.vh';text=tests.read_text()
    assert text.count('    NTRPU_FFN_RUN;\n    $finish;')==1
    tests.write_text(text.replace('    NTRPU_FFN_RUN;\n    $finish;',
                    '    NTRPU_FFN_RUN;\n    NTRPU_DMA_ERRORS;\n    $finish;'),encoding='utf-8',newline='\n')
    manifest.update(dma_error_cases=7,baseline_patch=baseline)
    for entry in manifest['files']:
        entry['after_sha256']=hashlib.sha256((example/entry['path']).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',type=Path)
    args=parser.parse_args()
    patch_errors(args.example)
