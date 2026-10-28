"""Attach production ingress RTL to a NEW copy of the public XDMA example."""
import argparse
import hashlib
import json
from pathlib import Path
from patch_ntrpu_pcie_bypass_bytes import transform as base_transform
from patch_ntrpu_pcie_multi import replace_once


def transform(original, tasks):
    changed, info = base_transform(original, tasks)
    changed[-1] = replace_once(changed[-1], 'ntrpu_pcie_bypass_bytes_memory ntrpu_bypass();',
                              'ntrpu_pcie_ingress_peer_harness ntrpu_bypass();')
    return changed, info


def patch(example):
    names = ['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v',
             'sample_tests.vh','xilinx_dma_pcie_ep.sv','board.v']
    paths = [example/'imports'/n for n in names]
    original = [p.read_bytes() for p in paths]
    tasks = Path(__file__).resolve().parents[1]/'fpga/tb/ntrpu_pcie_line_ingress_tasks.vh'
    changed, info = transform([t.decode() for t in original], tasks.read_text())
    manifest = dict(scope='Actual XDMA BAR bypass into ingress/bridge/peer receiver RTL;4 jobs,83 posted DWORD writes,3 destination writes,one source conflict. Destination AXI/isolation BFMs only;no DDR or P2P.',
                    pci_base=9<<32,axi_base=1<<32,job_offsets=[0,0x80000,0x80000,0xfffc0],
                    tasks_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),files=[],**info)
    for p,before,after in zip(paths,original,changed):
        assert before.decode()!=after
        p.write_bytes(after.encode())
        manifest['files'].append(dict(path=p.relative_to(example).as_posix(),
            before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (example/'NTRPU_BYPASS_PATCH.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('example',type=Path)
    patch(p.parse_args().example)
