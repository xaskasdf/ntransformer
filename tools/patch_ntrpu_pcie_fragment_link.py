"""Replace only public-example traffic and sink in a new native XDMA copy."""
import argparse,hashlib,json
from pathlib import Path
from patch_ntrpu_pcie_bypass_bytes import transform as base_transform
from patch_ntrpu_pcie_multi import replace_once


def transform(original,tasks):
    changed,info=base_transform(original,tasks)
    changed[-1]=replace_once(changed[-1],'ntrpu_pcie_bypass_bytes_memory ntrpu_bypass();','ntrpu_pcie_fragment_link ntrpu_bypass();')
    return changed,info


def patch(example):
    names=['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v','sample_tests.vh','xilinx_dma_pcie_ep.sv','board.v']
    paths=[example/'imports'/n for n in names];original=[p.read_bytes() for p in paths]
    tasks=Path(__file__).resolve().parents[1]/'fpga/tb/ntrpu_pcie_fragment_link_tasks.vh'
    changed,info=transform([t.decode() for t in original],tasks.read_text())
    manifest=dict(scope='Native64-byte TLP mapping and unchanged NTRPUF01 pure decoder; lease/assembly/memory remain BFMs. No production ingress, DDR, CDC or upstream-fence qualification.',pci_base=9<<32,axi_base=1<<32,tasks_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),files=[],**info)
    for p,old,new in zip(paths,original,changed):
        assert old.decode()!=new;p.write_bytes(new.encode())
        manifest['files'].append(dict(path=p.relative_to(example).as_posix(),before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (example/'NTRPU_BYPASS_PATCH.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('example',type=Path);patch(p.parse_args().example)
