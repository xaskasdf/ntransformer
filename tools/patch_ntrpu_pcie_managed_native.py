"""Connect the retained NTRD/two-MIG cold harness in a fresh public example."""
import argparse,hashlib,json
from pathlib import Path
from patch_ntrpu_pcie_fragment_native_host import transform as native_transform
from patch_ntrpu_pcie_multi import replace_once


def transform(original,tasks,driver):
    changed,info=native_transform(original,tasks,driver)
    changed[-1]=replace_once(changed[-1],'ntrpu_pcie_fragment_native_host_harness ntrpu_bypass();',
                            'ntrpu_pcie_managed_native_harness ntrpu_bypass();')
    return changed,info


def patch(example):
    root=Path(__file__).resolve().parents[1]
    names=['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v','sample_tests.vh','xilinx_dma_pcie_ep.sv','board.v']
    paths=[example/'imports'/n for n in names];original=[p.read_bytes() for p in paths]
    tasks=root/'fpga/tb/ntrpu_pcie_managed_native_tasks.vh';driver=root/'fpga/tb/ntrpu_pcie_frame512_driver.vh'
    changed,info=transform([s.decode() for s in original],tasks.read_text(),driver.read_text())
    manifest=dict(scope='Native cold XDMA BAR0 NTRD/BAR2 F01 into retained two-MIG core. External barriers and UI readback remain BFMs. No warm recovery, physical timing, commercial DIMM or inference qualification.',
        pci_base=9<<32,axi_base=1<<32,offset=0xfffc0,
        tasks_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),files=[],**info)
    for p,before,after in zip(paths,original,changed):
        assert before.decode()!=after;p.write_bytes(after.encode())
        manifest['files'].append(dict(path=p.relative_to(example).as_posix(),before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (example/'NTRPU_BYPASS_PATCH.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('example',type=Path);patch(parser.parse_args().example)
