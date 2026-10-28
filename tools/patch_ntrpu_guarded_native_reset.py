"""Add a public board PERST stimulus task; preserve protected IP and models."""
import argparse,hashlib,json
from pathlib import Path
from patch_ntrpu_guarded_native import transform as cold_transform
from patch_ntrpu_pcie_multi import replace_once


def transform(original,tasks,driver):
    changed,info=cold_transform(original,tasks,driver)
    changed[-1]=replace_once(changed[-1],'ntrpu_pcie_guarded_native_harness ntrpu_bypass();',
        '''ntrpu_pcie_guarded_native_reset_harness ntrpu_bypass();
    // Simulation-only stimulus through the public board reset, never force IP.
    task ntrpu_set_perst;
        input level;
        begin sys_rst_n=level; end
    endtask''')
    return changed,info


def patch(example):
    root=Path(__file__).resolve().parents[1]
    names=['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v','sample_tests.vh','xilinx_dma_pcie_ep.sv','board.v']
    paths=[example/'imports'/n for n in names];original=[p.read_bytes() for p in paths]
    tasks=root/'fpga/tb/ntrpu_pcie_guarded_native_reset_tasks.vh';driver=root/'fpga/tb/ntrpu_pcie_frame512_driver.vh'
    changed,info=transform([s.decode() for s in original],tasks.read_text(),driver.read_text())
    manifest=dict(scope='Native PERST assertion/release and retained DDR operation. Runtime peer ACKs withheld; no endpoint coordinator, runtime reuse or physical PERST qualification.',
        pci_base=9<<32,axi_base=1<<32,offset=0xfffc0,tasks_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),files=[],**info)
    for p,before,after in zip(paths,original,changed):
        assert before.decode()!=after;p.write_bytes(after.encode())
        manifest['files'].append(dict(path=p.relative_to(example).as_posix(),before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (example/'NTRPU_BYPASS_PATCH.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('example',type=Path);patch(parser.parse_args().example)
