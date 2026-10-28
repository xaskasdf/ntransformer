"""Attach the unchanged host control/data wrapper to a NEW public XDMA copy."""
import argparse,hashlib,json,re
from pathlib import Path
from patch_ntrpu_pcie_bypass_bytes import transform as data_transform
from patch_ntrpu_pcie_multi import replace_once


def transform(original,tasks):
    changed,info=data_transform(original,tasks)
    changed[-1]=replace_once(changed[-1],'ntrpu_pcie_bypass_bytes_memory ntrpu_bypass();',
                            'ntrpu_pcie_ingress_native_host_harness ntrpu_bypass();')
    ep=changed[-2];start=ep.index('xdma_app #(');end=ep.index(');',start)+2
    outputs={'awready','wready','bresp','bvalid','arready','rdata','rresp','rvalid'}
    seen=[]
    def disconnect(match):
        suffix=match.group(1);seen.append(suffix)
        return '.s_axil_'+suffix+'('+('' if suffix in outputs else "'0")+')'
    app=re.sub(r'\.s_axil_(\w+)\s*\([^)]*\)',disconnect,ep[start:end])
    assert len(seen)==17 and outputs.issubset(seen),seen
    changed[-2]=ep[:start]+app+ep[end:]
    return changed,dict(**info,disconnected_example_control_ports=seen)


def patch(example):
    names=['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v',
           'sample_tests.vh','xilinx_dma_pcie_ep.sv','board.v']
    paths=[example/'imports'/n for n in names];original=[p.read_bytes() for p in paths]
    task=Path(__file__).resolve().parents[1]/'fpga/tb/ntrpu_pcie_native_host_tasks.vh'
    changed,info=transform([s.decode() for s in original],task.read_text())
    manifest=dict(scope='Native PCIe BAR0 control plus BAR2 DWORD data, dualAMD CDC, unchanged guarded ingress and cold generic MIG. External isolation and directDDRreadback remain BFMs; no upstreamfence, physicalboard or warmreset qualification.',
                  pci_base=9<<32,axi_base=1<<32,job_offsets=[0,0x80000,0x80000,0xfffc0],
                  tasks_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),files=[],**info)
    for p,before,after in zip(paths,original,changed):
        assert before.decode()!=after;p.write_bytes(after.encode())
        manifest['files'].append(dict(path=p.relative_to(example).as_posix(),before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (example/'NTRPU_BYPASS_PATCH.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('example',type=Path)
    patch(parser.parse_args().example)
