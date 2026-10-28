"""Instrument only a new generated PIPE example; retain all AMD IP unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from patch_ntrpu_pcie_logger import patch_text
from patch_ntrpu_pcie_concurrent_log import patch_concurrent
from patch_ntrpu_pcie_rx_log import patch_rx
from patch_ntrpu_pcie_multi import replace_once


def transform(original, tasks):
    tx, rx, com, tests, ep, board = original
    tx, rx, com = patch_text(tx, rx, com)
    tx, com, producers = patch_concurrent(tx, com)
    rx, com, rx_calls = patch_rx(rx, com)
    tx = replace_once(tx, "BAR_INIT_P_MEM64_HI_START =  32'h0000_0001;",
                      "BAR_INIT_P_MEM64_HI_START =  32'h0000_0009;")
    start = tx.index('task TSK_WAIT_FOR_READ_DATA;')
    end = tx.index('endtask', start)
    waiter = replace_once(tx[start:end], 'j = 30;', 'j = 300;')
    waiter = replace_once(waiter, 'TSK_TX_CLK_EAT(500);', 'TSK_TX_CLK_EAT(50);')
    tx = tx[:start]+waiter+tx[end:]
    tx = replace_once(tx, 'endmodule', tasks+'\nendmodule')
    tests = tests[:tests.index('else if(testname')]+'''// Isolated, explicit BAR traffic.
else if(testname == "ntrpu_bypass") begin
    NTRPU_BYPASS_RUN;
    $finish;
end
'''
    start = ep.index('xdma_app #(')
    end = ep.index(');', start)+2
    app = ep[start:end]
    outputs = {'awready', 'wready', 'bid', 'bresp', 'bvalid', 'arready', 'rid', 'rdata', 'rresp', 'rlast', 'rvalid'}
    seen = []
    def disconnect(match):
        suffix = match.group(1)
        seen.append(suffix)
        # Original example accepts only 19 address bits. Disconnect its bypass
        # slave and attach a bounded full-address test sink, leaving DMA/AXIL.
        return '.s_axib_'+suffix+'('+('' if suffix in outputs else "'0")+')'
    app = re.sub(r'\.s_axib_(\w+)\s*\([^)]*\)', disconnect, app)
    assert len(seen)==29 and outputs.issubset(seen), seen
    ep = ep[:start]+app+ep[end:]
    # The generated wrapper uses RLAST without declaring it. XSIM rejects
    # cross-module assignments to implicit nets; make only this test-copy net explicit.
    assert not re.search(r'\bwire\s+m_axib_rlast\s*;', ep)
    ep = replace_once(ep, 'xdma_app #(', 'wire m_axib_rlast;\nxdma_app #(')
    board = replace_once(board, 'module board;', 'module board;\n    ntrpu_pcie_bypass_memory ntrpu_bypass();')
    return [tx,rx,com,tests,ep,board], dict(disconnected_example_bypass_ports=seen,
        logger_producers=producers,rx_logger_calls=rx_calls)


def patch(example):
    names = ['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v',
             'sample_tests.vh','xilinx_dma_pcie_ep.sv','board.v']
    paths = [example/'imports'/n for n in names]
    original = [p.read_bytes() for p in paths]
    task_path = Path(__file__).resolve().parents[1]/'fpga/tb/ntrpu_pcie_bypass_tasks.vh'
    changed, info = transform([t.decode() for t in original],task_path.read_text())
    manifest = dict(scope='Test-only 3 DWORD writes/reads through actual XDMA BAR bypass; no DDR/P2P',
                    pci_base=9<<32, axi_base=1<<32, offsets=[0,0x80000,0xfffc0],
                    tasks_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(), files=[], **info)
    for p,before,after in zip(paths,original,changed):
        assert before.decode()!=after
        p.write_bytes(after.encode())
        manifest['files'].append(dict(path=p.relative_to(example).as_posix(),
            before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (example/'NTRPU_BYPASS_PATCH.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',type=Path)
    patch(parser.parse_args().example)
