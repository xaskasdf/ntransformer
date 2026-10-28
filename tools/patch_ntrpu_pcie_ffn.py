"""Patch only a NEW AMD example copy for PCIe/FFN/MSI-X simulation.

Keep IP RTL unchanged. Retain AMD headers and record every modified file hash.
"""
import argparse
import hashlib
import json
import pathlib
import re

from patch_ntrpu_pcie_logger import patch_text
from patch_ntrpu_pcie_concurrent_log import patch_concurrent
from patch_ntrpu_pcie_rx_log import patch_rx
from patch_ntrpu_pcie_multi import replace_once


def patch(example, bench=None):
    bench_manifest = None
    if bench is not None:
        bench_manifest = json.loads((bench / 'bench.json').read_text())
        if (bench_manifest['format'], bench_manifest['dim'], bench_manifest['hidden']) != ('NTRPCIEBRANDONBENCH01', 256, 720):
            raise ValueError('Unsupported specialized bench')
        for name, hashes in bench_manifest['files'].items():
            if pathlib.Path(name).name != name or hashlib.sha256((bench / name).read_bytes()).hexdigest() != hashes['after_sha256']:
                raise ValueError('Specialized bench identity: '+name)
    else:
        bench = pathlib.Path(__file__).parents[1] / 'fpga/tb'
    names = ['usp_pci_exp_usrapp_tx.v', 'usp_pci_exp_usrapp_rx.v',
             'usp_pci_exp_usrapp_com.v', 'sample_tests.vh',
             'xilinx_dma_pcie_ep.sv', 'board.v']
    paths = [example / 'imports' / name for name in names]
    original = [p.read_bytes() for p in paths]
    tx, rx, com, tests, ep, board = [b.decode() for b in original]
    start = tx.index('task TSK_TX_COMPLETION_DATA;')
    end = tx.index('endtask', start)
    completion = tx[start:end]
    completion, n = re.subn(r'DATA_STORE_2\[ram_ptr \+', 'DATA_STORE[lower_addr_ +', completion)
    if n != 192:
        raise ValueError(f'Unexpected continuation expressions: {n}')
    completion, n = re.subn(r'input\s+\[10:0\]\s+lower_addr_;', 'input [63:0] lower_addr_;', completion)
    if n != 1:
        raise ValueError('Unexpected completion address port')
    completion, n = re.subn(r'DATA_STORE\[lower_addr_\s*\+\s*([^\]]+)\]',
                           r'board.ntrpu_host.read_completion(lower_addr_, len_*4, \1)', completion)
    if n != 296 or 'DATA_STORE' in completion:
        raise ValueError(f'Unexpected completion byte count {n}')
    tx = tx[:start] + completion + tx[end:]
    rx, n = re.subn(r'reg\s+\[10:0\](\s+lower_addr;)', r'reg [63:0]\1', rx)
    if n != 1:
        raise ValueError('Unexpected receive address register')
    start = rx.index('task TSK_BUILD_CPLD_PKT;')
    end = rx.index('endtask', start)
    completion = rx[start:end]
    if completion.count('cq_data[10:2]') != 5:
        raise ValueError('Unexpected receive address expressions')
    completion = completion.replace('cq_data[10:2]', 'cq_data[63:2]')
    completion = replace_once(completion, "lower_addr = 11'h0", "lower_addr = 64'h0")
    completion = replace_once(completion, 'lower_addr[10:0]', 'lower_addr[63:0]')
    rx = rx[:start] + completion + rx[end:]
    tx, rx, com = patch_text(tx, rx, com)
    tx, com, logger_producers = patch_concurrent(tx, com)
    rx, com, rx_logger_calls = patch_rx(rx, com)
    # The vendor waiter joins a timeout poll that sleeps 500 clocks even after
    # completion. Poll every 50 clocks, retaining the same 15000-clock budget.
    start = tx.index('task TSK_WAIT_FOR_READ_DATA;')
    end = tx.index('endtask', start)
    waiter = replace_once(tx[start:end], 'j = 30;', 'j = 300;')
    waiter = replace_once(waiter, 'TSK_TX_CLK_EAT(500);', 'TSK_TX_CLK_EAT(50);')
    tx = tx[:start] + waiter + tx[end:]

    # Clone the vendor BAR tasks, adding an explicit BAR selector and read result.
    additions = []
    for kind in ('READ', 'WRITE'):
        start = tx.index(f'task TSK_XDMA_REG_{kind};')
        end = tx.index('endtask', start) + len('endtask')
        task = tx[start:end].replace(f'TSK_XDMA_REG_{kind}', f'NTRPU_BAR_{kind}')
        task = task.replace('board.RP.tx_usrapp.xdma_bar', 'bar_index')
        task = task.replace(f'task NTRPU_BAR_{kind};', f'task NTRPU_BAR_{kind};\n  input integer bar_index;')
        if kind == 'READ':
            task = task.replace('input [31:0] read_addr;', 'input [31:0] read_addr;\n  output [31:0] value;')
            final_end = task.rindex('\nend\n')
            task = task[:final_end] + '\n  value = board.RP.tx_usrapp.P_READ_DATA;\n' + task[final_end:]
        additions.append(task)
    tasks = (bench / 'ntrpu_pcie_ffn_tasks.vh').read_text()
    tx = replace_once(tx, 'endmodule', '\n'.join(additions) + '\n' + tasks + '\nendmodule')
    tests = tests[:tests.index('else if(testname')]+'''// NTRPU replaces only the test selection in this copied example.
else if(testname == "ntrpu_ffn_msix") begin
    NTRPU_FFN_RUN;
    $finish;
end
'''
    ep, n = re.subn(r'reg\s+\[C_NUM_USR_IRQ-1:0\]\s+usr_irq_req\s*=\s*0;',
                    'wire [C_NUM_USR_IRQ-1:0] usr_irq_req;', ep)
    if n != 1:
        raise ValueError('Unexpected IRQ request declaration')
    app = (bench / 'ntrpu_pcie_ffn_app.sv').read_text()
    ports = re.findall(r'(?:input|output) logic\s*(?:\[[^\]]+\])?\s+(\w+)\s*[,\n]', app[:app.index(');')])
    if len(ports) != 58:
        raise ValueError(f'Unexpected application ports {len(ports)}')
    mapping = {'host_clk': 'user_clk', 'pcie_resetn': 'user_resetn',
               'irq': 'usr_irq_req[0]', 'usr_irq_ack': 'usr_irq_ack[0]'}
    for port in ports:
        if port.startswith('s_axil_'):
            mapping[port] = port.replace('s_axil_', 'm_axil_')
        elif port.startswith('s_host_axi_'):
            mapping[port] = port.replace('s_host_axi_', 'm_axi_')
    start = ep.index('xdma_app #(')
    end = ep.index(');', start) + 2
    instance = 'ntrpu_pcie_ffn_app ntrpu_app (\n' + ',\n'.join(f'    .{p}({mapping[p]})' for p in ports) + '\n);'
    ep = ep[:start] + instance + ep[end:]
    board = replace_once(board, 'module board;', 'module board;\n    ntrpu_pcie_ffn_host ntrpu_host();')
    manifest = {'scope': 'Test driver, bounded memory, FFN application; no protected IP edits',
                'dim': 32, 'hidden': 32, 'computed_jobs': 1, 'invalid_jobs': 2,
                'completion_address_bits': 64, 'completion_poll_cycles': 50,
                'completion_timeout_cycles': 15000, 'files': []}
    manifest['separate_rq_cc_log_state'] = True
    manifest['completion_callback_rx_only'] = True
    manifest['separate_cq_rc_log_state'] = True
    manifest['rx_logger_calls'] = rx_logger_calls
    manifest['logger_producers'] = logger_producers
    if bench_manifest is not None:
        manifest.update(dim=256, hidden=720, specialized_bench=bench_manifest)
    for path, before, after in zip(paths, original, [tx, rx, com, tests, ep, board]):
        path.write_bytes(after.encode())
        manifest['files'].append({'path': str(path.relative_to(example)),
                                  'before_sha256': hashlib.sha256(before).hexdigest(),
                                  'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (example / 'NTRPU_FFN_PATCH.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', type=pathlib.Path)
    parser.add_argument('--bench', type=pathlib.Path)
    args = parser.parse_args()
    patch(args.example, args.bench)
