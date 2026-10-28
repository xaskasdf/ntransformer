"""Freeze the reviewed commit-stage shared reader into a native HBM experiment."""
import argparse
import json
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save
from prepare_ntrpu_peer_commit import once

ROOT = Path(__file__).resolve().parents[1]


def prepare(selected, output):
    from check_ntrpu_peer_commit import lineage
    lineage(selected / 'repo')
    review = json.loads((selected / 'REVIEW.json').read_text())
    result = json.loads((selected / 'output/result.json').read_text())
    assert review['status'] == 'SHARED_ROUTED_INTERNAL_TIMING_PASS'
    assert review['result_sha256'] == sha(selected / 'output/result.json')
    assert result['sources'] == result['sources_after']
    assert all(sha(selected / 'repo' / n) == h for n, h in result['sources'].items())
    assert not output.exists() and not output.resolve().is_relative_to(ROOT)
    shutil.copytree(selected / 'repo', output)
    for name in ('prepare_ntrpu_peer_shared_native.py', 'run_ntrpu_peer_shared_native.py',
                 'check_ntrpu_peer_shared_native.py', 'ntrpu_peer_shared_native.tcl',
                 'ntrpu_peer_shared_native_run.tcl'):
        shutil.copyfile(ROOT / 'tools' / name, output / 'tools' / name)
    body = 'fpga/tb/ntrpu_peer_shared_native_body.svh'
    shutil.copyfile(ROOT / body, output / body)
    text = (output / 'fpga/tb/tb_ntrpu_peer_shared.sv').read_text()
    text = once(text, 'module tb_ntrpu_peer_shared;', 'module tb_ntrpu_peer_shared_native;\n    parameter integer NATIVE=0;')
    text = once(text, "wire [11:0] h_rid=0;", 'wire [11:0] h_rid;')
    text = once(text, ".init_complete_async(2'b11)", '.init_complete_async(init_complete_async)')
    dut_start = text.index('    ntrpu_peer_read_shared dut(')
    dut_end = text.index('    integer epoch=', dut_start)
    dut = text[dut_start:dut_end]
    text = text[:dut_start] + text[dut_end:]
    text = once(text, '(address^(address>>8)^(address>>16)^8\'h5b)',
                "(address^(address>>8)^(address>>16)^(address>>24)^(address>>32)^8'h5b)")
    start = text.index('    for(genvar l=0;l<2;l++)begin: responders')
    end = text.index('    task automatic check', start)
    writer_source = (output / 'fpga/tb/tb_ntrpu_read_return.sv').read_text()
    writer = writer_source[writer_source.index('    reg [1:0] wcmd_valid'):writer_source.index('    reg [1:0] ar_open')]
    binding = (output / 'fixtures/ntrpu_read_return_binding.svh').read_text()
    for name in ('araddr', 'arburst', 'arid', 'arlen', 'arsize', 'arvalid', 'rready'):
        binding = binding.replace('h_' + name, 'bus_' + name)
    (output / 'fixtures/ntrpu_shared_native_binding.svh').write_text(binding)
    extra = '''    wire [1:0] init_complete_async={apb_complete_1,apb_complete_0};
    wire [1:0] raw_arready,raw_rvalid,raw_rlast;
    wire [511:0] raw_rdata;
    wire [11:0] raw_rid;
    wire [3:0] raw_rresp;
    wire [1:0] stray=0;
    reg direct_mode=1;
    reg [1:0] direct_valid=0,direct_ready=0;
    reg [67:0] direct_address=0;
    wire [1:0] bus_arvalid=direct_mode?direct_valid:h_arvalid;
    wire [67:0] bus_araddr=direct_mode?direct_address:h_araddr;
    wire [7:0] bus_arlen=direct_mode?8'hff:h_arlen;
    wire [11:0] bus_arid=direct_mode?12'b0:h_arid;
    wire [5:0] bus_arsize=direct_mode?{3'd5,3'd5}:h_arsize;
    wire [3:0] bus_arburst=direct_mode?4'b0101:h_arburst;
    wire [1:0] bus_rready=direct_mode?direct_ready:h_rready;
    assign h_arready=direct_mode?0:(raw_arready&ar_open);
    assign h_rvalid=direct_mode?0:(raw_rvalid&r_open);
    assign h_rdata=raw_rdata;
    assign h_rid=raw_rid;
    assign h_rresp=raw_rresp;
    assign h_rlast=raw_rlast;
'''
    text = text[:start] + extra + writer + '    `include "ntrpu_shared_native_binding.svh"\n' + dut + text[end:]
    # This experiment initializes once; no reset silently reinitializes HBM.
    start = text.index('    task automatic boot(')
    end = text.index('    task automatic submit', start)
    text = text[:start] + text[end:]
    start = text.index('    request_t a,b,c,d,v;')
    text = text[:start] + (ROOT / body).read_text() + '\nendmodule\n'
    (output / 'fpga/tb/tb_ntrpu_peer_shared_native.sv').write_text(text)
    # Keep the independent ownership/event auditor, adapting only bench scope and pattern.
    audit = (output / 'tools/check_ntrpu_peer_shared.py').read_text()
    audit = once(audit, 'cases = 6 if full else 15', 'cases = 4 if full else 15')
    audit = once(audit, 'set(range(1, cases + 1)), \'epoch coverage\'', "{1}, 'epoch coverage'")
    audit = once(audit, '(a ^ (a >> 8) ^ (a >> 16) ^ 0x5b)',
                 '(a ^ (a >> 8) ^ (a >> 16) ^ (a >> 24) ^ (a >> 32) ^ 0x5b)')
    audit = once(audit, "stats['successful_retire'] == 27 and stats['successful_payload_bytes'] == 46848",
                 "stats['successful_retire'] == 16 and stats['successful_payload_bytes'] == 35200")
    (output / 'tools/check_ntrpu_shared_native_ledger.py').write_text(audit)
    rtl = {n: h for n, h in result['sources'].items()
           if n.endswith('.sv') and n.startswith(('fpga/rtl/', 'fpga/experimental/'))}
    native = json.loads((output / 'NATIVE_PROFILE.json').read_text())
    save(output / 'SHARED_NATIVE_PROFILE.json', dict(
        selected_result_sha256=sha(selected / 'output/result.json'),
        selected_checkpoint_sha256=sha(selected / 'output/suite/routed.dcp'),
        selected_rtl=rtl, selected_read_configs=native['selected_read_configs'],
        native_binding_sha256=sha(output / 'fixtures/native.sv'),
        native_config_sha256='d74a3e492748cf8a87100205e04bafe232fcdba09ae7e4a32105cdd02090539e',
        region_bases=[0, 8589934592], preload_bytes=[16384, 8192],
        scope='One initialization, two unique regions, direct AXI locality and shared commit-stage ownership',
        integrated_timing_qualified=False, pcie_integrated=False, board_qualified=False))
    assert all(sha(output / n) == h for n, h in rtl.items())
    shutil.copyfile(selected / 'REVIEW.json', output / 'COMMIT_PHYSICAL_REVIEW.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('selected', type=Path); ap.add_argument('output', type=Path)
    a = ap.parse_args(); prepare(a.selected.resolve(), a.output.resolve())
