"""Apply the availability fix to the sealed native-join source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected source anchor: ' + old[:80])
    return text.replace(old, new, 1)


def prepare(source, output, helpers):
    if output.exists():
        raise ValueError('New prepared directory required')
    top = source / 'ntrpu_axi_bridge_hbm_native_join.sv'
    if sha(top) != 'b11ea3496b171e3bd64319f794acd846990c96cb86b3ffc9cdcb9333e8d2ed35':
        raise ValueError('Wrong native-join baseline')
    text = top.read_text()
    declarations = re.findall(r'^    wire [^;\n]*\b(?:m_|h_)\w+;\n', text, re.M)
    if len(declarations) != 48:
        raise ValueError('Expected 48 native interface declarations')
    for declaration in declarations:
        text = replace_once(text, declaration, '')
    text = replace_once(text, '  ntrpu_bridge ntrpu_bridge_i (',
                        ''.join(declarations) + '\n  ntrpu_bridge ntrpu_bridge_i (')
    text = replace_once(text, 'output wire fabric_clock_locked,',
                        'output wire fabric_clock_locked,\n  output wire availability_armed,availability_fault,')
    text = replace_once(text, 'wire [1:0] init_complete_async=hbm_init_complete & {2{fabric_clock_locked}};',
        '''wire [1:0] init_complete_async;
    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_native_availability availability(
        .mgmt_clk(mgmt_clk),.cold_rst_n(management_cold_rst_n),
        .fabric_locked_async(fabric_clock_locked),.apb_complete_mgmt(hbm_init_complete),
        .ready_mgmt(init_complete_async),.armed(availability_armed),.fault(availability_fault));''')
    # Adapt the existing queue-retention bank; do not replace it with a toy queue.
    bench = (helpers / 'tb_axi_bridge_hbm_clocked.sv').read_text()
    bench = bench.replace('tb_axi_bridge_hbm_clocked', 'tb_hbm_native_availability')
    bench = bench.replace('init_complete', 'apb_complete')
    bench = bench.replace('.apb_complete_async(apb_complete)', '.init_complete_async(qualified_ready)')
    bench = replace_once(bench, 'wire startup_started,startup_fault;', '''wire startup_started,startup_fault;
    reg fabric_locked=0;
    wire [1:0] qualified_ready;
    wire availability_armed,availability_fault;
    ntrpu_hbm_native_availability availability(
        .mgmt_clk(mgmt_clk),.cold_rst_n(m_reset),.fabric_locked_async(fabric_locked),
        .apb_complete_mgmt(apb_complete),.ready_mgmt(qualified_ready),
        .armed(availability_armed),.fault(availability_fault));''')
    bench = bench.replace('apb_complete=3;', 'apb_complete=3;fabric_locked=1;')
    bench = replace_once(bench, 'if(test_id==10)begin\n', '''if(test_id==17)begin
            fabric_locked=0;#400;
            require(!effective_permit&&!availability_armed&&!availability_fault,"unlocked startup cannot arm");
            fabric_locked=1;wait(effective_permit);
        end else if(test_id==18)begin
            apb_complete=1;#400;
            require(!effective_permit&&!availability_armed&&!availability_fault,"both stacks required");
            apb_complete=3;wait(effective_permit);
        end else if(test_id==10)begin
''')
    bench = replace_once(bench, '            default:begin end\n', '''            13:fabric_locked=0;
            14:begin wait(r_clk==1);#0.01;r_run=0;fabric_locked=0;#70;fabric_locked=1;end
            15:begin wait(mem_clk==0);#0.01;mem_run=0;r_run=0;fabric_locked=0;#70;fabric_locked=1;end
            16:begin wait(r_clk==0);#0.01;r_run=0;apb_complete=2;#70;apb_complete=3;end
            19:begin fabric_locked=0;#30;fabric_locked=1;end
            default:begin end
''')
    bench = bench.replace('test_id!=0 && test_id!=10', 'test_id!=0 && test_id!=10 && test_id!=17 && test_id!=18')
    bench = replace_once(bench, '#1;require(!effective_permit,"new admission closed");retained();',
        '''#1;require(!effective_permit,"new admission closed");retained();
            if(test_id>=13 && test_id!=17 && test_id!=18)
                require(availability_fault && qualified_ready==0,"sampled availability loss is sticky");''')
    bench = bench.replace('HC PASS case=', 'NA PASS case=')
    output.mkdir(parents=True)
    for p in source.iterdir():
        if p.is_file():
            shutil.copy2(p, output / p.name)
    (output / top.name).write_text(text)
    (output / 'tb_hbm_native_availability.sv').write_text(bench)
    for name in ('ntrpu_hbm_native_availability.sv', 'ntrpu_axi_bridge_guard_v3.sv',
                 'prepare_hbm_native_fix.py', 'run_hbm_native_join.py', 'run_hbm_native_availability.py'):
        shutil.copy2(helpers / name, output / name)
    tcl = (output / 'elaborate_hbm_native_join.tcl').read_text()
    tcl = replace_once(tcl, 'ntrpu_hbm_native_rw ntrpu_axi_bridge_hbm_native_join}',
                       'ntrpu_hbm_native_rw ntrpu_hbm_native_availability ntrpu_axi_bridge_hbm_native_join}')
    # Clocking Wizard already defines its 100 MHz root. Keep its vendor XDC.
    tcl = replace_once(tcl, ' fabric_ref_clk_in fabric_refclk', '')
    (output / 'elaborate_hbm_native_join.tcl').write_text(tcl)
    result = dict(baseline_sha256=sha(top), top_sha256=sha(output / top.name),
        availability_sha256=sha(output / 'ntrpu_hbm_native_availability.sv'),
        benchmark_donor_sha256=sha(helpers / 'tb_axi_bridge_hbm_clocked.sv'),
        benchmark_sha256=sha(output / 'tb_hbm_native_availability.sv'),
        native_ip_configuration_changed=False, cold_reset_network_changed=False)
    (output / 'FIX_PREPARED.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path); p.add_argument('output', type=Path)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.output, Path(__file__).resolve().parent), indent=2))
