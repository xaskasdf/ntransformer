"""Review the actual sequential accumulator and its sign-error mutant."""
import re


def review(root, evidence, preparation):
    normal=evidence['logs']['registered_sum_normal/simulate.log']
    mutant=evidence['logs']['registered_sum_mutant/simulate.log']
    core=(root/preparation['accumulator']).read_text()
    bad=(root/preparation['mutant']).read_text()
    expected=core.replace('module ntrpu_pcie_inventory_sum_accumulator (','module ntrpu_pcie_inventory_sum_accumulator_bad_sign (',1).replace('-$signed({1\'b0,saved_removed})','+$signed({1\'b0,saved_removed})',1)
    assert bad==expected and bad!=core
    tb=(root/preparation['testbench']).read_text()
    assert "reference_wide=$signed({11'b0,expected[20:0]})+$signed(added_span)-$signed({17'b0,removed_sum});" in tb
    expected_tb=tb.replace('module tb_ntrpu_pcie_registered_sum;','module tb_ntrpu_pcie_registered_sum_mutant;',1).replace('ntrpu_pcie_inventory_sum_accumulator dut','ntrpu_pcie_inventory_sum_accumulator_bad_sign dut',1).replace('REGISTERED_SUM_MODE mutant=0','REGISTERED_SUM_MODE mutant=1',1)
    assert (root/preparation['mutant_testbench']).read_text()==expected_tb
    for key in ['accumulator','testbench','mutant','mutant_testbench']:
        exact=evidence['invocation']['source']+'/'+preparation[key]
        assert any(line.startswith('INFO: [VRFC 10-2263] Analyzing SystemVerilog file "'+exact+'"') for line in evidence['console'].splitlines())
    for variant,top in [('normal','tb_ntrpu_pcie_registered_sum'),('mutant','tb_ntrpu_pcie_registered_sum_mutant')]:
        assert any(line.startswith('Running:') and 'xelab' in line and 'xil_defaultlib.'+top+' ' in line for line in evidence['console'].splitlines())
        assert evidence['console'].splitlines().count('NTRPU_INVENTORY_REGISTERED_SUM_REVIEWED '+variant)==1

    def check(n,m):
        if n.splitlines().count('REGISTERED_SUM_MODE mutant=0')!=1:raise ValueError('Missing normal mode')
        if n.splitlines().count('REGISTERED_SUM_PASS checks=20265 injected_unknown=62')!=1:raise ValueError('Incomplete unit checks')
        if re.search(r'^\s*(Fatal:|ERROR:|FAIL |Failure:)',n,re.M):raise ValueError('Unit failure')
        ranges=re.findall(r'^REGISTERED_SUM_RANGE negative=(\d+) protocol=(\d+) overflow=(\d+)$',n,re.M)
        if len(ranges)!=1 or not all(int(v)>0 for v in ranges[0]):raise ValueError('Missing signed/truncation range coverage')
        if not re.search(r'\$finish called at time .*tb_ntrpu_pcie_registered_sum.sv',n):raise ValueError('Missing normal finish')
        if m.splitlines().count('REGISTERED_SUM_MODE mutant=1')!=1:raise ValueError('Missing actual mutant mode')
        if not re.search(r'^Fatal: REGISTERED_SUM_MISMATCH check=2 actual=000027 expected=000019\s*$',m,re.M):raise ValueError('Sign mutant not rejected at directed subtraction')
        if 'REGISTERED_SUM_PASS' in m:raise ValueError('False mutant pass')
        return [int(v) for v in ranges[0]]

    ranges=check(normal,mutant)
    controls=[]
    for name,n,m in [
        ('missing_pass',normal.replace('REGISTERED_SUM_PASS','REMOVED',1),mutant),
        ('missing_checks',normal.replace('checks=20265','checks=20264',1),mutant),
        ('missing_unknowns',normal.replace('injected_unknown=62','injected_unknown=0',1),mutant),
        ('missing_range',normal.replace('REGISTERED_SUM_RANGE','REMOVED',1),mutant),
        ('missing_mutant_failure',normal,mutant.replace('Fatal: REGISTERED_SUM_MISMATCH','REMOVED',1)),
        ('false_mutant_pass',normal,mutant+'\nREGISTERED_SUM_PASS\n')]:
        assert n!=normal or m!=mutant
        try:check(n,m)
        except ValueError:controls.append(dict(name=name,rejected=True))
        else:raise AssertionError('Accepted '+name)
    return dict(checks=20265,unknown_injections=62,negative_results=ranges[0],above_protocol_limit=ranges[1],
        above_low_storage_range=ranges[2],controls=controls,actual_sign_mutant_rejected=True,
        actual_sequential_dut=True,reference_width_bits=32,pre_edge_stability_checked=True,
        unit_equivalence_scope='Finite sequential recurrence including enable/reset/XZ, not universal formal proof.',
        full_rtl_integration_still_required=True)
