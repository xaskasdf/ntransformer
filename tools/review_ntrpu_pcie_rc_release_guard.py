"""Review actual-producer guard tests, their source mutant and corruption checks."""
import hashlib
import re
from review_ntrpu_pcie_rc_lane_banks import compilation

def review(root,evidence,preparation):
    normal_top='tb_ntrpu_pcie_rc_release_tag'
    mutant_top=normal_top+'_mutant'
    for top in [normal_top,mutant_top]:
        compilation(evidence,top,'fpga/tb/'+top+'.sv')
    normal_source=(root/('fpga/tb/'+normal_top+'.sv')).read_text()
    mutant_source=(root/('fpga/tb/'+mutant_top+'.sv')).read_text()
    expected=normal_source.replace('module '+normal_top+';', 'module '+mutant_top+';',1).replace(
        'ntrpu_pcie_inventory_scheduler dut (','ntrpu_pcie_inventory_scheduler_bad_release_tag dut (',1).replace(
        'mutant=0 canonical_state_injected=1','mutant=1 canonical_state_injected=1',1)
    assert mutant_source==expected
    candidate=(root/preparation['producer']).read_text()
    bad=(root/preparation['mutant']).read_text()
    assert hashlib.sha256((root/preparation['mutant']).read_bytes()).hexdigest()==preparation['mutant_sha256']
    assert bad==candidate.replace('module ntrpu_pcie_inventory_scheduler (',
        'module ntrpu_pcie_inventory_scheduler_bad_release_tag (',1).replace(
        'rc_release_tag[i]=rc_data[i*128+64+:8];',"rc_release_tag[i]=rc_data[i*128+64+:8] ^ 8'h01;",1)
    for relative,module in [(preparation['producer'],'ntrpu_pcie_inventory_scheduler'),
                            (preparation['mutant'],'ntrpu_pcie_inventory_scheduler_bad_release_tag')]:
        exact=evidence['invocation']['source']+'/'+relative
        assert any(s.startswith('INFO: [VRFC 10-2263] Analyzing SystemVerilog file "'+exact+'"') for s in evidence['console'].splitlines())
        assert any(s.startswith('INFO: [VRFC 10-311] analyzing module '+module) for s in evidence['console'].splitlines())
    for mode in ['normal','mutant']:
        assert evidence['console'].splitlines().count('NTRPU_INVENTORY_RC_RELEASE_GUARD_REVIEWED '+mode)==1
    normal=evidence['logs']['release_guard_normal/simulate.log']
    mutant=evidence['logs']['release_guard_mutant/simulate.log']

    def check(good,bad):
        if good.splitlines().count('RC_RELEASE_GUARD_MODE mutant=0 canonical_state_injected=1')!=1:
            raise ValueError('Missing normal guard mode')
        hits=re.findall(r'^RC_RELEASE_GUARD_PASS checks=(\d+) unknown_checks=(\d+) valid_checks=(\d+) masked_differences=(\d+)$',good,re.M)
        if len(hits)!=1:raise ValueError('Missing unique guard result')
        counts=list(map(int,hits[0]))
        if counts[:2]!=[12832,7680] or min(counts[2:])<=0:raise ValueError('Incomplete guard coverage')
        if re.search(r'^\s*(Fatal:|Failure:|ERROR:|FAIL )',good,re.M):raise ValueError('Normal guard failure')
        if not re.search(r'\$finish called at time .*'+normal_top+r'.sv',good):raise ValueError('Missing actual top finish')
        if bad.splitlines().count('RC_RELEASE_GUARD_MODE mutant=1 canonical_state_injected=1')!=1:
            raise ValueError('Missing actual mutant mode')
        if 'RC_RELEASE_GUARD_PASS' in bad:raise ValueError('Mutant incorrectly passed')
        if not re.search(r'^Fatal: RC_RELEASE_GUARD_VALID_TAG_MISMATCH check=0 lane=0 valid=1 actual=01 expected=00$',bad,re.M):
            raise ValueError('Actual released-tag mutation was not rejected at first case')
        if mutant_top+'.sv' not in bad:raise ValueError('Missing mutant source location')
        return dict(zip(['checks','unknown_checks','valid_checks','masked_differences'],counts))
    result=check(normal,mutant)
    controls=[]
    for label,good,bad in [
        ('missing_pass',normal.replace('RC_RELEASE_GUARD_PASS','REMOVED',1),mutant),
        ('wrong_checks',normal.replace('checks=12832','checks=12831',1),mutant),
        ('missing_unknowns',normal.replace('unknown_checks=7680','unknown_checks=0',1),mutant),
        ('missing_mutant_failure',normal,mutant.replace('Fatal: RC_RELEASE_GUARD_VALID_TAG_MISMATCH','REMOVED',1)),
        ('wrong_mutant_mode',normal,mutant.replace('mutant=1','mutant=0',1)),
        ('false_mutant_pass',normal,mutant+'\nRC_RELEASE_GUARD_PASS\n')]:
        assert good!=normal or bad!=mutant
        try:check(good,bad)
        except ValueError as exc:controls.append(dict(name=label,rejected=True,reason=str(exc)))
        else:raise AssertionError('Accepted guard corruption: '+label)
    return dict(**result,controls=controls,actual_bad_release_tag_mutant_rejected=True,
        exact_mutant_source_verified=True,actual_producer_guard_and_consumers_checked=True,
        canonical_state_injected_only_in_isolated_unit=True,
        sequential_ownership_reference=True,old_masked_tags_used_for_admission_and_bitmap_reference=True,
        full_rtl_integration_still_required=True,qualified_physical_design=False)
