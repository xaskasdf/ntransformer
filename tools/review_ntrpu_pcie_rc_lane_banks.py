"""Review extra R12 banks; inherited complete monitor reviews remain mandatory."""
import hashlib
import json
import re
from check_ntrpu_pcie_inventory_prefix_integration import review as review_integration
from generate_ntrpu_pcie_rc_lane_integration import generate


def compilation(evidence, top, relative):
    console = evidence['console'].splitlines()
    exact = evidence['invocation']['source'] + '/' + relative
    analysis = 'INFO: [VRFC 10-2263] Analyzing SystemVerilog file "' + exact + '" into library xil_defaultlib'
    if not any(line.startswith(analysis) for line in console):
        raise ValueError('Missing exact bank source analysis: ' + relative)
    if not any(line.startswith('INFO: [VRFC 10-311] analyzing module ' + top) for line in console):
        raise ValueError('Missing actual bank module')
    if not any(line.startswith('Running:') and 'xelab' in line and 'xil_defaultlib.' + top + ' ' in line for line in console):
        raise ValueError('Missing actual bank elaboration')


def sum_bank(root, evidence):
    top = 'tb_ntrpu_pcie_rc_lane_span_sum'
    source = 'fpga/tb/' + top + '.sv'
    compilation(evidence, top, source)
    assert 'actual=dut.rc_lane_span_sum(values);' in (root / source).read_text()
    log = evidence['logs']['lane_sum/simulate.log']

    def check(text):
        if text.splitlines().count('RC_LANE_SPAN_SUM_PASS checks=70464 unknown_checks=832') != 1:
            raise ValueError('Incomplete lane sum bank')
        if not re.search(r'\$finish called at time .*' + top + r'.sv', text):
            raise ValueError('Missing actual sum top finish')
        if re.search(r'^\s*(Fatal:|Failure:|ERROR:|FAIL )', text, re.M):
            raise ValueError('Lane sum failure')

    check(log)
    controls = []
    for name, changed in [('missing_pass', log.replace('RC_LANE_SPAN_SUM_PASS', 'REMOVED', 1)),
                          ('missing_cases', log.replace('checks=70464', 'checks=70463', 1)),
                          ('missing_unknowns', log.replace('unknown_checks=832', 'unknown_checks=0', 1)),
                          ('missing_finish', log.replace('$finish called', 'REMOVED', 1))]:
        assert changed != log
        try:
            check(changed)
        except ValueError:
            controls.append(dict(name=name, rejected=True))
        else:
            raise AssertionError('Lane sum corruption admitted: ' + name)
    return dict(checks=70464, unknown_checks=832, controls=controls,
                actual_dut_function_called=True, exact_top_analyzed=True)


def integration_bank(root, evidence, *, binary, internal, baseline_log=None):
    mode = 'binary' if binary else 'rtl'
    relative = 'fpga/tb/data/pcie_rc_lane_integration_' + mode + '.txt'
    vectors = (root / relative).read_text()
    manifest = json.loads((root / relative).with_suffix('.json').read_text())
    generated, binary_cases, binary_rows = generate()
    expected = generated.rows[:binary_rows] if binary else generated.rows
    scenarios = generated.scenarios[:binary_cases] if binary else generated.scenarios
    assert '\n'.join(expected) + '\n' == vectors and scenarios == manifest['scenarios']
    assert hashlib.sha256(vectors.encode()).hexdigest() == manifest['vectors_sha256']
    assert hashlib.sha256((root / 'tools/generate_ntrpu_pcie_rc_lane_integration.py').read_bytes()).hexdigest() == manifest['generator_sha256']
    assert (len(expected), len(scenarios)) == ((632, 60) if binary else (984, 92))
    unknown_rows = sum(any(c in row for c in 'xz') for row in expected)
    assert unknown_rows == (0 if binary else 32)
    top = 'tb_ntrpu_pcie_inventory_lane_integration'
    tb = 'fpga/tb/' + top + ('_binary' if binary else '') + '.sv'
    compilation(evidence, top, tb)
    vector_arg = 'VECTORS=' + evidence['invocation']['source'] + '/' + relative
    assert any('with args' in line and top in line and vector_arg in line for line in evidence['console'].splitlines())
    log = evidence['logs']['lane_integration/simulate.log']
    if binary:
        assert log.splitlines().count('JOIN_BINARY_SCOPE rows=632 scenarios=60 startup_ns=200 sample_ns=1 native=0') == 1
        assert baseline_log is not None
        baseline = '\n'.join(line for line in baseline_log.splitlines() if line.startswith('PCIE_INV_ROW '))
        # The full bank continues into X/Z scenarios; the binary bank ends with
        # its own final flush. Compare all 632 input reference rows, then use
        # the unchanged retirement checker for every binary publication row.
        baseline = '\n'.join(baseline.splitlines()[:632]) + '\n'
    else:
        baseline = log

    def check(text):
        return review_integration({'normal/simulate.log': text}, vectors, manifest, baseline,
                                  internal=internal, mutants=False)

    checked = check(log)
    assert checked['table_word_comparisons'] == (len(expected) * 256 if internal else 0)
    rows = [line for line in log.splitlines() if line.startswith('JOIN_ROW ')]
    active = next(line for line in rows if re.search(r' bytes=[1-9]', line))
    faulty = next(line for line in rows if not re.search(r' fault=0+ ', line))
    controls = []
    variants = [
        ('missing_row', log.replace(rows[0] + '\n', '', 1)),
        ('missing_final_flush', log.replace(rows[-1] + '\n', '', 1)),
        ('wrong_bytes', log.replace(active, re.sub(r' bytes=\S+', ' bytes=0', active), 1)),
        ('wrong_faults', log.replace(faulty, re.sub(r' fault=\S+', ' fault=0000', faulty), 1)),
        ('wrong_words', log.replace('words=' + ('256' if internal else '0'), 'words=255', 1)),
        ('missing_pass', log.replace('JOIN_SUITE_PASS', 'REMOVED', 1)),
    ]
    for name, changed in variants:
        assert changed != log
        try:
            check(changed)
        except ValueError as exc:
            controls.append(dict(name=name, rejected=True, reason=str(exc)))
        else:
            raise AssertionError('Lane integration corruption admitted: ' + name)
    return dict(measurements=checked, controls=controls, unknown_input_rows=unknown_rows,
                raw_vectors_regenerated=True, exact_top_and_vector_argument_verified=True,
                reference='Immutable R7 RTL, raw inputs only; Python expectations before faults',
                all_binary_source_reference_rows_identical=(binary and baseline_log is not None))
