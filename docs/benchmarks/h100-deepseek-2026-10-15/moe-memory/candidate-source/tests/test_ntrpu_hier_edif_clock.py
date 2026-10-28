"""Clock audit controls: topology, polarity, and exact logical fingerprint scope."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_ntrpu_hier_edif_clock_r2 import audit

FIXTURE = '''(edif fixture
 (library work (cell top (view v
  (interface (port clk (direction INPUT)) (port d (direction INPUT)) (port o (direction OUTPUT)))
  (contents
   (instance b (viewref v (cellref BUFGCE (libraryref hdi_primitives))))
   (instance v (viewref v (cellref VCC (libraryref hdi_primitives))))
   (instance g (viewref v (cellref GND (libraryref hdi_primitives))))
   (instance q (viewref v (cellref FDRE (libraryref hdi_primitives))))
   (instance l (viewref v (cellref LUT1 (libraryref hdi_primitives))) (property INIT (string "2'h2")))
   (net clk_raw (joined (portref clk) (portref I (instanceref b))))
   (net clk_buffered (joined (portref O (instanceref b)) (portref C (instanceref q))))
   (net vcc (joined (portref P (instanceref v)) (portref CE (instanceref b)) (portref CE (instanceref q))))
   (net gnd (joined (portref G (instanceref g)) (portref R (instanceref q))))
   (net data (joined (portref d) (portref I0 (instanceref l))))
   (net ld (joined (portref O (instanceref l)) (portref D (instanceref q))))
   (net out (joined (portref Q (instanceref q)) (portref o)))
 )))) (design top (cellref top (libraryref work))))'''
BUFFER = '(instance b (viewref v (cellref BUFGCE (libraryref hdi_primitives))))'
INPUT = '(net clk_raw (joined (portref clk) (portref I (instanceref b))))'
OUTPUT = '(net clk_buffered (joined (portref O (instanceref b)) (portref C (instanceref q))))'

def verify():
    good = audit(FIXTURE)
    assert good['combinational_graph_acyclic'] and good['injected_cycle_rejected']
    assert good['clock_buffer']['clock_loads'] == 1
    assert good['combinational_cells'] == 2
    base = FIXTURE.replace(BUFFER, '').replace(INPUT, '(net clk_raw (joined (portref clk) (portref C (instanceref q))))').replace(OUTPUT, '').replace('(portref CE (instanceref b))', '')
    baseline = audit(base, require_buffer=False)
    assert baseline['logical_fingerprint_sha256'] == good['logical_fingerprint_sha256']
    changed = audit(FIXTURE.replace("2'h2", "2'h1"))
    assert changed['logical_fingerprint_sha256'] != good['logical_fingerprint_sha256']
    changed = audit(FIXTURE.replace('(portref I0 (instanceref l))', '').replace(
        '(portref P (instanceref v))', '(portref P (instanceref v)) (portref I0 (instanceref l))'))
    assert changed['logical_fingerprint_sha256'] != good['logical_fingerprint_sha256']
    cases = {}
    for key, value in [('IS_I_INVERTED', "1'b1"), ('IS_CE_INVERTED', "1'b1"), ('CE_TYPE', 'ASYNC'), ('STARTUP_SYNC', 'TRUE')]:
        cases[key] = FIXTURE.replace(BUFFER, BUFFER[:-1] + f' (property {key} (string "{value}")))')
    cases['missing_buffer'] = base
    cases['missing_clock_load'] = FIXTURE.replace('(portref C (instanceref q))', '')
    cases['bypass_buffer'] = FIXTURE.replace(OUTPUT, '(net clk_buffered (joined (portref O (instanceref b))))').replace(INPUT, INPUT[:-2] + ' (portref C (instanceref q))))')
    cases['clock_drives_data'] = FIXTURE.replace('(portref I0 (instanceref l))', '').replace(OUTPUT, OUTPUT[:-2] + ' (portref I0 (instanceref l))))')
    cases['enable_from_ground'] = FIXTURE.replace('(portref CE (instanceref b))', '').replace('(portref G (instanceref g))', '(portref G (instanceref g)) (portref CE (instanceref b))')
    cases['input_from_data'] = FIXTURE.replace(INPUT, INPUT.replace('(portref clk)', '(portref d)')).replace('(net data (joined (portref d)', '(net data (joined (portref clk)')
    cases['clock_self_feedback'] = FIXTURE.replace('(portref I (instanceref b))', '').replace('(portref O (instanceref b))', '(portref O (instanceref b)) (portref I (instanceref b))')
    cases['unsupported_buffer_pin'] = FIXTURE.replace('(portref CE (instanceref b))', '(portref BAD (instanceref b))')
    cases['duplicate_buffer_property'] = FIXTURE.replace(BUFFER, BUFFER[:-1] + ' (property CE_TYPE (string "SYNC")) (property CE_TYPE (string "SYNC")))')
    controls = []
    for label, text in cases.items():
        assert text != FIXTURE
        try:
            audit(text)
        except ValueError as exc:
            controls.append(dict(name=label, rejected=True, reason=str(exc)))
        else:
            raise AssertionError('Accepted clock corruption: ' + label)
    return dict(negative_controls=controls, buffer_contraction_matches_baseline=True,
                data_parameter_mutation_changes_fingerprint=True, data_connection_mutation_changes_fingerprint=True,
                buffer_counted_as_combinational=True)

if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
