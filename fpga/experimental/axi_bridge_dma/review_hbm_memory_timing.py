"""Verify private memory timing evidence; retain the distinction from board closure."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from review_hbm_clocktree import review as physical_review

DONOR_SHA = 'cb110a36e78b08860b041443d7ba85fb913d66ee0c5e4f896d064f0767170ce6'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constraints(path):
    return Counter(line for line in path.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith('#'))


def domains(path):
    text = path.read_text().split('| Intra Clock Table\n', 1)[1].split('| Inter Clock Table\n', 1)[0]
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in ('mem_clk', 'mgmt_clk', 'pcie_clk', 'reader_clk'):
            if len(fields) != 13 or fields[0] in result:
                raise ValueError('Invalid clock timing row')
            result[fields[0]] = dict(wns_ns=float(fields[1]), tns_ns=float(fields[2]),
                setup_failures=int(fields[3]), setup_endpoints=int(fields[4]),
                whs_ns=float(fields[5]), hold_failures=int(fields[7]),
                wpws_ns=float(fields[9]), pulse_failures=int(fields[11]))
    if set(result) != {'mem_clk', 'mgmt_clk', 'pcie_clk', 'reader_clk'}:
        raise ValueError('Clock timing inventory incomplete')
    return result


def review(root, expected_donor=DONOR_SHA):
    manifest = json.loads((root / 'FILES.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest) | {'FILES.json'}:
        raise ValueError('Evidence inventory mismatch')
    for name, digest in manifest.items():
        if sha(root / name) != digest:
            raise ValueError('Evidence checksum mismatch: ' + name)
    result = json.loads((root / 'RESULT.json').read_text())
    if result['status'] != 'HBM_MEMORY_TIMING_EXPORTED' or result['error'] is not None or not result['sources_unchanged']:
        raise ValueError('Incomplete optimization')
    if result['donor_sha256'] != expected_donor or sha(root / 'source/donor.dcp') != expected_donor:
        raise ValueError('Wrong optimization donor')
    if [s['name'] for s in result['steps']] != ['refinement', 'reopen']:
        raise ValueError('Missing independent reopen')
    if any(s['returncode'] or s['timed_out'] for s in result['steps']):
        raise ValueError('Failed tool step')
    for name, digest in result['sources'].items():
        if sha(root / 'source' / name) != digest:
            raise ValueError('Executed source snapshot changed')
    out = root / 'refinement'
    if (out / 'STATUS.txt').read_text().strip() != 'HBM_MEMORY_TIMING_EXPORTED':
        raise ValueError('Missing Tcl completion')
    if constraints(out / 'before.xdc') != constraints(out / 'after.xdc'):
        raise ValueError('Optimization changed exported constraints')
    if sha(out / 'refined.dcp') != sha(root / 'physical/source/donor.dcp'):
        raise ValueError('Wrong optimized checkpoint reopened')
    physical = physical_review(root / 'physical')
    if constraints(out / 'after.xdc') != constraints(root / 'physical/output/before.xdc'):
        raise ValueError('Reopened constraints differ')
    before, after = domains(out / 'before_timing.rpt'), domains(out / 'after_timing.rpt')
    if after != domains(root / 'physical/output/audit/timing.rpt'):
        raise ValueError('Reopened timing differs')
    for name in ('refinement', 'reopen'):
        if re.search(r'(?im)^\s*(?:fatal|error):', (root / (name + '.log')).read_text()):
            raise ValueError('Tool error in ' + name)
    commands = {'replicate': 'phys_opt_design -critical_cell_opt -path_groups mem_clk',
                'restructure': 'phys_opt_design -restruct_opt -path_groups mem_clk',
                'explore': 'phys_opt_design -directive AggressiveExplore'}
    if result['strategy'] not in commands:
        raise ValueError('Unknown optimization strategy')
    if 'Command: ' + commands[result['strategy']] not in (root / 'refinement.log').read_text():
        raise ValueError('Optimization command not executed')
    return dict(status='HBM_MEMORY_TIMING_REVIEWED', strategy=result['strategy'],
                donor_sha256=expected_donor, checkpoint_sha256=sha(out / 'refined.dcp'),
                seconds=result['seconds'], before=before, after=after, physical=physical,
                constraints_unchanged=True, internal_timing_met=physical['internal_timing_met'],
                native_hbm_pcie_ip=False, board_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--expected-donor', default=DONOR_SHA)
    args = parser.parse_args()
    print(json.dumps(review(args.root, args.expected_donor), indent=2))
