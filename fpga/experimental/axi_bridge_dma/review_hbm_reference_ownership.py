"""Review closed reference-ownership evidence; never infer board qualification."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

TOP_SHA = 'f73a5c8ff85e6ba70325d0a1c123ee649c030d4fdc50024e77240311bd679462'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def config(path):
    return dict(line.split('\t', 1) for line in path.read_text().splitlines())


def closed(root, stage, allow_driver_check_failure=False, with_rtl=True):
    files = json.loads((root / 'FILES.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(files) | {'FILES.json'}, 'Closed file inventory differs')
    for name, digest in files.items():
        require(sha(root / name) == digest, 'Hash mismatch: ' + name)
    result = json.loads((root / 'RESULT.json').read_text())
    failed_check = allow_driver_check_failure and result['status'] == 'FAILED'
    require((result['status'] == 'HBM_NATIVE_JOIN_EXPORTED' or failed_check) and result['stage'] == stage,
            'Unexpected stage status')
    require(result['returncode'] == (1 if failed_check else 0) and not result['timed_out'] and result['sources_unchanged'],
            'Tool or source validation failed')
    if failed_check:
        log = (root / 'tool.log').read_text()
        require(len(re.findall(r'^Reference input-buffer driver differs$', log, re.M)) == 2 and
                'Synthesis finished with 0 errors, 0 critical warnings' in log and
                not re.search(r'^\s*(?:ERROR|FATAL):', log, re.M), 'Unexpected synthesis failure')
    for name, digest in result['sources'].items():
        require(sha(root / 'source' / name) == digest, 'Source differs: ' + name)
    if with_rtl:
        binding = json.loads((root / 'source/REFERENCE_OWNERSHIP_BINDING.json').read_text())
        for name, digest in binding['donor_files'].items():
            if Path(name).suffix in ('.sv', '.v', '.vh', '.xci'):
                require(sha(root / 'source' / name) == digest, 'Frozen RTL/XCI input changed: ' + name)
        require(sha(root / 'source/ntrpu_axi_bridge_hbm_controlled.sv') == TOP_SHA, 'Board RTL changed')
    return result, len(files)


def generation(root, baseline):
    result, count = closed(root, 'generate')
    out = root / 'output'
    old = baseline / 'output'
    names = {p.name for p in old.glob('*_config.tsv')}
    require(len(names) == 7, 'Expected seven donor IP configurations')
    changes = {}
    property_count = 0
    for name in sorted(names):
        before, after = config(old / name), config(out / name)
        property_count += len(after)
        delta = {k: [before.get(k), after.get(k)] for k in sorted(before.keys() | after.keys())
                 if before.get(k) != after.get(k)}
        expected = {'CONFIG.USER_DIS_REF_CLK_BUFG': ['FALSE', 'TRUE']} if name == 'native_hbm_config.tsv' else {}
        require(delta == expected, 'Unexpected IP change: ' + name)
        if delta:
            changes[name] = delta
    require(config(out / 'hbm_before_config.tsv') == config(old / 'native_hbm_config.tsv'),
            'Imported HBM configuration differs from frozen generation')
    prefix = 'project/native_join.gen/sources_1/ip/native_hbm'
    rtl = (out / prefix / 'hdl/rtl/hbm_v1_0_20.sv').read_text()
    require(not re.search(r'\b(?:IBUF|BUFG)\s+u_HBM_REF_CLK_', rtl), 'Generated reference buffers remain')
    for i in (0, 1):
        require(f'.HBM_REF_CLK_{i}(HBM_REF_CLK_{i})' in rtl, 'Missing direct reference connection')
    xdc = (out / prefix / 'hdl/par/hbm_ip.xdc').read_text()
    active = [line.strip() for line in xdc.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    require(not active, 'Unexpected native primary constraints: ' + repr(active))
    require((out / 'STATUS.txt').read_text().strip() == 'HBM_NATIVE_JOIN_GENERATED', 'Generation status differs')
    return dict(status='HBM_REFERENCE_GENERATION_REVIEWED', closed_files_verified=count,
                ip_configurations_compared=7, properties_compared=property_count, configuration_changes=changes,
                native_reference_buffers_generated=False, native_primary_clocks_generated=False,
                generated_rtl_sha256=sha(out / prefix / 'hdl/rtl/hbm_v1_0_20.sv'),
                generated_xdc_sha256=sha(out / prefix / 'hdl/par/hbm_ip.xdc'),
                tool_seconds=result['seconds'], board_qualified=False)


def synthesis(root, inspection=None):
    result, count = closed(root, 'synth', allow_driver_check_failure=inspection is not None)
    out = root / 'output'
    checkpoint_sha = sha(out / 'synth.dcp')
    inspection_count = 0
    if inspection:
        reopened, inspection_count = closed(inspection, 'inspect', with_rtl=False)
        require(reopened['checkpoint_sha256'] == checkpoint_sha, 'Inspection used another checkpoint')
        out = inspection / 'output'
        def commands(path):
            return [line.strip() for line in path.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
        require(commands(out / 'before.xdc') == commands(out / 'after.xdc'), 'Inspection changed constraints')
    native = root / 'project/native_join.gen/sources_1/ip/native_hbm'
    require(sha(native / 'hdl/rtl/hbm_v1_0_20.sv') ==
            '5eca1fb6dc2e3b191a40363416f9ac23a44ae65568d47a97c6b6108defb6abdc', 'Wrong generated HBM RTL')
    require(sha(native / 'hdl/par/hbm_ip.xdc') ==
            'e194bfde9bc7a58e4e6484206cc8cfa606a1e99b91abaa066dc06582f5eacbfd', 'Wrong generated HBM constraints')
    require((out / 'REFERENCE_STATUS.txt').read_text().strip() ==
            'HBM_REFERENCE_OWNERSHIP_SYNTHESIZED_REVIEW_REQUIRED', 'Missing native topology audit')
    rows = table(out / 'reference_tree.tsv')
    require(len(rows) == 2, 'Expected two reference trees')
    for i, row in enumerate(rows):
        for key, value in dict(stack=str(i), pad=f'hbm_ref{i}_clk_p',
                               input_buffer=f'board_inputs/hbm_reference{i}_input',
                               input_leaf_driver=f'board_inputs/hbm_reference{i}_input/IBUFCTRL_INST/O',
                               global_buffer=f'hbm_reference{i}', ip_pin=f'hbm_binding/memory/HBM_REF_CLK_{i}',
                               clock=f'hbm_reference{i}_clk', clock_source=f'hbm_ref{i}_clk_p').items():
            require(row[key] == value, 'Reference topology differs: ' + key)
        require(float(row['period_ns']) == 10.0 and row['leaf_loads'], 'Reference clock or load missing')
    clocks = table(out / 'reference_clock_inventory.tsv')
    require(not any('HBM_REF_CLK' in row['clock'] for row in clocks), 'Internal HBM primary clock remains')
    for i in (0, 1):
        matches = [row for row in clocks if row['clock'] == f'hbm_reference{i}_clk']
        require(len(matches) == 1 and matches[0]['generated'] == '0' and
                matches[0]['sources'] == f'hbm_ref{i}_clk_p', 'Wrong primary clock inventory')
    method = (out / 'reference_methodology.rpt').read_text()
    rules = {m[0]: int(m[1]) for m in re.findall(
        r'^\|\s*([A-Z]+-\d+)\s*\|[^|]+\|[^|]+\|\s*(\d+)\s*\|', method, re.M)}
    require('REPORT DETAILS' in method and rules, 'Incomplete methodology report')
    require(not any(rules.get(code, 0) for code in ('TIMING-2', 'TIMING-4', 'CLKC-5')),
            'Reference-clock methodology violations remain')
    ports = table(out / 'ports.tsv')
    require(len(ports) == 75, 'External board interface changed')
    require(len(table(out / 'native_instances.tsv')) == 4, 'Missing native blocks')
    require(re.search(r'^No waivers found\.$', (out / 'reference_waivers.rpt').read_text(), re.M),
            'Review waiver report')
    return dict(status='HBM_REFERENCE_SYNTHESIS_REVIEWED', closed_files_verified=count,
                inspection_closed_files_verified=inspection_count, original_stage_status=result['status'],
                reopened_same_checkpoint=inspection is not None,
                reference_trees=rows, clocks=len(clocks), external_port_bits=len(ports),
                corrected_methodology_counts={code: rules.get(code, 0) for code in ('TIMING-2', 'TIMING-4', 'CLKC-5')},
                remaining_methodology=rules, checkpoint_sha256=checkpoint_sha,
                tool_seconds=result['seconds'], board_qualified=False, physical_placement_executed=False,
                routed_timing_qualified=False, native_traffic_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('generation', 'synthesis'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--inspection', type=Path)
    args = parser.parse_args()
    if args.mode == 'generation' and not args.baseline:
        parser.error('generation requires --baseline')
    print(json.dumps(generation(args.root, args.baseline) if args.mode == 'generation' else synthesis(args.root, args.inspection), indent=2))
