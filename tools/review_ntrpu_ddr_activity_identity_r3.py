"""Audit restored DDR capture sources; never certify SAIF/DCP correspondence."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import shlex
import zipfile

VENDOR_GLBL_SHA256 = '6270251cf35e1fce3119fc79f8fb968912f7983027e43cf030409ba68f048936'
SOURCE_SHA = 'acb1a3923b940cc5d9d076c264c30787336ed188b7eaf97e837291e0fc2442a3'
PROJECT = 'output/example/'
XCI = 'ntrpu_ddr_geometry_ex.srcs/sources_1/ip/ntrpu_ddr_geometry/ntrpu_ddr_geometry.xci'
TB = 'activity_inputs/ntrpu_ddr_example_activity_identity_r1_tb.sv'
IDENTITY_BLOCK = '''    initial begin
        #1;
        $display("NTRPU_ACTIVITY_IDENTITY dut_simulation=%s tg_simulation=%s",
                 dut.SIMULATION, dut.u_ddr4_v2_2_30_axi_tg_top.SIMULATION);
        if (dut.SIMULATION != "FALSE" ||
            dut.u_ddr4_v2_2_30_axi_tg_top.SIMULATION != "FALSE")
            $fatal(1, "NTRPU_ACTIVITY hardware-mode parameter mismatch");
    end
'''


def require(ok, why):
    if not ok:
        raise ValueError(why)


def stream_digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def digest(path):
    with Path(path).open('rb') as f:
        return stream_digest(f)


def safe_relative(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and
            ':' not in name and '\\' not in name and str(p) == name, 'Unsafe relative path: ' + name)
    return p


def parse_project(text):
    """Read generated XSim PRJ syntax, retaining include/define options separately."""
    sources, includes, defines = [], [], []
    logical = text.replace('\\\r\n', ' ').replace('\\\n', ' ')
    for line in logical.splitlines():
        words = shlex.split(line, comments=True)
        if not words or words == ['nosort']:
            continue
        require(words[0] in ('sv', 'verilog', 'vhdl', 'vhdl2008'), 'Unknown PRJ language/directive')
        require(len(words) >= 3, 'Incomplete PRJ source record')
        language, library = words[:2]
        index = 2
        while index < len(words):
            token = words[index]
            if token in ('--include', '--define'):
                require(index + 1 < len(words), 'Missing PRJ option value')
                (includes if token == '--include' else defines).append(words[index + 1])
                index += 2
            else:
                require(not token.startswith('-'), 'Unknown PRJ option: ' + token)
                sources.append(dict(language=language, library=library, path=token))
                index += 1
    require(bool(sources), 'Empty PRJ source list')
    require(len(sources) == len({(r['language'], r['library'], r['path']) for r in sources}),
            'Duplicate PRJ source record')
    return sources, sorted(set(includes)), sorted(set(defines))


def component_values(raw):
    ip = json.loads(raw)['ip_inst']
    require(ip['component_reference'] == 'xilinx.com:ip:ddr4:2.2' and str(ip['ip_revision']) == '30',
            'Unexpected DDR IP identity')
    return {k: [row['value'] for row in v]
            for k, v in ip['parameters']['component_parameters'].items()}


def review(run, *, source_sha=SOURCE_SHA):
    run = Path(run).resolve()
    inputs, errors, pending = {}, [], []
    result = dict(run=run.name, errors=errors, pending=pending, inputs=inputs,
                  compiled_sources=[], include_directories=[], defines=[],
                  native_elaboration_observed=False, saif_dcp_correspondence_verified=False,
                  ready_for_power_estimate=False, board_power_qualified=False,
                  exact_dimm_qualified=False)

    def read(name):
        safe_relative(name)
        p = run / name
        require(p.resolve().is_relative_to(run) and not p.is_symlink(), 'Restored path escapes run')
        data = p.read_bytes()
        inputs[name] = hashlib.sha256(data).hexdigest()
        return data

    try:
        invocation = json.loads(read('invocation.json'))
        require('finished_utc' in invocation, 'Capture is still active')
        require(invocation['source_sha256'] == source_sha, 'Unexpected capture source identity')
        require(invocation.get('frozen_source_unchanged') is True, 'Frozen source changed or unverified')
        if invocation['exit_code'] != 0:
            errors.append('Native invocation failed; source audit is diagnostic only')
        remote_run = PurePosixPath(invocation['output'])
        require(remote_run.is_absolute() and '..' not in remote_run.parts, 'Invalid native output path')
        source = run / 'source.zip'
        require(digest(source) == source_sha, 'Frozen ZIP hash mismatch')
        inputs['source.zip'] = source_sha
        expected = {}
        with zipfile.ZipFile(source) as z:
            names = z.namelist()
            require(len(names) == len(set(names)), 'Duplicate frozen ZIP member')
            for name in names:
                safe_relative(name)
            hashes = json.loads(z.read('SNAPSHOT_FILES.json'))
            require(set(names) == set(hashes) | {'SNAPSHOT_FILES.json'}, 'Incomplete frozen ZIP inventory')
            for name, sha in hashes.items():
                with z.open(name) as f:
                    require(stream_digest(f) == sha, 'Frozen member hash mismatch: ' + name)
                if name.startswith('original/example/'):
                    expected[PROJECT + name[len('original/example/'):]] = (sha, 'archived_exact')
            result['frozen_source_files'] = len(hashes)
            old_parameters = component_values(z.read('original/example/' + XCI))
            for leaf, member, kind in (
                ('imports/ddr4_sdram_model_wrapper.sv', 'packet/ddr4_sdram_model_wrapper.sv', 'approved_model_wrapper'),
                ('activity_inputs/ntrpu_dci_startup.sv', 'packet/ntrpu_dci_startup.sv', 'archived_startup'),
            ):
                expected[PROJECT + leaf] = (hashes[member], kind)
            original_tb = z.read('packet/ntrpu_ddr_example_activity_tb.sv').decode().replace('\r\n', '\n')
            require(original_tb.count('endmodule') == 2, 'Unexpected frozen bank shape')
            index = original_tb.rfind('endmodule')
            derived = (original_tb[:index] + IDENTITY_BLOCK + original_tb[index:]).encode()
            expected[PROJECT + TB] = (hashlib.sha256(derived).hexdigest(), 'observational_identity_bank')
            geometry_sha = hashes['packet/geometry.csv']
            global_sha = hashes['original/example/imports/glbl.v']

        new_parameters = component_values(read(PROJECT + XCI))
        require(set(old_parameters) == set(new_parameters), 'DDR component parameter set changed')
        differences = {k:dict(before=v, after=new_parameters[k]) for k, v in old_parameters.items()
                       if v != new_parameters[k]}
        require(set(differences) <= {'C0.DDR4_CustomParts'}, 'DDR component parameter values changed')
        geometry_values = new_parameters['C0.DDR4_CustomParts']
        require(len(geometry_values) == 1, 'Ambiguous private geometry path')
        xci_remote = remote_run / (PROJECT + XCI)
        geometry_remote = PurePosixPath(posixpath.normpath(posixpath.join(str(xci_remote.parent), geometry_values[0])))
        require(geometry_remote == remote_run / 'output/geometry.csv', 'Geometry is not private to capture')
        require(hashlib.sha256(read('output/geometry.csv')).hexdigest() == geometry_sha, 'Geometry bytes changed')
        result['component_parameter_differences'] = differences
        result['parameters_equivalent_except_private_geometry'] = True

        projects = sorted((run / 'output').rglob('ntrpu_ddr_example_activity_tb_*.prj'))
        require({p.name for p in projects} == {'ntrpu_ddr_example_activity_tb_vlog.prj',
                                              'ntrpu_ddr_example_activity_tb_vhdl.prj'} and len(projects) == 2,
                'Missing or ambiguous native XSim compile projects')
        compiled = set()
        for project in projects:
            local_name = project.relative_to(run).as_posix()
            sources, includes, defines = parse_project(read(local_name).decode())
            native_dir = remote_run / project.relative_to(run).parent.as_posix()
            expected[project.relative_to(run).parent.as_posix() + '/glbl.v'] = (global_sha, 'archived_global_copy')
            result['include_directories'].extend(posixpath.normpath(posixpath.join(str(native_dir), p)) for p in includes)
            result['defines'].extend(defines)
            for row in sources:
                native_path = PurePosixPath(posixpath.normpath(posixpath.join(str(native_dir), row['path'])))
                require(native_path.is_relative_to(remote_run), 'Compiled file outside restored capture: ' + str(native_path))
                name = native_path.relative_to(remote_run).as_posix()
                require(name not in compiled, 'Duplicate resolved compiled source: ' + name)
                compiled.add(name)
                actual = hashlib.sha256(read(name)).hexdigest()
                known = expected.get(name)
                if known and actual == known[0]:
                    classification = known[1]
                elif name == project.relative_to(run).parent.as_posix() + '/glbl.v' and actual == VENDOR_GLBL_SHA256:
                    classification = 'installed_vivado_2026_1_global_library'
                    result.setdefault('global_library_variants', []).append(dict(path=name,
                        archived_sha256=global_sha, compiled_sha256=actual,
                        archived_bytes_identical=False, reset_behavior_equivalence_qualified=False))
                    pending.append('Installed Vivado glbl differs from archived import; review reset-global behavior and DCP scope')
                elif '.gen/' in name and name.startswith(PROJECT):
                    classification = 'generated_changed' if known else 'generated_new'
                    pending.append('Review regenerated simulation source: ' + name)
                else:
                    raise ValueError('Unapproved compiled source or bytes: ' + name)
                result['compiled_sources'].append(dict(path=name, sha256=actual,
                    classification=classification, language=row['language'], library=row['library']))
        required = {PROJECT + leaf for leaf in (
            'imports/example_top.sv', 'imports/ddr4_v2_2_axi_tg_top.sv',
            'imports/ddr4_v2_2_prbs_mode_gen.sv', 'imports/ddr4_rdimm_wrapper.sv',
            'activity_inputs/ntrpu_dci_startup.sv', TB)}
        require(required <= compiled, 'Required hardware, model or observational bank absent from compile plan')
        unused=PROJECT+'imports/ddr4_sdram_model_wrapper.sv'
        require(hashlib.sha256(read(unused)).hexdigest()==expected[unused][0], 'Prepared unused model wrapper bytes changed')
        result['model_source_selection']=dict(active=PROJECT+'imports/ddr4_rdimm_wrapper.sv',
            active_sha256=inputs[PROJECT+'imports/ddr4_rdimm_wrapper.sv'],
            prepared_wrapper=unused,prepared_wrapper_compiled=unused in compiled,
            geometry_owner=PROJECT+TB,
            scope='The RDIMM wrapper is selected by the archived observational bank. Macro edits in the standalone SDRAM wrapper do not qualify the active RDIMM model.')
        if unused not in compiled:pending.append('Prepared SDRAM macro wrapper was not compiled; active RDIMM geometry comes from explicit bank parameters')

        for leaf in ('elaborate.log', 'simulate.log'):
            logs = list((run / 'output').rglob(leaf))
            require(len(logs) == 1, 'Missing or ambiguous native ' + leaf)
            text = read(logs[0].relative_to(run).as_posix()).decode(errors='replace')
            if leaf == 'elaborate.log':
                require('Completed static elaboration' in text and
                        'Built simulation snapshot ntrpu_ddr_example_activity_tb_behav' in text,
                        'Native elaboration completion missing')
            else:
                import re
                require(re.search(r'NTRPU_ACTIVITY_IDENTITY dut_simulation=\s*FALSE tg_simulation=\s*FALSE', text),
                        'Native hardware-mode identity missing')
        result['native_elaboration_observed'] = True
    except Exception as exc:
        errors.append(str(exc))
    result['include_directories'] = sorted(set(result['include_directories']))
    result['defines'] = sorted(set(result['defines']))
    pending.extend(['Audit precompiled vendor libraries, include files and macro definitions',
                    'Reconcile simulation hierarchy with the original routed DCP in Vivado',
                    'Review activity, refresh/ranks, X/Z and unmatched SAIF nets before power use'])
    result['status'] = 'DIAGNOSTIC_DDR_ACTIVITY_SOURCE_IDENTITY_R3' if errors else 'DDR_ACTIVITY_SOURCE_IDENTITY_REVIEWED_WITH_OPEN_ITEMS'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.resolve().is_relative_to(args.run.resolve()), 'Report must be outside immutable capture')
    report = review(args.run)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({k:report[k] for k in ('status', 'errors', 'native_elaboration_observed')}))
