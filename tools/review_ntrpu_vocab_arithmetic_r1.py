"""Review native full vocabulary simulation against frozen sources and fixture."""
import json
from pathlib import Path
import re
import zipfile
import check_ntrpu_vocab_memory_r2 as check


def review(run, source_sha):
    run = Path(run).resolve()
    result = dict(run=run.name, source_sha256=source_sha, errors=[],
        board_power_qualified=False, exact_dimm_qualified=False, whole_model_rtl_qualified=False,
        whole_head_arithmetic_qualified=False, physical_timing_qualified=False)
    try:
        invocation = json.loads((run/'invocation.json').read_text())
        assert 'finished_utc' in invocation and invocation['exit_code'] == 0
        assert invocation['source_sha256'] == source_sha and check.sha(run/'source.zip') == source_sha
        assert invocation['frozen_source_unchanged'] and invocation['frozen_fixture_unchanged']
        assert invocation['cpu_affinity'] == [0,1] and invocation['timeout_seconds'] == 57600
        source = Path(invocation['source'])
        with zipfile.ZipFile(run/'source.zip') as z:
            hashes = json.loads(z.read('SNAPSHOT_FILES.json'))
            assert len(z.namelist()) == len(set(z.namelist()))
            assert set(z.namelist()) == set(hashes) | {'SNAPSHOT_FILES.json'}
            assert all(check.sha(source/n) == h for n,h in hashes.items())
            plan = json.loads(z.read('NATIVE_PLAN.json'))
            binding = json.loads(z.read('FIXTURE_BINDING.json'))
        assert binding == invocation['fixture_binding'] == json.loads((run/'FIXTURE_BINDING.json').read_text())
        assert binding['manifest_sha256'] == check.MANIFEST_SHA
        assert check.sha(binding['archive']) == binding['archive_sha256']
        assert check.sha(run/'fixture-manifest.json') == check.MANIFEST_SHA
        fixture = check.load_fixture(binding['directory'])
        console = (run/'console.log').read_text(errors='replace')
        assert console.splitlines().count('NTRPU_VOCAB_ARITHMETIC_R1_EXPORTED') == 1
        assert not re.search(r'(?im)^\s*(Fatal:|ERROR:)', console)
        projects = sorted((run/'output/project').rglob('*.prj'))
        compiled = {}
        for project in projects:
            for name in re.findall(r'"([^\\"]+)"', project.read_text()):
                path = (project.parent/name).resolve()
                if path.is_file(): compiled[str(path)] = check.sha(path)
        expected = {}
        for name in plan['compiled_sources']:
            path = run/'output/mac_rtl'/Path(name).name if name.startswith('generated/') else source/name
            assert compiled.get(str(path)) == hashes[name], name
            expected[str(path)] = hashes[name]
        extra = {n:h for n,h in compiled.items() if n not in expected}
        assert len(expected) == 23 and len(extra) == 5
        assert all(Path(n).is_relative_to(run/'output/project') for n in extra)
        assert sorted(Path(n).name for n in extra) == plan['generated_simulation_filenames']
        assert not any('stub' in n or 'protocol_model' in n for n in compiled)
        log = run/'output/arithmetic.log'
        text = log.read_text()
        metrics = check.review(text, fixture)
        outputs = check.output_files(run/'output', fixture)
        rejected = check.controls(text, fixture)
        assert len(rejected) == 11
        result.update(metrics=metrics, output_files=outputs, rejected_controls=rejected,
            compiled_sources=compiled, generated_vendor_sources=extra,
            invocation_sha256=check.sha(run/'invocation.json'), log_sha256=check.sha(log),
            fixture_archive_sha256=binding['archive_sha256'], whole_head_arithmetic_qualified=True,
            behavioral_clock_period_ns=5, simulated_full_command_ns=metrics['full_command_cycles']*5,
            scope='Full 128256-row RTL vocabulary arithmetic and memory publication from a CPU-crosschecked input. '
                  'Five generated vendor simulation files recorded; no physical timing or complete model claim.')
    except Exception as exc:
        result['errors'].append(repr(exc))
    result['status'] = 'PASS_FULL_VOCAB_R1_NATIVE_RTL' if not result['errors'] else 'DIAGNOSTIC_FULL_VOCAB_R1'
    return result
