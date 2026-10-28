"""Synthetic source-provenance controls; these do not execute/elaborate RTL."""
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import review_ntrpu_ddr_activity_identity_r2 as checker


def fixture(base):
    run = base / 'synthetic-capture'
    run.mkdir()
    native = '/synthetic/native/capture'
    model = dict(ip_inst=dict(component_reference='xilinx.com:ip:ddr4:2.2', ip_revision='30',
                 parameters=dict(component_parameters={'C0.DDR4_CustomParts':[dict(value='../../../../../geometry.csv')],
                     'System_Clock':[dict(value='No_Buffer')], 'C0.DDR4_Ecc':[dict(value='true')]})))
    source = {'original/example/' + checker.XCI:json.dumps(model).encode(),
              'original/example/imports/example_top.sv':b'module example_top; endmodule\n',
              'original/example/imports/ddr4_v2_2_axi_tg_top.sv':b'module tg; endmodule\n',
              'original/example/imports/ddr4_v2_2_prbs_mode_gen.sv':b'module prbs; endmodule\n',
              'original/example/imports/ddr4_sdram_model_wrapper.sv':b'module old_model; endmodule\n',
              'original/example/imports/glbl.v':b'module glbl; endmodule\n',
              'original/example/ntrpu_ddr_geometry_ex.gen/helper.vhd':b'-- synthetic IP\n',
              'packet/ntrpu_dci_startup.sv':b'module startup; endmodule\n',
              'packet/ddr4_sdram_model_wrapper.sv':b'module prepared_model; endmodule\n',
              'packet/ntrpu_ddr_example_activity_tb.sv':b'module model; endmodule\nmodule tb;\nendmodule\n',
              'packet/geometry.csv':b'geometry\n'}
    with zipfile.ZipFile(run / 'source.zip', 'x') as z:
        for n, data in source.items():
            z.writestr(n, data)
        z.writestr('SNAPSHOT_FILES.json', json.dumps({n:hashlib.sha256(d).hexdigest() for n,d in source.items()}))
    source_sha = checker.digest(run / 'source.zip')
    paths = {}

    def put(name, data):
        p = run / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode() if isinstance(data, str) else data)
        paths[name] = p
        return p

    for name, raw in source.items():
        if name.startswith('original/example/'):
            put(checker.PROJECT + name[len('original/example/'):], raw)
    put('output/geometry.csv', source['packet/geometry.csv'])
    put(checker.PROJECT + 'imports/ddr4_sdram_model_wrapper.sv', source['packet/ddr4_sdram_model_wrapper.sv'])
    put(checker.PROJECT + 'activity_inputs/ntrpu_dci_startup.sv', source['packet/ntrpu_dci_startup.sv'])
    tb = source['packet/ntrpu_ddr_example_activity_tb.sv'].decode()
    at = tb.rfind('endmodule')
    put(checker.PROJECT + checker.TB, tb[:at] + checker.IDENTITY_BLOCK + tb[at:])
    model['ip_inst']['parameters']['component_parameters']['C0.DDR4_CustomParts'][0]['value'] = native + '/output/geometry.csv'
    put(checker.PROJECT + checker.XCI, json.dumps(model))
    invocation = dict(output=native, source_sha256=source_sha, finished_utc='synthetic',
                      exit_code=0, frozen_source_unchanged=True)
    put('invocation.json', json.dumps(invocation))
    simdir = checker.PROJECT + 'ntrpu_ddr_geometry_ex.sim/sim_1/behav/xsim/'
    leaves = ['imports/example_top.sv', 'imports/ddr4_v2_2_axi_tg_top.sv',
              'imports/ddr4_v2_2_prbs_mode_gen.sv', 'imports/ddr4_sdram_model_wrapper.sv',
              'activity_inputs/ntrpu_dci_startup.sv', checker.TB]
    vlog = 'sv xil_defaultlib --include "../../../../imports" --define "SYNTHETIC=1" \\\n'
    vlog += ' \\\n'.join('"../../../../' + n + '"' for n in leaves)
    vlog += '\nverilog xil_defaultlib "glbl.v"\nnosort\n'
    vhdl = 'vhdl xil_defaultlib "../../../../ntrpu_ddr_geometry_ex.gen/helper.vhd"\nnosort\n'
    put(simdir + 'ntrpu_ddr_example_activity_tb_vlog.prj', vlog)
    put(simdir + 'ntrpu_ddr_example_activity_tb_vhdl.prj', vhdl)
    put(simdir + 'glbl.v', source['original/example/imports/glbl.v'])
    put(simdir + 'elaborate.log', 'Completed static elaboration\nBuilt simulation snapshot ntrpu_ddr_example_activity_tb_behav\n')
    put(simdir + 'simulate.log', 'NTRPU_ACTIVITY_IDENTITY dut_simulation= FALSE tg_simulation= FALSE\n')
    return run, source_sha, paths, simdir


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='ntrpu-ddr-identity-') as temporary:
        run, sha, paths, simdir = fixture(Path(temporary))
        baseline = checker.review(run, source_sha=sha)
        assert not baseline['errors'] and baseline['native_elaboration_observed'], baseline
        assert len(baseline['compiled_sources']) == 8
        assert baseline['defines'] == ['SYNTHETIC=1']
        assert baseline['pending'] and not baseline['ready_for_power_estimate']
        before = {p.relative_to(run).as_posix():checker.digest(p) for p in run.rglob('*') if p.is_file()}
        checks.append('synthetic_positive_still_requires_DCP_and_activity_review')

        def rejection(name, path, transform, expected):
            original = path.read_bytes()
            try:
                path.write_bytes(transform(original))
                result = checker.review(run, source_sha=sha)
                assert any(expected in e for e in result['errors']), (name, result)
                assert not result['ready_for_power_estimate']
                checks.append(name)
            finally:
                path.write_bytes(original)

        inv = paths['invocation.json']
        def json_change(key, value):
            def change(raw):
                obj = json.loads(raw)
                if value is None:obj.pop(key)
                else:obj[key] = value
                return json.dumps(obj).encode()
            return change
        rejection('running_capture', inv, json_change('finished_utc', None), 'still active')
        rejection('foreign_source', inv, json_change('source_sha256', '0'*64), 'source identity')
        rejection('mutable_frozen_source', inv, json_change('frozen_source_unchanged', False), 'changed or unverified')
        rejection('failed_invocation_preserved', inv, json_change('exit_code', 1), 'Native invocation failed')
        rejection('nonabsolute_remote_root', inv, json_change('output', 'relative/output'), 'output path')
        rejection('altered_source_zip', run/'source.zip', lambda raw:raw+b'x', 'ZIP hash mismatch')
        for leaf in ('imports/example_top.sv', 'imports/ddr4_v2_2_axi_tg_top.sv',
                     'imports/ddr4_v2_2_prbs_mode_gen.sv', 'imports/ddr4_sdram_model_wrapper.sv',
                     'activity_inputs/ntrpu_dci_startup.sv', checker.TB):
            rejection('changed_'+Path(leaf).name, paths[checker.PROJECT+leaf], lambda raw:raw+b'\n', 'Unapproved compiled')
        rejection('changed_geometry', paths['output/geometry.csv'], lambda raw:raw+b'x', 'Geometry bytes changed')
        xci = paths[checker.PROJECT+checker.XCI]
        def change_parameter(key, value):
            def change(raw):
                obj=json.loads(raw);obj['ip_inst']['parameters']['component_parameters'][key]=[dict(value=value)]
                return json.dumps(obj).encode()
            return change
        rejection('changed_ECC',xci,change_parameter('C0.DDR4_Ecc','false'),'parameter values changed')
        rejection('external_geometry',xci,change_parameter('C0.DDR4_CustomParts','/elsewhere/geometry.csv'),'not private')
        rejection('extra_parameter',xci,change_parameter('unexpected','1'),'parameter set changed')
        vlog=paths[simdir+'ntrpu_ddr_example_activity_tb_vlog.prj']
        rejection('unknown_compile_option',vlog,lambda raw:raw.replace(b'--include',b'--surprise'),'Unknown PRJ option')
        rejection('external_compiled_source',vlog,lambda raw:raw.replace(b'../../../../imports/example_top.sv',b'/outside/example_top.sv'),'outside restored capture')
        rejection('missing_top',vlog,lambda raw:raw.replace(b'"../../../../imports/example_top.sv"',b''),'Required hardware')
        rejection('duplicate_top',vlog,lambda raw:raw+b'\nsv xil_defaultlib "../../../../imports/example_top.sv"\n','Duplicate PRJ')
        rejection('wrong_global_library',paths[simdir+'glbl.v'],lambda raw:raw+b'\n','Unapproved compiled')
        rejection('missing_elaboration_marker',paths[simdir+'elaborate.log'],lambda raw:b'not complete','completion missing')
        rejection('wrong_native_SIMULATION',paths[simdir+'simulate.log'],lambda raw:raw.replace(b'FALSE',b'TRUE'),'hardware-mode identity missing')
        generated=paths[checker.PROJECT+'ntrpu_ddr_geometry_ex.gen/helper.vhd'];original=generated.read_bytes()
        generated.write_bytes(original+b'-- regenerated\n')
        diagnostic=checker.review(run,source_sha=sha)
        assert not diagnostic['errors'] and any('regenerated' in p for p in diagnostic['pending'])
        assert not diagnostic['ready_for_power_estimate']
        generated.write_bytes(original)
        checks.append('regenerated_IP_explicitly_left_unqualified')
        global_path=paths[simdir+'glbl.v'];archived_global=global_path.read_bytes()
        vendor=VENDOR_INPUT.read_bytes()
        assert hashlib.sha256(vendor).hexdigest()==checker.VENDOR_GLBL_SHA256
        global_path.write_bytes(vendor)
        vendor_report=checker.review(run,source_sha=sha)
        assert not vendor_report['errors'] and vendor_report['native_elaboration_observed']
        assert len(vendor_report['global_library_variants'])==1
        assert not vendor_report['global_library_variants'][0]['reset_behavior_equivalence_qualified']
        assert any('reset-global behavior' in p for p in vendor_report['pending']) and not vendor_report['ready_for_power_estimate']
        checks.append('installed_vendor_global_identified_but_equivalence_pending')
        for label, before_text, after_text in [('changed_GRESTORE',b'GRES_WIDTH = 10000',b'GRES_WIDTH = 10001'),
                ('changed_GSR_strength',b'(strong1, weak0) GSR',b'(weak1, weak0) GSR'),
                ('changed_global_reset_time',b'ROC_WIDTH = 100000',b'ROC_WIDTH = 100001')]:
            assert before_text in vendor
            rejection(label,global_path,lambda raw,a=before_text,b=after_text:raw.replace(a,b,1),'Unapproved compiled')
        global_path.write_bytes(archived_global)
        after={p.relative_to(run).as_posix():checker.digest(p) for p in run.rglob('*') if p.is_file()}
        assert before==after
        checks.append('all_input_bytes_preserved')
    print(json.dumps(dict(status='SYNTHETIC_DDR_ACTIVITY_IDENTITY_CONTROLS_PASS',
                         checks=checks, count=len(checks), native_elaboration_tested=False)))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor-glbl',type=Path,required=True)
    VENDOR_INPUT=parser.parse_args().vendor_glbl
    main()
