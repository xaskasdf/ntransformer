"""Relocate the archived DDR example inside a new, private activity batch."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    assert not output.exists()
    original = source / 'original/example'
    packet = source / 'packet'
    manifest = json.loads((packet / 'MANIFEST.json').read_text())
    for row in manifest['packet_files']:
        assert digest(packet / row['name']) == row['sha256'], row['name']
    output.mkdir(parents=True)
    project = output / 'example'
    shutil.copytree(original, project)
    shutil.copy2(packet / 'geometry.csv', output / 'geometry.csv')
    shutil.copy2(source / 'original/startup_clock.xdc', output / 'startup_clock.xdc')
    private = project / 'activity_inputs'
    private.mkdir()
    shutil.copy2(packet / 'ntrpu_dci_startup.sv', private)
    # Preserve each external File node separately, avoiding duplicate XPR paths.
    xpr = project / 'ntrpu_ddr_geometry_ex.xpr'
    tree = ET.parse(xpr)
    tree.getroot().set('Path', xpr.as_posix())
    changes = []
    aliases = {'$PPRDIR': project, '$PSRCDIR': project / 'ntrpu_ddr_geometry_ex.srcs'}
    for index, node in enumerate(tree.findall('.//File')):
        old = node.get('Path')
        expanded = old
        for token, path in aliases.items():
            expanded = expanded.replace(token, path.as_posix())
        assert '$' not in expanded, old
        path = Path(expanded).resolve()
        if not path.is_relative_to(project):
            if path.name == 'geometry.csv':
                target = private / ('geometry_external_%d.csv' % index)
                shutil.copy2(packet / 'geometry.csv', target)
            elif path.name == 'startup_clock.xdc':
                target = private / path.name
                shutil.copy2(source / 'original/startup_clock.xdc', target)
            elif path.name == 'ntrpu_dci_startup.sv':
                target = private / path.name
            else:
                raise ValueError('Unresolved external XPR file: ' + old)
            node.set('Path', '$PPRDIR/' + target.relative_to(project).as_posix())
            changes.append(dict(before=old, after=node.get('Path'), sha256=digest(target)))
        else:
            assert path.is_file(), str(path)
        for attr in node.findall('.//Attr[@Name="ImportPath"]'):
            changes.append(dict(metadata_before=attr.get('Val'), metadata_after=node.get('Path')))
            attr.set('Val', node.get('Path'))
    tree.write(xpr, encoding='utf-8', xml_declaration=True)
    # The XCI's existing ../../../../../geometry.csv now resolves privately.
    xci = project / 'ntrpu_ddr_geometry_ex.srcs/sources_1/ip/ntrpu_ddr_geometry/ntrpu_ddr_geometry.xci'
    assert (xci.parent / '../../../../../geometry.csv').resolve() == output / 'geometry.csv'
    assert digest(xci) == digest(original / xci.relative_to(project))
    wrapper = project / 'imports/ddr4_sdram_model_wrapper.sv'
    before = wrapper.read_text()
    for row in manifest['model_wrapper_replacements']:
        assert before.count(row['before']) == 1
        before = before.replace(row['before'], row['after'])
    assert before == (packet / wrapper.name).read_text()
    shutil.copy2(packet / wrapper.name, wrapper)
    # Only an observational identity block is added to a derived TB. The supplied
    # immutable TB, reset stimulus, monitors, and capture Tcl remain archived.
    tb_original = packet / 'ntrpu_ddr_example_activity_tb.sv'
    text = tb_original.read_text()
    assert text.count('endmodule') == 2
    block = '''    initial begin
        #1;
        $display("NTRPU_ACTIVITY_IDENTITY dut_simulation=%s tg_simulation=%s",
                 dut.SIMULATION, dut.u_ddr4_v2_2_30_axi_tg_top.SIMULATION);
        if (dut.SIMULATION != "FALSE" ||
            dut.u_ddr4_v2_2_30_axi_tg_top.SIMULATION != "FALSE")
            $fatal(1, "NTRPU_ACTIVITY hardware-mode parameter mismatch");
    end
'''
    pos = text.rfind('endmodule')
    tb = private / 'ntrpu_ddr_example_activity_identity_r1_tb.sv'
    tb.write_text(text[:pos] + block + text[pos:], encoding='utf-8', newline='\n')
    record = dict(xpr_relocations=changes, original_xpr_sha256=digest(original / xpr.name),
                  relocated_xpr_sha256=digest(xpr), original_tb_sha256=digest(tb_original),
                  observed_tb_sha256=digest(tb), observer_block=block,
                  xci_unchanged_before_vivado=True, xci_sha256=digest(xci),
                  top_sha256=digest(project / 'imports/example_top.sv'),
                  tg_sha256=digest(project / 'imports/ddr4_v2_2_axi_tg_top.sv'),
                  stimuli_changed=False, hardware_sources_changed=False)
    (output / 'relocation.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
