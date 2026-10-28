"""Freeze rev2, replace only its wrapper, and add a cycle-matched RTL reference."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(out, donor):
    source = donor / 'rtl1/repo'
    result = json.loads((donor / 'rtl1/output/result.json').read_text())
    if result['status'] != 'REV2_BEHAVIOR_PASS':
        raise ValueError('Unqualified behavioral donor')
    for name, sha in result['sources'].items():
        if digest(source / name) != sha:
            raise ValueError('Changed donor: ' + name)
    shutil.copytree(source, out)
    top = out / 'fpga/experimental/ntrpu_hbm_dma_bridge.sv'
    reference = top.read_text().replace('module ntrpu_hbm_dma_bridge (',
                                       'module ntrpu_hbm_dma_bridge_reference (')
    (top.parent / 'ntrpu_hbm_dma_bridge_reference.sv').write_text(reference, encoding='utf-8')
    shutil.copyfile(ROOT / 'fpga/experimental/hbm_dma_rev3/ntrpu_hbm_dma_bridge.sv', top)
    for name in ('fpga/tb/ntrpu_hbm_dma_rev3_cases.svh', 'tools/prepare_ntrpu_hbm_dma_rev3.py',
                 'tools/run_ntrpu_hbm_dma_rev3.py', 'tools/ntrpu_hbm_dma_rev3_vectors.py'):
        shutil.copyfile(ROOT / name, out / name)
    # Use the verified elaborated port inventory, including packed struct bits.
    ports = donor / 'synth1/output/reports/ports.tsv'
    files = json.loads((donor / 'synth1/output/FILES.json').read_text())
    if digest(ports) != files['reports/ports.tsv']:
        raise ValueError('Changed port inventory')
    counts = Counter()
    directions = {}
    for row in csv.DictReader(ports.read_text().splitlines(), delimiter='\t'):
        name = row['name'].split('[')[0]
        counts[name] += 1
        directions[name] = row['direction']
    outputs = sorted(n for n in counts if directions[n] == 'OUT')
    declarations = '\n'.join(f'    wire [{counts[n]-1}:0] reference_{n};' for n in outputs)
    connections = ',\n'.join('        .' + n + '(' + ('reference_' if directions[n] == 'OUT' else '') + n + ')'
                              for n in sorted(counts))
    observed = '{' + ','.join(outputs) + '}'
    expected = '{' + ','.join('reference_' + n for n in outputs) + '}'
    monitor = declarations + '\n    ntrpu_hbm_dma_bridge_reference reference_dut(\n' + connections + '\n    );\n'
    monitor += ('    always @(posedge clk or posedge dma_clk) begin\n'
                '        #0.001;\n'
                f'        if(rst_n && dma_rst_n && {observed} !== {expected})\n'
                '            $fatal(1,"REV3 CYCLE_OUTPUT_MISMATCH case=%0d",case_id);\n'
                '        if(rst_n && dut.configured && !cfg_valid && submit_valid &&\n'
                '           dut.admission_error !== reference_dut.admission_error)\n'
                '            $fatal(1,"REV3 ADMISSION_MISMATCH case=%0d",case_id);\n'
                '    end\n')
    bench = out / 'fpga/tb/tb_ntrpu_hbm_dma_bridge.sv'
    text = bench.read_text()
    text = text.replace('    ntrpu_hbm_dma_bridge dut(.*);', '    ntrpu_hbm_dma_bridge dut(.*);\n' + monitor)
    needle = '    initial begin\n        if($test$plusargs("revision_only"))'
    if text.count(needle) != 1:
        raise ValueError('Unexpected frozen bench entrypoint')
    text = text.replace(needle, '    `include "ntrpu_hbm_dma_rev3_cases.svh"\n'
                        '    initial begin\n'
                        '        if($test$plusargs("bounds_only"))begin revision3_bounds();$finish;end\n'
                        '        if($test$plusargs("rev3_only"))begin revision3_transactions();$finish;end\n'
                        '        if($test$plusargs("revision_only"))')
    bench.write_text(text, encoding='utf-8')
    # Existing runners use one fixed compilation list. Add only the reference
    # and include directory to their private copy; their checks stay unchanged.
    runner = out / 'tools/run_ntrpu_hbm_dma_bridge_rtl.py'
    text = runner.read_text()
    text = text.replace("'fpga/experimental/ntrpu_hbm_dma_bridge.sv',", "'fpga/experimental/ntrpu_hbm_dma_bridge.sv',\n"
                        "       'fpga/experimental/ntrpu_hbm_dma_bridge_reference.sv',")
    text = text.replace("[compiler, '-g2012', '-Wall',", "[compiler, '-g2012', '-Wall', '-I', ROOT / 'fpga/tb',")
    runner.write_text(text, encoding='utf-8')
    prepared = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file()}
    (out / 'PREPARED_REV3.json').write_text(json.dumps(dict(donor=result['sources'], prepared=prepared,
            wrapper_only=True, reference_matches_all_output_ports=outputs), indent=2) + '\n')
    return dict(output=str(out), files=len(prepared))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--donor', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(prepare(args.output.resolve(), args.donor.resolve())))
