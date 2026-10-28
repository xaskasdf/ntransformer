"""Review the registered-return variant, independent lease traces and integrated AXI transport."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_candidate_transport import require, replay, negative_controls
from check_ntrpu_transport_study import physical, performance, netlist, mutations
from check_ntrpu_transport_overlap import extra, negative
from check_ntrpu_overlap_physical import routed_extra
from check_ntrpu_overlap_bankplan import plan_review, physical_regions
from prepare_ntrpu_overlap_axi import integrated
from prepare_ntrpu_return_cut import allocator, receiver
from check_ntrpu_return_cut_allocator import replay as allocator_replay


def allocator_controls(text, profile):
    lines = text.splitlines(); trace = [i for i,l in enumerate(lines) if l.startswith('TRACE ')]
    changes = {}
    absent = lines.copy(); del absent[trace[len(trace)//2]]; changes['missing_edge'] = '\n'.join(absent)
    fields = [lines[i].split() for i in trace]
    # Forge a grant on the input-capture edge of a return.
    i = next(i for i,f in zip(trace,fields) if int(f[2],16) and int(f[5],16))
    bad = lines.copy(); row=bad[i].split(); row[8]='1'; bad[i]=' '.join(row); changes['capture_grant'] = '\n'.join(bad)
    # Flip a committed ownership bit and hide a reported invalid-return error.
    i = next(i for i,f in zip(trace,fields) if int(f[17],16)); bad=lines.copy(); row=bad[i].split(); row[17]=format(int(row[17],16)^1,'x'); bad[i]=' '.join(row); changes['ownership_corruption']='\n'.join(bad)
    i = next(i for i,f in zip(trace,fields) if int(f[11],16)); bad=lines.copy(); row=bad[i].split(); row[11]='0'; bad[i]=' '.join(row); changes['hidden_return_error']='\n'.join(bad)
    rejected=[]
    for name, candidate in changes.items():
        try: allocator_replay(candidate,profile,1)
        except (RuntimeError,ValueError,IndexError): rejected.append(name)
        else: raise ValueError('Undetected lease-ledger mutation '+name)
    return rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    ap.add_argument('--review', type=Path, required=True); a = ap.parse_args(); out = a.output
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = json.loads((out/'result.json').read_text()); files = json.loads((out/'FILES.json').read_text())
    require(record['experiment'] == 'registered_return_axi', 'Wrong experiment')
    require(all(sha(out/n) == h for n, h in files.items()), 'Evidence hashes')
    require(all(record[k] == record[k+'_after'] for k in ('sources', 'inputs', 'rtl_evidence', 'plan_evidence')), 'Inputs changed')
    stage = record['stage']; s = out/'suite'; console = (out/'console.log').read_text()
    if record['status'] != 'EXPORTED_PENDING_REVIEW':
        require(record['status'] == 'FAILED_OR_TIMEOUT', 'Invalid result')
        result = dict(status='TIMEOUT_DIAGNOSTIC' if record['timed_out'] else 'TOOL_FAILURE_DIAGNOSTIC',
                      log_tail=console.splitlines()[-25:])
    else:
        marker = 'NTRPU_RETURN_CUT_COMPLETE '
        require(marker+stage in console and not re.search(r'^ERROR:', console, re.M), 'Tool completion')
        if stage in ('rtl', 'synth', 'netlist'):
            props = dict(l.split('\t', 1) for l in (s/'ip_config.tsv').read_text().splitlines())
            expected = dict(PROTOCOL='AXI4', READ_WRITE_MODE='WRITE_ONLY', ADDR_WIDTH='64', DATA_WIDTH='512', ID_WIDTH='4', REG_AW='1', REG_W='1', REG_B='1')
            require(all(props['CONFIG.'+k] == v for k, v in expected.items()), 'IP configuration')
            fixture = json.loads((s/'generated/AXI_INTEGRATION.json').read_text())
            root = Path(__file__).resolve().parents[1]
            original = (root/'fpga/rtl/ntrpu_candidate_transport.sv').read_text()
            require(sha(root/'fpga/rtl/ntrpu_candidate_transport.sv') == record['sources']['fpga/rtl/ntrpu_candidate_transport.sv'], 'Reviewer reference differs')
            require(hashlib.sha256(original.encode()).hexdigest() == fixture['reference_transport_sha256'], 'Reference transport provenance')
            require((s/'generated/ntrpu_candidate_transport.sv').read_text() == integrated(original), 'Unexpected transport changes')
            require(sha(s/'generated/ntrpu_candidate_transport.sv') == fixture['integrated_transport_sha256'], 'Integrated hash')
            cut = json.loads((s/'generated/RETURN_CUT.json').read_text())
            require(cut['return_capture_cycles'] == 1 and cut['block_admission_during_capture'] and
                    cut['block_admission_during_validation'] and not cut['same_edge_reuse'], 'Return contract changed')
            original_allocator = root/'fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv'
            require(sha(original_allocator) == record['sources']['fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv'] == cut['original_allocator_sha256'], 'Allocator source identity')
            require((s/'generated/ntrpu_buffer_pool_allocator_return_cut.sv').read_text() == allocator(original_allocator.read_text()), 'Unexpected allocator changes')
            for n in ('ntrpu_destination_candidate', 'ntrpu_peer_multibuffer_offer', 'ntrpu_peer_multibuffer_reserved', 'ntrpu_candidate_dispatch'):
                baseline = out/'rtl_evidence'/(n+'.sv'); generated = s/'generated'/(n+'.sv')
                require(sha(baseline) == fixture['logic_sha256'][n], 'Original compact logic changed')
                expected = receiver(baseline.read_text()) if n == 'ntrpu_peer_multibuffer_offer' else baseline.read_text()
                require(generated.read_text() == expected and sha(generated) == cut['logic_sha256'][n], 'Unexpected transport logic')
            for n, digest in cut['logic_sha256'].items():
                generated = s/'generated'/(n+'.sv')
                source = generated if generated.exists() else root/'fpga/rtl'/(n+'.sv')
                require(sha(source) == digest, 'Full logic manifest mismatch: '+n)
            if stage == 'synth':
                require(json.loads((out/'inputs/AXI_INTEGRATION.json').read_text()) == fixture, 'Synth differs from AXI fixture')
                require(json.loads((out/'inputs/RETURN_CUT.json').read_text()) == cut, 'Synth differs from staged-return fixture')
        if stage == 'rtl':
            text = (s/'protocol.log').read_text(); other = (s/'extra.log').read_text(); perf = (s/'perf.log').read_text()
            require(not any(re.search(r'^\s*(Fatal:|ERROR:|FAIL )', x, re.M) for x in (text, other, perf)), 'Simulation failure')
            lines = [l for l in text.splitlines() if l.startswith('TE ')]
            result = replay(lines); result['negative_controls'] = negative_controls(lines)
            result['extra'] = extra(other); result['extra_negative_controls'] = negative(other, extra, ['TE W ', 'TE CPL ', 'OV LAST '])
            result['performance'] = performance(perf); result['performance_negative_controls'] = negative(perf, performance, ['TE IN ', 'TE W ', 'TE CPL ', 'PF RELEASE '])
            result['allocator'] = {}
            for name, profile in cut['profiles'].items():
                log = (s/(name+'.log')).read_text()
                checked = allocator_replay(log, profile, 1)
                checked['negative_controls'] = allocator_controls(log, profile)
                result['allocator'][name] = checked
            result['status'] = 'RETURN_CUT_RTL_PROTOCOL_PERFORMANCE_PASS'
        elif stage == 'netlist':
            text = (s/'protocol.log').read_text(); other = (s/'extra.log').read_text()
            require(not any(re.search(r'^\s*(Fatal:|ERROR:|FAIL )', x, re.M) for x in (text, other)), 'Netlist simulation failure')
            result = netlist(text); result['extra'] = routed_extra(other)
            result['trace_negative_controls'] = mutations(text, netlist)
            result['extra_negative_controls'] = negative(other, routed_extra, ['TN W ', 'TN FETCH ', 'TN CPL ', 'TN MATCH '])
            comp = cut['comparison']
            require(comp == dict(netlist_sha256=record['inputs']['physical_funcsim.v'], latency_edges=0, startup_ns=150, sdf=False, force=False, reversible_renaming=True), 'Comparison scope')
        else:
            result = physical(out, stage); util = (s/'utilization.rpt').read_text()
            result['resources'] = {}
            for key in ('CLB LUTs', 'CLB Registers', 'Block RAM Tile', 'RAMB36/FIFO*', 'RAMB18'):
                m = re.search(r'^\|\s*'+re.escape(key)+r'\*?\s*\|\s*([0-9.]+)\s*\|', util, re.M)
                if m: result['resources'][key] = float(m[1])
            if stage in ('place', 'route'):
                for p in (out/'inputs/constraints_after.xdc', s/'constraints_before.xdc', s/'constraints_after.xdc'):
                    require(not re.search(r'^\s*(create_pblock|resize_pblock|add_cells_to_pblock)\b', p.read_text(), re.M), 'Unexpected floorplan constraints')
                result['floorplan'] = 'NO_PBLOCKS_VERIFIED'
    result.update(stage=stage, seconds=record['seconds'], checker_sha256=sha(Path(__file__)),
                  scope='Same-clock AXI write slices; external ports, PCIe, HBM and CDC unqualified')
    a.review.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'timing_checks', 'drc', 'performance', 'log_tail')}))


if __name__ == '__main__':
    main()
