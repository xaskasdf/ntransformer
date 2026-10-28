"""Review host-loaded FFN memory images, contention traffic and output readback."""
import argparse
import json
from pathlib import Path
import re
from check_ntrpu_ffn_host import records, review as review_host
from check_ntrpu_ffn_memory_fixture import load_fixture


def bursts(size, offset=0):
    """Four-beat host requests, split at the physical 4 KiB page boundary."""
    while offset < size:
        beats = min(4, (size-offset)//64, (4096-(4032+offset) % 4096)//64)
        yield offset, beats
        offset += beats*64


def review(text, path, seed=1, host_reviewer=review_host):
    fixture = load_fixture(path)
    result = host_reviewer(text, seed, True, fixture)
    errors = result['errors']

    def need(ok, why):
        if not ok:
            errors.append(why)

    # Existing MMIO and independent scalar numerical reviewers cover every
    # core output byte, native Q8 agreement, descriptor and retained completion.
    expected_loads = []
    expected_aw = []
    expected_b = []
    expected_ar = []
    expected_reads = []
    for ident in (1, 2, 3):
        images = {b: bytes.fromhex((path/(f'job-{ident:03d}-input.hex' if b == 0 else f'weight-{b}.hex')).read_text())[4032:]
                  for b in range(4)}
        final = bytes.fromhex((path/f'job-{ident:03d}-final-fp32.hex').read_text())
        for b, raw in images.items():
            for off in range(0, len(raw), 64):
                expected_loads.append(dict(id=ident, buffer=b, offset=off, data=int.from_bytes(raw[off:off+64], 'little')))
            for off, beats in bursts(len(raw)):
                expected_aw.append(dict(id=ident, host=1, addr=0x100000000+b*0x20000000+4032+off, beats=beats))
                expected_b.append(dict(id=ident, buffer=b, offset=off, beats=beats))
        requests = []
        for k in range(64):
            off = 64*((k*7) % 256)
            beats = min(4, (4096-(4032+off) % 4096)//64)
            requests.append((1, off, beats))
        requests.extend((7, off, beats) for off, beats in bursts(1024))
        for b, off, beats in requests:
            expected_ar.append(dict(id=ident, host=1, addr=0x100000000+b*0x20000000+4032+off, beats=beats))
            raw = final if b == 7 else images[b]
            for beat in range(beats):
                start = off+beat*64
                expected_reads.append(dict(id=ident, buffer=b, offset=start, final=int(b == 7),
                                           data=int.from_bytes(raw[start:start+64], 'little')))
    loads = records(text, 'NTRPU_TRANSFER_LOAD')
    source_w = records(text, 'NTRPU_TRANSFER_HOST_W')
    reads = records(text, 'NTRPU_TRANSFER_HOST_R')
    all_ar = records(text, 'NTRPU_TRANSFER_AR')
    all_aw = records(text, 'NTRPU_TRANSFER_AW')
    need(loads == expected_loads, 'every physical input/weight byte came through host AXI')
    need(source_w == expected_loads, 'host W bytes and ordering')
    need(reads == expected_reads, 'concurrent weight reads / final host readback match pinned bytes')
    need([r for r in all_ar if r.get('host') == 1] == expected_ar, 'physical host AR addresses/lengths/pages')
    need([r for r in all_aw if r.get('host') == 1] == expected_aw, 'physical host AW addresses/lengths/pages')
    need(records(text, 'NTRPU_TRANSFER_HOST_B') == expected_b, 'host acknowledges every load burst')
    cases = records(text, 'NTRPU_TRANSFER_CASE')
    need([r.get('id') for r in cases] == [1, 2, 3], 'transfer case identities')
    for row in cases:
        ident = row['id']
        arcs = [r for r in expected_ar if r['id'] == ident]
        aws = [r for r in expected_aw if r['id'] == ident]
        need(row.get('host_ar') == len(arcs) and row.get('host_r') == sum(r['beats'] for r in arcs), 'host read accounting')
        need(row.get('host_aw') == row.get('host_b') == len(aws) and row.get('host_w') == 9256, 'host write accounting')
        need(row.get('load_bytes') == 592384 and row.get('result_bytes') == 1024, 'transfer byte accounting')
        need(all(row.get(k, 0) > 0 for k in ('load_cycles', 'control_compute_cycles', 'result_cycles', 'contention')), 'load/compute/readback timing and actual contention')
        need(len([r for r in all_ar if r.get('id') == ident and r.get('host') == 0]) == 418 and
             len([r for r in all_aw if r.get('id') == ident and r.get('host') == 0]) == 82, 'compute physical address counts')
    completed = set()
    starts = set()
    for line in text.splitlines():
        if line.startswith('NTRPU_TRANSFER_COMPLETE '):
            row = records(line, 'NTRPU_TRANSFER_COMPLETE')[0]
            need(row.get('id') not in completed and row.get('clock', 0) > 0, 'single valid completion per job')
            completed.add(row['id'])
        elif line.startswith('NTRPU_FFN_HOST_WRITE '):
            row = records(line, 'NTRPU_FFN_HOST_WRITE')[0]
            if row.get('addr') == 0x10 and row.get('data') == 1 and row.get('resp') == 0:
                starts.add(row['id'])
        elif line.startswith('NTRPU_TRANSFER_LOAD '):
            row = records(line, 'NTRPU_TRANSFER_LOAD')[0]
            need(row['id'] not in starts, 'immutable weights loaded before START')
        elif line.startswith('NTRPU_TRANSFER_HOST_R '):
            row = records(line, 'NTRPU_TRANSFER_HOST_R')[0]
            need((row['id'] in completed) == bool(row.get('final')), 'output publication after completion / background reads before completion')
    need(completed == starts == {1, 2, 3}, 'host lifecycle coverage')
    need(records(text, 'NTRPU_FFN_TRANSFER_PASS') == [dict(cases=3, load_bytes=1777152, result_bytes=3072)], 'terminal transfer PASS')
    result.update(status='FAIL' if errors else 'PASS', transfer_cases=cases,
                  physical_load_bytes=len(loads)*64, host_result_bytes=sum(64 for r in reads if r.get('final')),
                  scope='Host AXI -> registered arbiter -> empty behavioral RAM -> real FFN/HLS, MMIO completion, host output readback; '+
                        'three real Brandon layer0 weights with synthetic inputs; no PCIe/DMA IP, DDR, CDC, driver or full inference.')
    return result


def mutations(text):
    def flip(prefix, field):
        return re.sub(r'(^'+prefix+r'[^\n]*\b'+field+r'=)([0-9a-f]+)',
                      lambda m: m[1]+format(int(m[2], 16)^1, 'x'), text, count=1, flags=re.M)
    return {
        'seed': text.replace('NTRPU_FFN_MEMORY_SEED seed=1', 'NTRPU_FFN_MEMORY_SEED seed=2', 1),
        'fatal': text+'\nFatal: injected evidence fault\n',
        'missing_pass': text.replace('NTRPU_FFN_TRANSFER_PASS ', 'REMOVED_PASS ', 1),
        'loaded_byte': flip('NTRPU_TRANSFER_LOAD', 'data'),
        'host_source_byte': flip('NTRPU_TRANSFER_HOST_W', 'data'),
        'physical_address': flip('NTRPU_TRANSFER_AW', 'addr'),
        'readback': flip('NTRPU_TRANSFER_HOST_R[^\n]*final=1', 'data'),
        'background_read': flip('NTRPU_TRANSFER_HOST_R[^\n]*final=0', 'data'),
        'missing_completion': re.sub(r'^NTRPU_TRANSFER_COMPLETE id=1[^\n]+\n', '', text, count=1, flags=re.M),
        'missing_load_ack': re.sub(r'^NTRPU_TRANSFER_HOST_B id=1[^\n]+\n', '', text, count=1, flags=re.M),
        'core_output': flip('NTRPU_FFN_MEMORY_W', 'data'),
        'mmio_address': text.replace('addr=00000104 data=00000001', 'addr=00000104 data=00000000', 1),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    result = review(args.log.read_text(encoding='utf-8'), args.fixture, args.seed)
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'PASS')
