"""Check MMIO programming/completions against the independently reviewed FFN RAM trace."""
import argparse
import json
from pathlib import Path
import re
from check_ntrpu_ffn_memory import CASES, review as review_memory
from check_ntrpu_ffn_memory_fixture import load_fixture, review_log as review_fixture


def records(text, prefix):
    hexadecimal = {'addr', 'data', 'sequence', 'epoch', 'flags'}
    return [{k: int(v, 16 if k in hexadecimal else 10)
             for k, v in re.findall(r'(\w+)=([0-9a-f]+)', line)}
            for line in text.splitlines() if line.startswith(prefix+' ')]


def expected_writes(ident, dim, hidden, scenario, fp32, tiles=(2, 2, 2)):
    """Host ABI calculated separately from the SV driver, from the test contract."""
    spans = [((dim+31)//32)*64, hidden*((dim+31)//32)*34,
             hidden*((dim+31)//32)*34, dim*((hidden+31)//32)*34,
             ((hidden+7)//8)*64, ((hidden+7)//8)*64,
             ((hidden+31)//32)*64, ((dim*(4 if fp32 else 8)+63)//64)*64]
    config = [0xfffffffe, ident+600, ident+600, hidden*65536+dim, 2,
              sum(value << (8*i) for i, value in enumerate(tiles))]
    out = [(0x14, 1, 0)] + [(0x20+4*i, v, 0) for i, v in enumerate(config)]
    for b, span in enumerate(spans):
        base = 0x100000000+b*0x20000000
        size = 4032+span-int(scenario == 5 and b == 7)
        words = [base+4032, base, size]
        fields = [part for word in words for part in (word & 0xffffffff, word >> 32)] + [ident+600]
        out.extend((0x100+b*32+f*4, value, 0) for f, value in enumerate(fields))
    out.append((0x10, 1, 0))
    out.extend((0x20+4*i, 0, 0) for i in range(6))
    out.extend((0x100+b*32+4*f, 0, 0) for b in range(8) for f in range(7))
    if scenario == 3:
        out.append((0x10, 2, 0))
    if scenario != 4:
        out.extend([(0x10, 1, 2), (0x10, 4, 0)])
    return out


def review(text, seed, fp32=True, fixture=None):
    result = review_memory(text, seed, fp32) if fixture is None else review_fixture(text, fixture, seed, fp32)
    specs = CASES if fixture is None else [(256, 720, 0, 0)]*3
    tiles = (2, 2, 2) if fixture is None else (4, 4, 8)
    errors = result['errors']

    def need(ok, reason):
        if not ok:
            errors.append(reason)

    host = records(text, 'NTRPU_FFN_HOST_CASE')
    writes = records(text, 'NTRPU_FFN_HOST_WRITE')
    need([c.get('id') for c in host] == list(range(1, len(specs)+1)), 'Host case coverage/order')
    need(all(w.get('id') in range(1, len(specs)+1) for w in writes), 'MMIO case identity')
    memory = {c['id']: c for c in result['cases']}
    for ident, (row, spec) in enumerate(zip(host, specs), 1):
        dim, hidden, scenario, target = spec
        actual = [w for w in writes if w.get('id') == ident]
        need([(w.get('addr'), w.get('data'), w.get('resp')) for w in actual]
             == expected_writes(ident, dim, hidden, scenario, fp32, tiles), 'MMIO command trace '+str(ident))
        need({w.get('order') for w in actual} == {0, 1, 2}, 'AW/W order coverage')
        need(row.get('reset') == int(scenario == 4), 'Host reset coverage')
        if scenario == 4:
            need(set(row) == {'id', 'reset'}, 'Aborted host job must not publish completion')
            continue
        status = 6 if scenario in (1, 2) else 7 if scenario == 3 else 1 if scenario == 5 else 0
        stages = 4 if scenario == 0 else 0 if scenario == 5 else target
        need(row.get('status') == status and row.get('stages') == stages, 'MMIO completion status/stages')
        need(row.get('sequence') == 0xfffffffe and row.get('epoch') == ident+600 and row.get('flags') == 0,
             'MMIO completion identity/flags')
        mem = memory.get(ident, {})
        need(row.get('bytes') == mem.get('down_bytes') and row.get('stage_bytes') == mem.get('bytes'),
             'MMIO vs actual physical write counts')
        need(row.get('cancel') == int(scenario == 3), 'MMIO cancellation coverage')
        need(row.get('writes') == len(actual), 'MMIO write count')
        # Five initial reads; one per poll; nine completion, two counters, one idle.
        need(row.get('reads') == row.get('polls', -100)+17 and row.get('polls', 0) > 0, 'MMIO read count')
        need(0 < row.get('compute_cycles', 0) < row.get('total_cycles', 0) < (250000 if fixture is None else 1000000), 'Device elapsed range')
        need(0 < row.get('program_cycles', 0) < row.get('total_cycles', 0), 'Programming cycles')
        need(row.get('program_cycles', 0)+row.get('poll_cycles', 0) < row.get('total_cycles', 0), 'Host lifecycle timing')
        need(row.get('total_cycles', 0) > mem.get('cycles', 0), 'ACK follows consumer completion')
    result.update(status='FAIL' if errors else 'PASS', host_cases=host, mmio_writes=len(writes),
                  scope='AXI-Lite host BFM -> real CSR/FFN/MAC/SwiGLU RTL -> shared AXI RAM model; '+
                  ('synthetic weights' if fixture is None else 'pinned Brandon FFN weights, synthetic shared input')+
                  '; weights preloaded, no PCIe/DMA/DDR or full inference.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--fp64', action='store_true')
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    fixture = load_fixture(args.fixture) if args.fixture else None
    result = review(args.log.read_text(encoding='utf-8'), args.seed, not args.fp64, fixture)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
