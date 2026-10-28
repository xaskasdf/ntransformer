"""Review the CSR protocol unit's complete trace, coverage and completion markers."""
import argparse
import json
from pathlib import Path
import re

CASES = ["reset_map", "decode_rejections", "strobes_and_channel_skew",
         "atomic_descriptor", "cancel_and_irq", "success_metadata",
         "completion_faults", "concurrent_read_write", "reset_recovery"]


def review(text, seed, fp32=True):
    def require(ok, reason):
        if not ok:
            raise ValueError(reason)

    require(not re.search(r"(?im)^\s*(Fatal:|ERROR:|FAIL\b)|NTRPU_CSR_FAIL", text), "failure diagnostic")
    require(re.findall(r"^NTRPU_CSR_SEED seed=(\d+) fp32=(\d+)$", text, re.M)
            == [(str(seed), str(int(fp32)))], "seed/format")
    cases = re.findall(r"^NTRPU_CSR_CASE id=(\d+) name=(\w+)$", text, re.M)
    require(cases == [(str(i+1), name) for i, name in enumerate(CASES)], "case coverage/order")
    read_re = r"^NTRPU_CSR_READ id=(\d+) addr=([0-9a-f]{8}) data=([0-9a-f]{8}) expected=([0-9a-f]{8}) resp=(\d+) stall=(\d+)$"
    write_re = r"^NTRPU_CSR_WRITE id=(\d+) addr=([0-9a-f]{8}) data=([0-9a-f]{8}) strb=([0-9a-f]) resp=(\d+) order=(\d+) stall=(\d+)$"
    reads = re.findall(read_re, text, re.M)
    writes = re.findall(write_re, text, re.M)
    require(len(reads) == 298 and len(writes) == 412, "bus traffic coverage")
    for i, (ident, addr, data, expected, resp, stall) in enumerate(reads):
        require(int(ident) == i and data == expected and int(resp) in (0, 3), "read mismatch/sequence")
    for i, (ident, addr, data, strb, resp, order, stall) in enumerate(writes):
        require(int(ident) == i and int(resp) in (0, 2, 3) and int(order) in (0, 1, 2), "write sequence/response")
    require({int(w[3], 16) for w in writes} == set(range(16)), "WSTRB coverage")
    require({int(w[5]) for w in writes} == {0, 1, 2}, "AW/W order coverage")
    require({int(w[4]) for w in writes} == {0, 2, 3}, "write response coverage")
    require(any(int(w[1], 16) == 0x10000100 and w[4] == '3' for w in writes), "high address alias rejection")
    for b in range(8):
        for field in range(7):
            addr = 0x100+b*32+field*4
            require(any(int(w[1], 16) == addr and w[4] == '0' for w in writes), "buffer coverage")
    dones = re.findall(r"^NTRPU_CSR_DONE status=(\d+) fault=(\d+) cycles=(\d+)$", text, re.M)
    expected_done = [(7, 0), (0, 0)] + [(9 if f == 7 else 0, f) for f in range(1, 9)] + [(s, 0) for s in range(1, 9)] + [(1, 0)]
    require([(int(s), int(f)) for s, f, _ in dones] == expected_done, "completion coverage")
    require(all(int(c) > 0 for _, _, c in dones), "elapsed cycles")
    summaries = re.findall(r"^NTRPU_CSR_PASS (.+)$", text, re.M)
    require(len(summaries) == 1, "single terminal PASS")
    result = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", summaries[0])}
    for key, expected in dict(cases=9, reads=len(reads), writes=len(writes), resets=15,
                              jobs=20, done=19, cancel=1).items():
        require(result.get(key) == expected, "summary "+key)
    require(result.get('checks', 0) > 4000, "assertion coverage")
    for key, order in [('together', 0), ('aw_first', 1), ('w_first', 2)]:
        require(result.get(key) == sum(int(w[5]) == order for w in writes), key)
    require(result.get('b_stalls') == sum(int(w[6]) for w in writes), "B stalls")
    require(result.get('r_stalls') == sum(int(r[5]) for r in reads), "R stalls")
    require(result['b_stalls'] > 0 and result['r_stalls'] > 0, "backpressure coverage")
    result.update(seed=seed, fp32=bool(fp32), compute_peer='protocol_stub')
    return result


def mutations(text):
    """Negative evidence checks: each corruption must be rejected by review()."""
    return {
        'read_value': text.replace('data=4652544e expected=4652544e', 'data=4652544f expected=4652544e', 1),
        'missing_case': re.sub(r'^NTRPU_CSR_CASE id=4 .+\n', '', text, count=1, flags=re.M),
        'missing_read': re.sub(r'^NTRPU_CSR_READ id=1 .+\n', '', text, count=1, flags=re.M),
        'missing_write': re.sub(r'^NTRPU_CSR_WRITE id=1 .+\n', '', text, count=1, flags=re.M),
        'missing_done': re.sub(r'^NTRPU_CSR_DONE status=9 .+\n', '', text, count=1, flags=re.M),
        'missing_pass': re.sub(r'^NTRPU_CSR_PASS .+\n', '', text, count=1, flags=re.M),
        'fatal_after_pass': text+'\nFatal: injected evidence error\n',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fp64', action='store_true')
    args = parser.parse_args()
    print(json.dumps(review(args.log.read_text(encoding='utf-8'), args.seed, not args.fp64), indent=2))
