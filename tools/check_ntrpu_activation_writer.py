"""Review raw-record bytes, AXI completion and explicit fault/reset coverage."""
import re

CASES = [(1, 0, 0, 0), (7, 4032, 0, 0), (8, 0, 0, 0),
         (16, 4032, 0, 0), (31, 0, 0, 0), (32, 4032, 0, 0),
         (0, 0, 1, 1), (33, 0, 1, 1)]
STATUSES = {2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1,
            **dict.fromkeys(range(9, 16), 4), **dict.fromkeys(range(20, 23), 3),
            **dict.fromkeys(range(23, 26), 8), **dict.fromkeys(range(29, 37), 7),
            42: 8, 43: 7, 46: 4}
CASES += [(31 if mode == 15 else 32, 4032, mode, STATUSES.get(mode, 0))
          for mode in range(2, 47)]


def expected_record(case_id, elements, mode):
    """Byte-level ABI oracle; never reinterpret the INT8 field as doubles."""
    data = bytearray(64)
    for index in range(min(elements, 32)):
        data[index] = (17 * case_id + 31 * index) % 256
    scale = {16: 0x4780204070E1C387, 17: 0x8000000000000000,
             18: 1, 19: 0x4070100000000000}.get(mode, 0x3FE0000000000000)
    if mode == 19:
        data[:32] = bytes([127] + [255] * 31)
    data[32:40] = scale.to_bytes(8, "little")
    return bytes(data)


def review(text, seed, netlist=False):
    errors = []

    def need(ok, why):
        if not ok:
            errors.append(why)

    watchdog = 65536 if netlist else 32
    need(not re.search(r"^\s*(Fatal:|ERROR:|FAIL\b)", text, re.M | re.I), "Executed failure")
    need(re.findall(r"^NTRPU_ACT_WRITER_SEED seed=(\d+)\s*$", text, re.M) == [str(seed)], "Seed")
    need(re.findall(r"^NTRPU_ACT_WRITER_CONFIG watchdog=(\d+)\s*$", text, re.M) == [str(watchdog)], "Watchdog")
    rows = [{k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
            for line in text.splitlines() if line.startswith("NTRPU_ACT_WRITER_CASE ")]
    need([r.get("id") for r in rows] == list(range(1, 54)), "Case identities")
    writes = re.findall(r"^NTRPU_ACT_WRITE id=(\d+) data=([0-9a-f]{128})\s*$", text, re.M)
    need(len(writes) == len(re.findall(r"^NTRPU_ACT_WRITE ", text, re.M)), "Write record format")
    need(len({ident for ident, _ in writes}) == len(writes), "Unique write identity")
    expected_ids = []
    for row, contract in zip(rows, CASES):
        elements, offset, mode, status = contract
        ident = row.get("id")
        need(set(row) == {"id", "elements", "offset", "scenario", "status", "reset", "cycles",
                          "aw", "w", "b", "bytes", "stored", "cancel"}, "Case fields")
        need(tuple(row.get(k) for k in ["elements", "offset", "scenario", "status"]) == contract,
             "Case contract " + str(ident))
        need(row.get("reset") == int(37 <= mode <= 41), "Reset contract")
        need(row.get("cancel") == int(status == 7), "Cancel contract")
        need(0 < row.get("cycles", 0) < (80000 if netlist else 2000), "Latency")
        traffic = (1, 1, 1, 64, 64)
        if status in {1, 2, 4} or 29 <= mode <= 32:
            traffic = (0, 0, 0, 0, 0)
        traffic = {37: (0, 0, 0, 0, 0), 38: (0, 1, 0, 64, 0),
                   39: (1, 0, 0, 0, 0), 40: (1, 1, 0, 64, 64)}.get(mode, traffic)
        need(tuple(row.get(k) for k in ["aw", "w", "b", "bytes", "stored"]) == traffic,
             "AXI/private memory " + str(ident))
        if status == 8:
            need(row.get("cycles", 0) >= watchdog + 48, "Timeout stimulus")
        if traffic[1]:
            expected_ids.append(str(ident))
            found = [value for key, value in writes if key == str(ident)]
            need(found == [expected_record(ident, elements, mode)[::-1].hex()], "Independent ABI bytes " + str(ident))
    need([ident for ident, _ in writes] == expected_ids, "Write order/count")
    markers = re.findall(r"^NTRPU_ACT_WRITER_PASS (.+)$", text, re.M)
    need(len(markers) == 1, "Suite marker")
    summary = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", markers[0])} if markers else {}
    need(summary.get("cases") == 53 and summary.get("resets") == 5 and summary.get("cancels") == 9,
         "Suite cases/reset/cancel")
    for key in ["aw", "w", "b", "stored"]:
        need(summary.get(key) == sum(row.get(key, 0) for row in rows), "Total " + key)
    for key in ["aw_stalls", "w_stalls", "done_stalls", "aw_before_w", "w_before_aw"]:
        need(summary.get(key, 0) > 0, "Coverage " + key)
    return dict(status="FAIL" if errors else "PASS", errors=errors, seed=seed,
                cases=rows, records_checked=len(writes), summary=summary)
