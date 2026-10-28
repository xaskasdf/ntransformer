"""Bounded cycle scheduler for buffer sizing, NOT an AXI/PCIe/RTL simulator.

One ingress beat/cycle, complete frames before issue, AW before/same-cycle W,
one W stream per destination, ordered B, one result consumed per cycle.
Ideal arbitration and same-cycle credit reuse; no CDC, RAM pipeline or errors.
Every beat carries a unique marker to detect reuse, ordering and loss errors.
"""

from collections import Counter, deque
from dataclasses import dataclass
from itertools import islice


@dataclass(frozen=True)
class Config:
    channels: int = 2
    dedicated: int = 2
    reserve: int = 4
    bank_bytes: int = 4096
    beat_bytes: int = 64
    outstanding: int = 4
    descriptors: int = 64
    result_slots: int = 64
    b_delay: int = 32
    read_ports: int = 2
    reserve_read_ports: int = 2
    reserve_quota: int | None = None
    ingress: str = "eligible"
    release_on_b: bool = False
    clock_mhz: int = 200
    ingress_window: int | None = None

    def __post_init__(self):
        for name in ("channels", "bank_bytes", "beat_bytes", "outstanding",
                     "descriptors", "result_slots", "read_ports",
                     "reserve_read_ports", "clock_mhz"):
            if getattr(self, name) <= 0:
                raise ValueError(name)
        if min(self.dedicated, self.reserve, self.b_delay) < 0:
            raise ValueError("negative capacity/delay")
        if self.dedicated * self.channels + self.reserve == 0:
            raise ValueError("no banks")
        if self.bank_bytes % self.beat_bytes or self.ingress not in ("ordered", "eligible"):
            raise ValueError("bank alignment or ingress policy")
        if self.reserve_quota is not None and self.reserve_quota < 0:
            raise ValueError("reserve_quota")
        if self.ingress_window is not None and (
                not isinstance(self.ingress_window, int) or self.ingress_window < 1 or
                self.ingress != "eligible"):
            raise ValueError("ingress_window requires eligible ingress and positive entries")


@dataclass(frozen=True)
class Job:
    tag: int
    channel: int
    beats: int
    available: int = 0


@dataclass(frozen=True)
class Stalls:
    # Half-open (channel, start, end) intervals; channel -1 means all.
    aw: tuple = ()
    w: tuple = ()
    b: tuple = ()
    result: tuple = ()  # (start, end)
    w_period: int = 0
    w_closed: int = 0

    def __post_init__(self):
        if not 0 <= self.w_closed <= self.w_period:
            raise ValueError("periodic pause")
        for intervals in (self.aw, self.w, self.b):
            if any(ch < -1 or start < 0 or end <= start for ch, start, end in intervals):
                raise ValueError("pause interval")
        if any(start < 0 or end <= start for start, end in self.result):
            raise ValueError("result pause")

    def ready(self, kind, channel, cycle):
        if kind == "w" and self.w_period and cycle % self.w_period < self.w_closed:
            return False
        return not any(ch in (-1, channel) and start <= cycle < end
                       for ch, start, end in getattr(self, kind))


def simulate(config, jobs, stalls=Stalls(), *, max_cycles=100000, audit=False):
    """Return metrics + transaction trace, including incomplete bounded runs.

    Eligible ingress assumes the upstream scheduler can select a destination
    before starting a frame. Ordered ingress cannot skip a blocked head.
    ingress_window bounds visible unadmitted descriptors in original order;
    None keeps the original ideal per-destination queues. Payload must remain
    selectively fetchable upstream; this is not reordering of an inline stream.
    Reserved descriptors live admission->B, results admission->consumption.
    """
    c = config
    if not __debug__:
        raise RuntimeError("run without -O; model invariants require assertions")
    jobs = list(jobs)
    if max_cycles <= 0 or not jobs:
        raise ValueError("positive budget and nonempty workload required")
    if len({j.tag for j in jobs}) != len(jobs):
        raise ValueError("duplicate tag")
    if any(j.tag < 0 or not 0 <= j.channel < c.channels or j.available < 0 or
           not 0 < j.beats <= c.bank_bytes // c.beat_bytes for j in jobs):
        raise ValueError("invalid job")
    queues = [deque() for _ in range(c.channels)]
    for j in jobs:
        queues[j.channel].append(j)
    ordered = deque(jobs)
    homes = [p for p in range(c.channels) for _ in range(c.dedicated)] + [None] * c.reserve
    owners = [None] * len(homes)
    data = [None] * len(homes)
    reserved_banks = [0] * c.channels
    descriptor_use = [0] * c.channels
    result_use = [0] * c.channels
    outstanding = [0] * c.channels
    ready = [deque() for _ in range(c.channels)]
    active = [None] * c.channels
    pending = [deque() for _ in range(c.channels)]
    results = deque()
    records = {}
    fill = None
    ingress_rr = output_rr = 0
    counters = Counter()
    peak_banks = peak_reserve = peak_outstanding = peak_results = 0
    bank_cycles = reserve_cycles = useful_capacity_cycles = 0
    completed = 0
    per_channel = [0] * c.channels

    def free_bank(rec):
        bank = rec["bank"]
        assert owners[bank] == rec["job"].tag, "bank ownership lost"
        owners[bank] = None
        if homes[bank] is None:
            reserved_banks[rec["job"].channel] -= 1
        rec["bank_freed"] = cycle

    def admission(job):
        p = job.channel
        if result_use[p] >= c.result_slots:
            return None, "result_credit"
        if descriptor_use[p] >= c.descriptors:
            return None, "descriptor_credit"
        for bank, home in enumerate(homes):
            if home == p and owners[bank] is None:
                return bank, None
        if c.reserve_quota is None or reserved_banks[p] < c.reserve_quota:
            for bank, home in enumerate(homes):
                if home is None and owners[bank] is None:
                    return bank, None
        return None, "bank_or_quota"

    for cycle in range(max_cycles):
        # Consumer sees only B responses accepted in previous cycles.
        if results and not any(start <= cycle < end for start, end in stalls.result):
            tag = results.popleft()
            rec = records[tag]
            rec["complete"] = cycle
            result_use[rec["job"].channel] -= 1
            completed += 1

        # At most one ordered B per destination/cycle, at least one cycle after WLAST.
        for p in range(c.channels):
            if pending[p] and pending[p][0][0] <= cycle:
                if stalls.ready("b", p, cycle):
                    _, tag = pending[p].popleft()
                    rec = records[tag]
                    rec["b"] = cycle
                    outstanding[p] -= 1
                    descriptor_use[p] -= 1
                    if c.release_on_b:
                        free_bank(rec)
                    results.append(tag)
                else:
                    counters["b_stalled_channel_cycles"] += 1

        # AW issue; W of an earlier burst must finish first on this destination.
        for p in range(c.channels):
            if active[p] is None and ready[p]:
                if outstanding[p] >= c.outstanding:
                    counters["outstanding_stalled_channel_cycles"] += 1
                elif not stalls.ready("aw", p, cycle):
                    counters["aw_stalled_channel_cycles"] += 1
                else:
                    tag = ready[p].popleft()
                    active[p] = tag
                    records[tag]["aw"] = cycle
                    outstanding[p] += 1

        used_ports = used_reserve_ports = 0
        for offset in range(c.channels):
            p = (output_rr + offset) % c.channels
            if active[p] is None:
                continue
            rec = records[active[p]]
            bank, job = rec["bank"], rec["job"]
            is_reserve = homes[bank] is None
            if not stalls.ready("w", p, cycle):
                counters["w_stalled_channel_cycles"] += 1
                continue
            if used_ports >= c.read_ports or (is_reserve and used_reserve_ports >= c.reserve_read_ports):
                counters["read_port_stalled_channel_cycles"] += 1
                continue
            used_ports += 1
            used_reserve_ports += int(is_reserve)
            index = len(rec["received"])
            assert owners[bank] == job.tag
            marker = data[bank][index]
            assert marker == (job.tag, index), "payload corruption/reordering"
            rec["received"].append(marker)
            per_channel[p] += c.beat_bytes
            if index == 0:
                rec["first_w"] = cycle
            if index + 1 == job.beats:
                rec["last_w"] = cycle
                active[p] = None
                pending[p].append((cycle + max(1, c.b_delay), job.tag))
                if not c.release_on_b:
                    free_bank(rec)
        output_rr = (output_rr + 1) % c.channels

        # Non-preemptive ingress frame. Reusing a bank this cycle is idealized.
        if fill is None:
            candidates = ([ordered[0]] if ordered and ordered[0].available <= cycle else [])
            if c.ingress == "eligible":
                candidates = []
                visible = (None if c.ingress_window is None else
                           {j.tag for j in islice(ordered, c.ingress_window)})
                for offset in range(c.channels):
                    p = (ingress_rr + offset) % c.channels
                    if (queues[p] and queues[p][0].available <= cycle and
                            (visible is None or queues[p][0].tag in visible)):
                        candidates.append(queues[p][0])
            blocked = set()
            for job in candidates:
                bank, reason = admission(job)
                if reason:
                    blocked.add(reason)
                    continue
                if c.ingress == "ordered":
                    ordered.popleft()
                else:
                    queues[job.channel].popleft()
                    if c.ingress_window is not None:
                        ordered.remove(job)
                ingress_rr = (job.channel + 1) % c.channels
                fill = job.tag
                owners[bank] = job.tag
                data[bank] = []
                reserved_banks[job.channel] += int(homes[bank] is None)
                descriptor_use[job.channel] += 1
                result_use[job.channel] += 1
                records[job.tag] = {"job": job, "bank": bank, "admitted": cycle, "received": []}
                break
            if candidates and fill is None:
                counters["ingress_blocked_cycles"] += 1
                for reason in blocked:
                    counters["ingress_" + reason + "_cycles"] += 1
        if fill is not None:
            rec = records[fill]
            bank, job = rec["bank"], rec["job"]
            data[bank].append((job.tag, len(data[bank])))
            counters["ingress_beats"] += 1
            if len(data[bank]) == job.beats:
                rec["filled"] = cycle
                ready[job.channel].append(job.tag)
                fill = None

        occupied = sum(owner is not None for owner in owners)
        reserve_use = sum(reserved_banks)
        peak_banks = max(peak_banks, occupied)
        peak_reserve = max(peak_reserve, reserve_use)
        peak_outstanding = max(peak_outstanding, sum(outstanding))
        peak_results = max(peak_results, sum(result_use))
        bank_cycles += occupied
        reserve_cycles += reserve_use
        useful_capacity_cycles += sum(records[tag]["job"].beats * c.beat_bytes
                                      for tag in owners if tag is not None)
        assert all(0 <= n <= c.outstanding for n in outstanding)
        assert all(0 <= n <= c.descriptors for n in descriptor_use)
        assert all(0 <= n <= c.result_slots for n in result_use)
        if audit:
            live_tags = [tag for tag in owners if tag is not None]
            assert len(live_tags) == len(set(live_tags)), "double allocation"
            for p in range(c.channels):
                mine = [r for r in records.values() if r["job"].channel == p]
                assert descriptor_use[p] == sum("b" not in r for r in mine)
                assert result_use[p] == sum("complete" not in r for r in mine)
                assert outstanding[p] == sum("aw" in r and "b" not in r for r in mine)
                assert reserved_banks[p] == sum(homes[b] is None and tag is not None
                    and records[tag]["job"].channel == p for b, tag in enumerate(owners))
        if completed == len(jobs):
            assert not any(owner is not None for owner in owners)
            assert not any(descriptor_use + result_use + outstanding)
            break

    elapsed = cycle + 1
    latencies = sorted(r["complete"] - r["admitted"] + 1 for r in records.values() if "complete" in r)
    metrics = {
        "drained": completed == len(jobs), "cycles": elapsed,
        "offered": len(jobs), "admitted": len(records), "completed": completed,
        "completed_by_channel": [sum(r["job"].channel == p and "complete" in r
                                      for r in records.values()) for p in range(c.channels)],
        "bytes_by_channel": per_channel, "payload_bytes": sum(per_channel),
        "gbps_including_fill_and_drain": sum(per_channel) / elapsed * c.clock_mhz / 1000,
        "last_ingress_cycle": max((r.get("filled", -1) for r in records.values()), default=-1),
        "peak_banks": peak_banks, "peak_reserve": peak_reserve,
        "mean_banks": bank_cycles / elapsed, "mean_reserve": reserve_cycles / elapsed,
        "peak_outstanding_total": peak_outstanding, "peak_result_reservations": peak_results,
        "payload_capacity_fraction": useful_capacity_cycles / (bank_cycles * c.bank_bytes) if bank_cycles else 0,
        "max_ready_to_first_w_cycles": max((r["first_w"] - r["filled"] for r in records.values() if "first_w" in r), default=0),
        "p95_admission_to_result_cycles": latencies[(95 * len(latencies) + 99) // 100 - 1] if latencies else None,
        "counters": dict(counters),
    }
    trace = [{**{k: v for k, v in rec.items() if k not in ("job", "received")},
              "tag": tag, "channel": rec["job"].channel, "beats": rec["job"].beats,
              "received_beats": len(rec["received"])} for tag, rec in records.items()]
    return {"metrics": metrics, "trace": trace}
