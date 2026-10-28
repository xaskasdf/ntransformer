"""Independent lease ledger for captured returns; grants close through validation."""

from collections import Counter
import re


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def replay(text, profile, seed):
    channels, dedicated, reserve, quota, generation_bits = profile
    banks = channels * dedicated + reserve
    bank_bits = max(1, (banks - 1).bit_length())
    max_generation = (1 << generation_bits) - 1
    generations = [0] * banks
    leases = {}  # bank -> (channel, generation)
    turn = 0
    fault = False
    pending = (0, 0, 0)
    cycles = 0
    grants = [0] * channels
    counters = Counter()
    peak_owned = peak_reserve = 0
    cases = set()
    pass_lines = []

    def mask():
        return sum(1 << bank for bank in leases)

    def retired():
        return sum(1 << bank for bank in range(banks)
                   if bank not in leases and generations[bank] == max_generation)

    for line in text.splitlines():
        if line.startswith("CASE "):
            cases.add(line[5:])
        if line.startswith("PASS "):
            pass_lines.append(line)
        if not line.startswith("TRACE "):
            continue
        try:
            fields = [int(value, 16) for value in line.split()[1:]]
        except ValueError as error:
            raise RuntimeError(f"unknown/non-numeric signal at cycle {cycles}") from error
        require(len(fields) == 19, f"trace field count at {cycles}")
        cycle, reset, stop, requests, returns, return_banks, return_gens = fields[:7]
        require(cycle == cycles, "missing or duplicate trace cycle")
        staged_returns, staged_banks, staged_gens = pending
        chosen = None
        valid_returns = set()
        errors = 0
        expected_ready = expected_bank = expected_generation = 0
        if not reset:
            require(not leases and not pending[0], f"uncoordinated reset with live leases at {cycle}")
            expected_pre = [0] * 8  # TB normalizes pre-state during synchronous reset.
            generations = [0] * banks
            turn = 0
            fault = False
            pending = (0, 0, 0)
        else:
            for channel in range(channels):
                if staged_returns & (1 << channel):
                    bank = (staged_banks >> (channel * bank_bits)) & ((1 << bank_bits) - 1)
                    gen = (staged_gens >> (channel * generation_bits)) & max_generation
                    if bank in leases and leases[bank] == (channel, gen):
                        valid_returns.add(bank)
                    else:
                        errors |= 1 << channel
            remaining = {bank: lease for bank, lease in leases.items() if bank not in valid_returns}
            available = {bank for bank in range(banks)
                         if bank not in remaining and generations[bank] < max_generation}
            if not stop and not fault and not returns and not staged_returns:
                for channel in [(turn + offset) % channels for offset in range(channels)]:
                    if not requests & (1 << channel):
                        continue
                    own = available & set(range(channel * dedicated, (channel + 1) * dedicated))
                    loans = sum(bank >= channels * dedicated and lease[0] == channel
                                for bank, lease in remaining.items())
                    shared = {bank for bank in available if bank >= channels * dedicated}
                    candidates = own if own else (shared if loans < quota else set())
                    if candidates:
                        chosen = channel
                        expected_bank = min(candidates)
                        expected_generation = generations[expected_bank] + 1
                        expected_ready = 1 << channel
                        break
            expected_pre = [expected_ready, expected_bank, expected_generation, errors,
                            (1 << channels) - 1, int(fault), mask(), retired()]
            counters["valid_returns"] += len(valid_returns)
            counters["invalid_returns"] += errors.bit_count()
            counters["returns_during_stop"] += len(valid_returns) * bool(stop)
            counters["returns_during_fault"] += len(valid_returns) * bool(fault)
            leases = remaining
            if chosen is not None:
                counters["same_cycle_reuse"] += expected_bank in valid_returns
                leases[expected_bank] = (chosen, expected_generation)
                generations[expected_bank] = expected_generation
                turn = (chosen + 1) % channels
                grants[chosen] += 1
            fault = fault or bool(errors)
            counters["capture_blocked"] += bool(returns and requests)
            counters["validation_blocked"] += bool(staged_returns and requests)
            pending = (returns, return_banks, return_gens)
        require(fields[7:15] == expected_pre,
                f"pre-edge mismatch cycle {cycle}: got {fields[7:15]}, expected {expected_pre}")
        expected_post = [int(fault), mask(), retired(), int(not leases and not pending[0])]
        require(fields[15:] == expected_post,
                f"post-edge mismatch cycle {cycle}: got {fields[15:]}, expected {expected_post}")
        peak_owned = max(peak_owned, len(leases))
        peak_reserve = max(peak_reserve, sum(bank >= channels * dedicated for bank in leases))
        cycles += 1

    expected_cases = {"cold_reset", "simultaneous_requests", "staged_return_reuse",
                      "stop_keeps_returns_live", "quota", "stale_generation_after_reuse",
                      "fault_is_sticky_but_drains", "double_return", "seeded_random", "consecutive_returns", "final_drain"}
    if channels > 1:
        expected_cases |= {"wrong_owner", "conflicting_returns"}
    if (1 << bank_bits) > banks:
        expected_cases.add("invalid_bank_index")
    if generation_bits <= 4 and banks == 1:
        expected_cases.add("generation_exhaustion")
    require(expected_cases == cases, f"missing/unexpected directed cases: {expected_cases ^ cases}")
    require(not leases and not fault, "final state is not clean and drained")
    expected_pass = (f"PASS seed={seed} cycles={cycles} channels={channels} dedicated={dedicated} "
                     f"reserve={reserve} quota={quota} generation={generation_bits}")
    require(pass_lines == [expected_pass], "missing/duplicate/wrong PASS marker")
    require(cycles >= 512 and all(grants), "insufficient stimulus")
    require(counters["same_cycle_reuse"] == 0 and counters["returns_during_fault"] > 0 and
            counters["capture_blocked"] > 0 and counters["validation_blocked"] > 0 and counters["invalid_returns"] > 0,
            "missing staged-return fault/drain coverage")
    require(not re.search(r"FATAL:|ERROR:", text), "simulator error in positive run")
    return {"status": "REGISTERED_RETURN_EDGE_REPLAY_PASS", "cycles": cycles, "seed": seed,
            "profile": list(profile), "cases": sorted(cases), "grants_by_channel": grants,
            "peak_owned": peak_owned, "peak_reserve": peak_reserve, "counters": dict(counters)}
