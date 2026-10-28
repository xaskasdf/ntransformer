"""Transaction and byte oracle for the isolated multibuffer RTL prototype."""
from collections import defaultdict, deque
import re


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def pattern(tag, beat):
    return sum((((tag * 0x9e3779b9) ^ (beat * 0x10203) ^ (lane * 0x01010101)) & 0xffffffff)
               << (32 * lane) for lane in range(16))


def replay(text, profile, seed, *, partial=False, check_bank_leases=True):
    channels, reserve, beats_max, depth, pending_limit = profile
    jobs = {}
    issue = [deque() for _ in range(channels)]
    pending = [deque() for _ in range(channels)]
    results = [deque() for _ in range(channels)]
    writing = [None] * channels
    leases = {}
    generation = defaultdict(int)
    fill = None
    fault = False
    cases = set()
    passes = []
    groups = defaultdict(list)
    foreign = errors = early_reuse = bytes_checked = concurrent_cycles = 0
    peak_pending = [0] * channels
    last_cycle = -1
    for line in text.splitlines():
        if line.startswith("CASE "):
            cases.add(line[5:])
        elif line.startswith("PASS "):
            passes.append(line)
        elif line.startswith("TRACE "):
            fields = line.split()[1:]
            cycle = int(fields[1])
            require(cycle >= last_cycle, "nonmonotonic trace")
            last_cycle = cycle
            groups[cycle].append(fields)

    for cycle, events in groups.items():
        # Hardware returns banks and consumes old completions on the same edge
        # as a new command. Check old transactions before allocating new leases.
        rank = {"AW": 0, "W": 1, "B": 2, "CPL": 3, "IN": 4, "CMD": 5, "DRAIN": 6, "RESET": 7}
        require(all(e[0] in rank for e in events), "unknown trace event")
        w_ports = set()
        for e in sorted(events, key=lambda event: rank[event[0]]):
            kind = e[0]
            if kind == "CMD":
                p, identity = int(e[2]), int(e[3])
                address, beats, bank, gen = int(e[4],16), int(e[5]), int(e[6]), int(e[7])
                require(not fault and fill is None, f"command admitted under fault or during fill at {cycle}")
                require(0 <= p < channels and 1 <= beats <= beats_max, "invalid admitted descriptor")
                require(identity not in jobs and len(results[p]) < depth, "duplicate tag or descriptor overflow")
                if check_bank_leases:
                    require(0 <= bank < 2*channels+reserve and bank not in leases, "bank allocated before final W")
                    require(bank >= 2*channels or bank // 2 == p, "foreign dedicated bank")
                    require(gen == generation[bank]+1, "incorrect bank generation")
                require(address == 0x100000000+(identity<<12), "address truncation/mismatch")
                require(address % 64 == 0 and address % 4096 + 64*beats <= 4096, "burst boundary")
                if check_bank_leases:
                    if any(j["bank"] == bank and "last_w" in j and "b" not in j for j in jobs.values()):
                        early_reuse += 1
                    generation[bank]=gen
                    leases[bank]=identity
                else:
                    bank=gen=None  # Internal allocation signals are not observed in netlist mode.
                jobs[identity]={"p":p,"address":address,"beats":beats,"bank":bank,"gen":gen,
                                "cmd":cycle,"input":[],"written":[]}
                issue[p].append(identity); results[p].append(identity)
                fill=identity
            elif kind == "IN":
                require(fill is not None, "payload without accepted descriptor")
                j=jobs[fill]
                data=int(e[2],16)
                require(data==pattern(fill,len(j["input"])), "input stimulus corruption")
                j["input"].append(data)
                if len(j["input"])==j["beats"]:
                    j["filled"]=cycle
                    fill=None
            elif kind == "AW":
                p=int(e[2]); address=int(e[3],16); beats=int(e[4])
                require(issue[p] and writing[p] is None, "AW interleave or no descriptor")
                identity=issue[p].popleft(); j=jobs[identity]
                require(j.get("filled",cycle)>=0 and "filled" in j and j["filled"]<cycle, "AW exposed before full input")
                require((address,beats)==(j["address"],j["beats"]), "AW descriptor mismatch")
                j["aw"]=cycle
                writing[p]=identity; pending[p].append(identity)
                require(len(pending[p])<=pending_limit, "AXI pending limit exceeded")
                peak_pending[p]=max(peak_pending[p],len(pending[p]))
            elif kind == "W":
                p=int(e[2]); last=int(e[3]); data=int(e[4],16)
                require(writing[p] is not None, "W without AW")
                identity=writing[p]; j=jobs[identity]; offset=len(j["written"])
                require(offset<j["beats"] and data==j["input"][offset], f"payload mismatch tag {identity} beat {offset}")
                require(last==int(offset+1==j["beats"]), "WLAST mismatch")
                if check_bank_leases:
                    require(leases.get(j["bank"])==identity, "W after premature bank release")
                j["written"].append(data); bytes_checked+=64; w_ports.add(p)
                if last:
                    j["last_w"]=cycle
                    if check_bank_leases:
                        del leases[j["bank"]]
                    writing[p]=None
            elif kind == "B":
                p,bid,resp=map(int,e[2:])
                require(pending[p] and "last_w" in jobs[pending[p][0]], "B accepted without completed W")
                if bid!=0:
                    foreign+=1; fault=True
                    continue
                identity=pending[p].popleft(); j=jobs[identity]
                require(j["last_w"]<cycle, "B before WLAST handshake")
                j["b"]=cycle; j["status"]=resp
                if resp:
                    errors+=1; fault=True
            elif kind == "CPL":
                p,identity,status=map(int,e[2:])
                require(results[p] and results[p][0]==identity, "completion order/tag mismatch")
                j=jobs[identity]
                require("b" in j and j["b"]<cycle, "completion before matching B")
                require(status==j["status"] and "complete" not in j, "completion status/duplication")
                j["complete"]=cycle; results[p].popleft()
            elif kind in ("DRAIN","RESET"):
                require(fill is None and not leases and not any(issue+pending+results) and all(w is None for w in writing),
                        "premature drain or unsafe reset")
                if kind=="RESET":
                    fault=False; generation.clear()
        concurrent_cycles += len(w_ports)>1

    if partial:
        return {"partial_checked_events":sum(map(len,groups.values()))}
    require(cases=={"traffic_and_isolation","parallel_ports","aw_pause_and_stop","stop_during_fill","result_credit",
                    "error_response","foreign_response","invalid_descriptor"}, "missing test cases")
    require(len(passes)==1 and re.fullmatch(rf"PASS seed={seed} cycles=\d+ peak_pending=\d+",passes[0]), "missing PASS")
    require(not re.search(r"FATAL:|ERROR:",text), "simulator failure")
    require(jobs and all("complete" in j for j in jobs.values()), "missing completion")
    require(foreign>0 and errors>0, "missing error coverage")
    if check_bank_leases:
        require(early_reuse>0, "missing lifetime coverage")
    if pending_limit>1:
        require(max(peak_pending)>1, "no overlapping AXI transactions")
    if channels>1:
        require(concurrent_cycles>0, "no simultaneous W transfers")
    return {"status":"MULTIBUFFER_TRANSACTION_REPLAY_PASS","jobs":len(jobs),"bytes_checked":bytes_checked,
            "early_bank_reuses_before_b":early_reuse if check_bank_leases else None,
            "bank_lease_checks":check_bank_leases,"foreign_b":foreign,"error_b":errors,
            "peak_accepted_pending_by_port":peak_pending,"simultaneous_w_cycles":concurrent_cycles,
            "cycles":int(re.search(r"cycles=(\d+)",passes[0])[1])}
