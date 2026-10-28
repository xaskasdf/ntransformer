"""Cold multibuffer periodic witness, including legitimately zero model warnings.

Fork of the existing review_periodic body with only its mandatory warning
coverage assertion removed. Every diagnostic still needs exact physical
causality, and every unwritten physical read still requires its warnings.
The original board reviewer is unchanged.
"""
from collections import Counter
from decimal import Decimal
import re
from check_ntrpu_pcie_ddr_periodic import (WARNING, PIN_DELAY, RETURN_DELAY,
    WARNING_DELAY, BEAT_PS, ROW_MASK, records, ordered, require)


def review_periodic(text, *, terminal=True, summary_marker='NTRPU_DDR_PERIODIC_PASS'):
    """Partial observations may be inspected, but never receive PASS status."""
    require(summary_marker in {'NTRPU_DDR_PERIODIC_PASS','NTRPU_DDR_PERIODIC_FINAL_PASS'},
            'known periodic summary stage')
    require(not re.search(r'(?mi)\b(?:ERROR|FATAL)\s*:|^\s*FAIL\b|^\s*CRITICAL WARNING:', text),
            'no tool/model error')
    config = records(text, 'NTRPU_DDR_PERIODIC_CONFIG', ['ps','interval','abits','colbits'], ['config'])
    require(len(config) == 1 and {k:v for k,v in config[0].items() if k != 'ps'} ==
            dict(interval=200, abits=18, colbits=10, config=3), 'exact periodic configuration')
    phase = config[0]['ps'] % 5000
    periodic = records(text, 'NTRPU_DDR_PERIODIC',
                       ['ps','req','done','rst','axi_rst','aw','ar','rvalid','rready'], ['accept'])
    cas = records(text, 'NTRPU_DDR_READ_CAS', ['ps','injected','rmw'], ['rank','bg','bank','col','rows'])
    returns = records(text, 'NTRPU_DDR_READ_RETURN', ['ps'], ['phy','injected','rmw','native','periodic_done'])
    pins = records(text, 'NTRPU_DDR_PIN_CMD', ['ps','slot','rank','act_n','cke'], ['cmd','adr','bg','bank'])
    for items, label in [(periodic,'periodic'),(cas,'CAS'),(returns,'return')]:
        require(items, label+' coverage'); ordered(items, label)
        require(all(r['ps'] > config[0]['ps'] and r['ps'] % 5000 == phase for r in items), label+' actual UI edge')
    require(pins and all(a['ps'] <= b['ps'] for a,b in zip(pins,pins[1:])), 'physical pin chronology')
    require(all(p['ps'] % 1250 == (phase+PIN_DELAY) % 1250 for p in pins),
            'physical commands sampled on actual DDR clock edge')

    # All warnings must be this exact diagnostic, then undergo causal review.
    warnings = []
    for line in text.splitlines():
        if re.search(r':\s*WARNING\s*:', line):
            match = WARNING.fullmatch(line)
            require(match, 'unknown model warning')
            values = [int(x,16) for x in match.groups()[:6]] + [int(match[7])]
            warnings.append(tuple(values))
    # This workload may initialize every periodically read address.
    # Keep the per-physical-read warning iff unwritten invariant below.
    require(all(a[-1] <= b[-1] for a,b in zip(warnings,warnings[1:])), 'warning chronology')
    by_time = {}
    for warning in warnings:
        by_time.setdefault(warning[-1], Counter())[warning[:-1]] += 1

    # Reconstruct open rows from ACT/PRE/REF, independently of MC row metadata.
    opened = {}; reads = []; writes = []; initialized = set()
    for p in pins:
        require(p['rank'] in range(4) and p['slot'] == p['rank']//2 and
                p['bg'] in range(4) and p['bank'] in range(4) and
                p['adr'] < 1<<18 and p['act_n'] in (0,1) and p['cke'] == 1,
                'known selected-rank physical command')
        # The original observer labels adr[17:15] as cmd. Decode actual
        # RAS/CAS/WE from saved adr[16:14], per the unmodified RCD source.
        require(p['cmd'] == (p['adr'] >> 15) & 7, 'recorded command slice')
        command = (p['adr'] >> 14) & 7
        key = (p['rank'],p['bg'],p['bank'])
        if not p['act_n']:
            require(key not in opened, 'ACT requires a closed bank')
            opened[key] = p['adr']
        elif command in (4,5):
            require(key in opened, 'READ/WRITE requires observed ACT')
            require(p['adr'] & 0x1000 == 0, 'fixed BL8 physical command')
            event = dict(p, row=opened[key], col=p['adr'] & 1023)
            address = (*key, event['row'], event['col'] & ~7)
            if command == 5:
                event['previously_written'] = address in initialized
                reads.append(event)
            else:
                writes.append(event); initialized.add(address)
            if p['adr'] & 1024:
                del opened[key]
        elif command == 2:
            if p['adr'] & 1024:
                opened = {k:v for k,v in opened.items() if k[0] != p['rank']}
            else:
                opened.pop(key, None)
        elif command == 1:
            require(not any(k[0] == p['rank'] for k in opened), 'refresh requires closed rank')
        else:
            require(command in (6,7), 'no unreviewed physical command')

    # This bank drains every observed CAS in the inspected prefix/full log.
    # A prefix ending mid-transaction is intentionally not accepted.
    require(len(cas) == len(reads) == len(returns), 'one physical READ/return per MC CAS')
    chains = []; injected_cas = []; consumed_warning_times = set()
    for c,p,r in zip(cas,reads,returns):
        require(c['injected'] in (0,1) and c['rmw'] == 0 and c['rank'] in range(4) and
                c['bg'] in range(4) and c['bank'] in range(4) and c['col'] < 1024 and
                c['rows'] < 1<<72, 'known MC fields, no unreviewed RMW')
        row = c['rows'] >> (c['bg']*18) & ((1<<18)-1)
        require((c['rank'],c['bg'],c['bank'],c['col'],row) ==
                (p['rank'],p['bg'],p['bank'],p['col'],p['row']), 'MC CAS matches physical READ/open row')
        require(p['ps']-c['ps'] == PIN_DELAY and r['ps']-c['ps'] == RETURN_DELAY,
                'exact measured CAS/pin/return timing')
        require({k:v for k,v in r.items() if k != 'ps'} == dict(
            phy=1, injected=c['injected'], rmw=0, native=1-c['injected'], periodic_done=c['injected']),
            'injected response blocked from native user data; user response retained')
        if c['injected']:
            injected_cas.append(c)
        times = [p['ps']+WARNING_DELAY+i*BEAT_PS for i in range(8)]
        present = [t in by_time for t in times]
        # Fully strobed writes and exact numerical user reads are checked by
        # review_transfers. This witness also checks the physical write history.
        require(any(present) == (not p['previously_written']), 'warning iff physical BL8 address unwritten')
        if any(present):
            require(all(present) and c['injected'] == 1, 'full BL8 warning only on injected read')
            require(c['col'] & 7 == 1 and c['rank'] == 0, 'observed warning burst start/rank contract')
            for i,t in enumerate(times):
                col = (c['col'] & ~7) + (1,2,3,0,5,6,7,4)[i]
                a = (0,c['bg'],c['bank'],row,col,0)
                b = (0,c['bg']^3,c['bank']^3,row^ROW_MASK,col^0x3f8,0)
                require(by_time[t] == Counter({a:9,b:9}), 'exact nine A/nine B warnings with RCD address inversion')
                require(t not in consumed_warning_times, 'warning beat has unique causal READ')
                consumed_warning_times.add(t)
            chains.append(dict(cas_ps=c['ps'],pin_ps=p['ps'],first_warning_ps=times[0],
                               last_warning_ps=times[-1],return_ps=r['ps'],rank=c['rank'],
                               bg=c['bg'],bank=c['bank'],row=row,col=c['col'],native=0))
    require(consumed_warning_times == set(by_time), 'every warning matched exactly once')

    pending = None; lifecycle = []; counts = Counter()
    for p in periodic:
        require(all(p[k] in (0,1) for k in p if k != 'ps'), 'periodic flags known single-bit values')
        require(p['rst'] == 0 and p['req']+p['accept']+p['done'] == 1, 'one active periodic event outside MC reset')
        if p['req']:
            require(pending is None, 'one outstanding periodic request')
            pending = dict(request_ps=p['ps']); counts['requests'] += 1
        elif p['accept']:
            require(pending is not None and 'accept_ps' not in pending, 'accept follows one request')
            pending['accept_ps'] = p['ps']; counts['accepts'] += 1
        else:
            require(pending is not None and 'accept_ps' in pending, 'completion follows accepted request')
            pending['return_ps'] = p['ps']; lifecycle.append(pending); pending = None; counts['returns'] += 1
    require(pending is None and len(lifecycle) == len(injected_cas), 'periodic requests/CAS/completions drained')
    for life,c in zip(lifecycle,injected_cas):
        require(life['request_ps'] < life['accept_ps'] < c['ps'] and
                life['return_ps'] == c['ps']+RETURN_DELAY, 'request/accept/injected CAS/completion association')
    summary = records(text, summary_marker, ['requests','accepts','returns','periodic_cas','user_cas'])
    reconstructed = dict(counts, periodic_cas=len(injected_cas), user_cas=len(cas)-len(injected_cas))
    if terminal:
        require(summary == [reconstructed], 'terminal periodic counters match all observations')
        require(reconstructed['user_cas'] > 0 and writes, 'real user reads and writes observed')
    else:
        require(not summary, 'partial inspection requires a nonterminal trace')
    return dict(status='PASS_PERIODIC_WITNESS' if terminal else 'NONTERMINAL_WITNESS_ONLY',
                configuration=config[0], counters=reconstructed, physical_reads=len(reads),
                physical_writes=len(writes), warning_lines=len(warnings), warning_beats=len(by_time),
                warning_bursts=len(chains), warning_chains=chains, lifecycle=lifecycle,
                measured_delays_ps=dict(cas_to_pin=PIN_DELAY,cas_to_return=RETURN_DELAY,
                                        pin_to_first_warning=WARNING_DELAY,beat=BEAT_PS),
                limitations=['Cold generic DDR4-1600 configuration only; not a DIMM/hardware qualification.',
                    'Warning text has no chip identity: check nine instances per A/B address, not unique chip tags.',
                    'Recorded cmd is adr[17:15]; actual command decoded from retained adr[16:14].',
                    'Fixed measured latencies and BL8 order are scoped to this generated model/configuration.'])



def review(text):
    periodic=review_periodic(text)
    rows=records(text,'NTRPU_DDR_PERIODIC_DRAIN',['begin_ps','end_ps','falling_edges','requests','returns'])
    require(len(rows)==1,'one final periodic drain')
    d=rows[0];phase=(periodic['configuration']['ps']+2500)%5000
    first=d['begin_ps']+(phase-d['begin_ps'])%5000
    if first==d['begin_ps']:first+=5000
    require(0<d['falling_edges']<200 and d['end_ps']==first+5000*(d['falling_edges']-1),'bounded consecutive drain edges')
    counts=periodic['counters']
    require(d['requests']==d['returns']==counts['requests']==counts['accepts']==counts['returns']==counts['periodic_cas'],'periodic drain counters')
    ends=re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)
    require(ends and {int(Decimal(v)*{'ps':1,'ns':1000,'us':1000000}[u]) for v,u in ends}=={d['end_ps']},'finish at completed drain')
    returns=records(text,'NTRPU_DDR_READ_RETURN',['ps'],['phy','injected','rmw','native','periodic_done'])
    require(returns[-1]['ps']<d['end_ps'] and all(life['return_ps']<d['end_ps'] for life in periodic['lifecycle']),'all observed periodic returns precede finish')
    return dict(periodic=periodic,drain=d)
