"""Qualify the read-only periodic witness for the cold generic DDR4 bank.

This is an exact configuration/trace contract, not a general DDR model.
Keep all diagnostics: an unexpected model warning or incomplete chain fails.
"""
from collections import Counter
import re

from check_ntrpu_pcie_ddr_ffn import require, rows, review_transfers


WARNING = re.compile(
    r'Model:WARNING: Reading unwritten address: C:([0-9a-f]+) '
    r'BG:([0-9a-f]+) B:([0-9a-f]+) R:([0-9a-f]+) '
    r'C:([0-9a-f]+) dm:([0-9a-f]+) @([0-9]+)')
PIN_DELAY = 17188  # Measured MC observation -> selected-rank physical READ, ps.
RETURN_DELAY = 65000  # Measured MC observation -> PHY return, ps.
WARNING_DELAY = 15625  # Physical READ -> first model beat, this RDIMM model.
BEAT_PS = 625
ROW_MASK = 0x22bf8  # Unmodified ddr4_rcd_model.sv B-side inversions.


def records(text, marker, decimal, hexadecimal=()):
    out = []
    for row in rows(text, marker):
        require(set(row) == set(decimal) | set(hexadecimal), marker+' exact fields')
        value = {}
        for name, token in row.items():
            base = 10 if name in decimal else 16
            require(re.fullmatch(r'[0-9]+' if base == 10 else r'[0-9a-f]+', token),
                    marker+' known unsigned '+name)
            value[name] = int(token, base)
        out.append(value)
    return out


def ordered(items, label):
    require(all(a['ps'] < b['ps'] for a, b in zip(items, items[1:])),
            label+' strict observation order')


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
    require(warnings, 'unwritten-read diagnostic coverage')
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
    result = review_transfers(text)
    result['periodic'] = review_periodic(text)
    result['status'] = 'PASS_COLD_TRANSFERS_AND_PERIODIC_CAUSALITY'
    return result


def mutations(text, *, terminal=True):
    """Corrupt each independent link of the warning/physical/native chain."""
    out = {}
    def edit(name, marker, old, new):
        line = next(l for l in text.splitlines() if l.startswith(marker+' ') and old in l)
        out[name] = text.replace(line, line.replace(old,new,1),1)
    for name, marker in [('missing_config','NTRPU_DDR_PERIODIC_CONFIG'),
                         ('missing_cas','NTRPU_DDR_READ_CAS'),('missing_return','NTRPU_DDR_READ_RETURN'),
                         ('missing_pin_act','NTRPU_DDR_PIN_CMD'),('missing_lifecycle','NTRPU_DDR_PERIODIC'),
                         ('missing_summary','NTRPU_DDR_PERIODIC_PASS')]:
        if not terminal and name == 'missing_summary':
            continue
        line = next(l for l in text.splitlines() if l.startswith(marker+' '))
        out[name] = text.replace(line+'\n','',1)
    for name,old,new in [('cas_injected','injected=1','injected=0'),('cas_rmw','rmw=0','rmw=1'),
                         ('cas_rank','rank=0','rank=1'),('cas_bg','bg=0','bg=1'),
                         ('cas_bank','bank=0','bank=1'),('cas_col','col=001','col=009'),
                         ('cas_row','rows=000000000000000000','rows=000000000000000001'),
                         ('cas_unknown','injected=1','injected=x')]:
        edit(name,'NTRPU_DDR_READ_CAS',old,new)
    for name,old,new in [('native_leak','native=0','native=1'),('return_injected','injected=1','injected=0'),
                         ('return_phy','phy=1','phy=0'),('return_rmw','rmw=0','rmw=1'),
                         ('return_done','periodic_done=1','periodic_done=0')]:
        edit(name,'NTRPU_DDR_READ_RETURN',old,new)
    edit('user_return_lost','NTRPU_DDR_READ_RETURN','native=1','native=0')
    edit('config_interval','NTRPU_DDR_PERIODIC_CONFIG','interval=200','interval=201')
    edit('pin_rank','NTRPU_DDR_PIN_CMD','rank=0','rank=1')
    edit('pin_cke','NTRPU_DDR_PIN_CMD','cke=1','cke=0')
    edit('pin_slice','NTRPU_DDR_PIN_CMD','cmd=0','cmd=1')
    edit('pin_row','NTRPU_DDR_PIN_CMD','adr=00000','adr=00001')
    edit('pin_read_address','NTRPU_DDR_PIN_CMD','adr=14001','adr=14009')
    edit('request_reset','NTRPU_DDR_PERIODIC','rst=0','rst=1')
    edit('accept_without_request','NTRPU_DDR_PERIODIC','req=1 accept=0','req=0 accept=1')
    for marker,name in [('NTRPU_DDR_READ_CAS','cas_time'),('NTRPU_DDR_READ_RETURN','return_time'),
                        ('NTRPU_DDR_PIN_CMD','pin_time'),('NTRPU_DDR_PERIODIC','request_time')]:
        line = next(l for l in text.splitlines() if l.startswith(marker+' '))
        token = re.search(r'ps=(\d+)',line)[0]
        edit(name,marker,token,'ps='+str(int(token[3:])+1))
    warning = next(l for l in text.splitlines() if WARNING.fullmatch(l))
    out['missing_warning'] = text.replace(warning+'\n','',1)
    out['duplicate_warning'] = text.replace(warning,warning+'\n'+warning,1)
    out['unknown_warning'] = text.replace(warning,warning.replace('unwritten','unknown'),1)
    for name,old,new in [('warning_row','R:0','R:1'),('warning_col','C:1 dm:','C:2 dm:'),
                         ('warning_bg','BG:0','BG:1'),('warning_bank','B:0','B:1'),
                         ('warning_dm','dm:0','dm:1'),('warning_chip','C:0 BG:','C:1 BG:')]:
        out[name] = text.replace(warning,warning.replace(old,new,1),1)
    bside = next(l for l in text.splitlines() if WARNING.fullmatch(l) and 'R:22bf8' in l)
    out['rcd_inversion'] = text.replace(bside,bside.replace('R:22bf8','R:22bf9'),1)
    out['warning_time'] = text.replace(warning,warning.replace('@'+WARNING.fullmatch(warning)[7],
                                '@'+str(int(WARNING.fullmatch(warning)[7])+1)),1)
    out['hierarchical_error'] = text+'\nModel:ERROR: injected corruption\n'
    return out


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--mutations', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = args.log.read_text(encoding='utf-8')
    result = review(raw)
    if args.mutations:
        rejected = []
        for name, corrupted in mutations(raw).items():
            require(corrupted != raw, 'mutation must change evidence: '+name)
            try:
                review(corrupted)
            except (ValueError, AssertionError):
                rejected.append(name)
            else:
                raise ValueError('accepted corrupted evidence: '+name)
        result['rejected_periodic_mutations'] = rejected
    encoded = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(encoded, encoding='utf-8')
    else:
        print(encoded, end='')
