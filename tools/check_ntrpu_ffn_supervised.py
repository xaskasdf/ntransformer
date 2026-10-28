"""Review real FFN/ABI2 results together with observed supervisor boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from check_ntrpu_ffn_irq_async import review as ffn_review, mutations as ffn_mutations
from check_ntrpu_ffn_host import records


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def review(text, seed, host_period_ps):
    ffn = ffn_review(text, seed, host_period_ps, None)
    errors = list(ffn['errors'])

    def need(ok, why):
        if not ok:
            raise ValueError(why)

    try:
        need(host_period_ps in (4000, 6200), 'host clock profile')
        releases = Counter()
        reset_assertions = []
        release_cases = []
        asserted = {'rst': 0, 'hrst': 0}
        counts = Counter()
        last = None
        block = None
        passed = False
        supervision = []
        mmio = {'B': [], 'R': []}
        binary = {'ext','rst','hrst','da','ha','run','req','can','held','ready','mh','mr','hh','hr','allow'}
        for line in text.splitlines():
            if not line.startswith('NTRPU_SFFN_'):
                continue
            need(not passed, 'supervision trace after terminal PASS')
            kind = line.split()[0].removeprefix('NTRPU_SFFN_')
            raw = fields(line)
            if kind in mmio:
                row = {k: int(v,16 if k in {'addr','data'} else 10) for k,v in raw.items()}
                need(row['total'] == len(mmio[kind])+1, 'MMIO handshake counter')
                need((row['ps']-700) % host_period_ps == host_period_ps//2, 'MMIO handshake edge')
                need(last and last['ha'] and last['run'] and last['rst'] and last['hrst'], 'MMIO before admission')
                mmio[kind].append(row)
                continue
            need(all(re.fullmatch(r'\d+', v) for v in raw.values()), 'unknown supervision pin')
            row = {k: int(v) for k, v in raw.items()}
            if kind == 'PASS':
                need(row == dict(cases=21, resets=5, memory_peer_blocked=1), 'supervised terminal coverage')
                passed = True
                continue
            if kind == 'PEER_BLOCK':
                need(block is None and last is not None, 'duplicate peer-block observation')
                need(last['id'] == 0 and not any(last[k] for k in ['ext','rst','hrst','run','can','mh','allow']), 'unacknowledged memory did not block reset release')
                need(row['ps'] >= 1600000, 'memory-block observation too early')
                block = row
                continue
            need(kind in {'M','OUT'}, 'unexpected supervision record')
            need(set(row) == binary | {'id','ps','mt','dt','ht'}, 'supervision fields')
            need(all(row[k] in (0, 1) for k in binary), 'nonbinary supervision pin')
            need(row['id'] in {0,16,17,18,19}, 'unexpected reset/recovery case')
            t = row['ps']
            for key, period, phase in [('dt',5000,2500),('ht',host_period_ps,700+host_period_ps//2),('mt',10000,5000)]:
                expected = 0 if t < phase else 1+(t-phase)//period
                need(row[key] == expected, 'actual clock edge counter '+key)
            if kind == 'M':
                need(t % 10000 == 5001 and not row['run'], 'management observation clock')
            if last:
                need(t >= last['ps'], 'supervision time order')
                for reset, admit, counter, period, phase in [('rst','da','dt',5000,2500),('hrst','ha','ht',host_period_ps,700+host_period_ps//2)]:
                    if last[reset] and not row[reset]:
                        asserted[reset] = row[counter]
                        if reset == 'rst':
                            reset_assertions.append(row['id'])
                    if not last[reset] and row[reset]:
                        need(row[counter]-asserted[reset] >= 64, 'insufficient actual reset edges')
                        need((t-1-phase) % period == 0, 'asynchronous reset release')
                        need(not row['ext'] and not row['req'], 'release before supervisor permission')
                        releases[reset] += 1
                        if reset == 'rst':
                            release_cases.append(row['id'])
                    if not last[admit] and row[admit]:
                        need((t-1-phase) % period == 0, 'asynchronous admission release')
                        releases[admit] += 1
            if row['da'] or row['ha']:
                need(row['run'] and row['rst'] and row['hrst'] and row['ready'] and row['mr'] and row['hr'] and not row['req'] and not row['ext'], 'unsafe admission')
            if row['can']:
                need(row['held'] and row['mh'] and row['hh'] and row['req'] and not row['ext'], 'rearm without peer acknowledgement')
            if not row['allow']:
                need(not row['mh'] and not row['can'] and not row['rst'] and not row['hrst'], 'memory acknowledgement bypassed')
            last = row
            counts[kind] += 1
            supervision.append(row)
        need(passed and block is not None, 'missing peer block or terminal PASS')
        need(releases == {'rst':5,'hrst':5,'da':5,'ha':5}, 'release coverage')
        need(reset_assertions == [16,17,18,19], 'in-flight reset coverage')
        need(release_cases == [0,16,17,18,19], 'recovery order')
        need(counts['M'] > 300 and counts['OUT'] >= 25, 'insufficient supervision observations')
        writes = records(text, 'NTRPU_FFN_HOST_WRITE')
        reads = records(text, 'NTRPU_FFN_IRQ_READ')
        need(len(mmio['B']) == len(writes) and len(mmio['R']) == len(reads), 'MMIO handshake coverage')
        for bus, logged in zip(mmio['B'], writes):
            need({k:bus[k] for k in ['id','addr','data','resp','order']} == logged, 'observed B identity')
        for bus, logged in zip(mmio['R'], reads):
            need(bus['resp'] == 0 and {k:bus[k] for k in ['id','addr','data']} | {'time_ps':bus['ps']} == logged, 'observed R identity')
        return dict(status='FAIL' if errors else 'PASS', errors=errors, ffn=ffn,
                    supervision=dict(releases=dict(releases),observations=dict(counts),peer_block=block,
                                     reset_cases=reset_assertions,release_cases=release_cases,mmio_handshakes={k:len(v) for k,v in mmio.items()}),
                    scope='Actual supervisor, ABI2, AMD AXI converters and quad/FIFO HLS FFN. External protocol peers are models. Five coordinated resets per seed; clocks remain running. Host AXI4 DMA port is idle in this bank. No XDMA, DDR or board qualification.')
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
        return dict(status='FAIL',errors=errors,ffn=ffn)


def mutations(text):
    result = ffn_mutations(text, False)
    lines = text.splitlines()
    def change(name, predicate, key=None, value=None, remove=False):
        changed = lines.copy()
        i = next(i for i,l in enumerate(lines) if predicate(l,fields(l)))
        if remove:
            changed.pop(i)
        else:
            changed[i] = re.sub(r'\b'+key+r'=\S+',key+'='+value,changed[i],count=1)
        result[name] = '\n'.join(changed)+'\n'
    change('supervisor_missing_pass',lambda l,r:l.startswith('NTRPU_SFFN_PASS'),remove=True)
    change('supervisor_missing_peer_block',lambda l,r:l.startswith('NTRPU_SFFN_PEER_BLOCK'),remove=True)
    change('supervisor_unsafe_admission',lambda l,r:l.startswith('NTRPU_SFFN_M') and r['run']=='0','da','1')
    change('supervisor_counter',lambda l,r:l.startswith('NTRPU_SFFN_OUT') and r['rst']=='1','dt','0')
    change('supervisor_unknown',lambda l,r:l.startswith('NTRPU_SFFN_M'),'hrst','x')
    change('supervisor_false_ack',lambda l,r:l.startswith('NTRPU_SFFN_M') and r['allow']=='0','can','1')
    change('supervisor_missing_recovery',lambda l,r:l.startswith('NTRPU_SFFN_OUT') and r['id']=='19' and r['ha']=='1',remove=True)
    change('missing_b_handshake',lambda l,r:l.startswith('NTRPU_SFFN_B '),remove=True)
    change('wrong_b_count',lambda l,r:l.startswith('NTRPU_SFFN_B '),'total','0')
    change('wrong_r_edge',lambda l,r:l.startswith('NTRPU_SFFN_R '),'ps','1')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--host-period-ps',type=int,required=True)
    args = parser.parse_args()
    result = review(args.log.read_text(encoding='utf-8'),args.seed,args.host_period_ps)
    print(json.dumps(result,indent=2))
    raise SystemExit(result['status'] != 'PASS')
