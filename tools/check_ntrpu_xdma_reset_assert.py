"""Independent VCD/marker review of the bounded optional-reset assertion probe."""
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshots(text):
    header, body = text.split('$enddefinitions $end', 1)
    require(re.search(r'\$timescale\s+1ps\s+\$end', header), 'VCD timescale')
    scopes, names, widths = [], {}, {}
    for line in header.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == '$scope':
            scopes.append(fields[2])
        elif fields[0] == '$upscope':
            scopes.pop()
        elif fields[0] == '$var':
            name = '/' + '/'.join(scopes + [fields[4]])
            require(name not in names, 'duplicate VCD name')
            names[name] = fields[3]
            widths[fields[3]] = int(fields[2])
    values, updates, now = {}, {}, 0
    rows = []
    def flush():
        before = {n: values.get(c, 'x'*widths[c]) for n, c in names.items()}
        values.update(updates)
        after = {n: values.get(c, 'x'*widths[c]) for n, c in names.items()}
        rows.append((now, before, after))
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('$'):
            continue
        if line.startswith('#'):
            flush()
            updates = {}
            new = int(line[1:])
            require(new >= now, 'VCD time reversal')
            now = new
        else:
            if line.startswith('b'):
                raw, code = line[1:].split()
            else:
                raw, code = line[0], line[1:]
            require(code in widths and set(raw) <= set('01xz'), 'invalid VCD bits')
            require(len(raw) <= widths[code], 'overwide VCD value')
            updates[code] = raw.rjust(widths[code], raw[0] if raw[0] in 'xz' else '0')
    flush()
    return rows, {n: widths[c] for n, c in names.items()}


def review(sim, vcd):
    require(not re.search(r'^\s*(?:Fatal:|ERROR:|CRITICAL WARNING:)', sim, re.M), 'simulator failure')
    def one(prefix):
        lines = [s for s in sim.splitlines() if s.startswith(prefix+' ')]
        require(len(lines) == 1, 'unique '+prefix)
        return {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', lines[0])}
    enum, quiet, assertion, final = [one(x) for x in (
        'XDMA_RESET_ENUM', 'XDMA_RESET_QUIET', 'XDMA_RESET_ASSERT', 'XDMA_RESET_ASSERT_OBSERVED')]
    require(enum['ps'] < quiet['ps'] < assertion['ps'] < final['ps'], 'marker order')
    require(final['samples'] == 256 and final['held'] == 1 and final['release_tested'] == 0, 'bounded scope')
    samples = [{k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
               for line in sim.splitlines() if line.startswith('XDMA_RESET_SAMPLE ')]
    require(len(samples) == 256 and [r['index'] for r in samples] == list(range(256)), 'all reference samples')
    rows, widths = snapshots(vcd)
    root, ep, bench = '/board/RP/', '/board/EP/', '/board/ntrpu_reset_probe/'
    valids = [ep+p+'_'+s+'valid' for p in ('m_axi', 'm_axib', 'm_axil') for s in ('aw', 'w', 'b', 'ar', 'r')]
    controls = ['/board/sys_rst_n', ep+'user_resetn', ep+'user_lnk_up', root+'user_lnk_up', bench+'reset_n']
    expected = set(controls+valids+['/board/ep_sys_clk_p', root+'user_clk', ep+'user_clk', root+'cfg_ltssm_state', bench+'phase'])
    expected |= {root+p+'_'+s for p in ('s_axis_rq', 'm_axis_rc') for s in ('tvalid', 'tready')}
    require(set(widths) == expected and len(widths) == 29, 'exact native signal inventory')
    for n in controls+valids+['/board/ep_sys_clk_p', root+'user_clk', ep+'user_clk']:
        require(widths[n] == 1, 'scalar width '+n)
    require(widths[bench+'phase'] == 32 and widths[root+'cfg_ltssm_state'] == 6, 'control widths')
    require(rows[-1][0] == final['ps'], 'VCD final timestamp')
    def number(bits):
        require(bool(bits) and set(bits) <= set('01'), 'unknown observed value')
        return int(bits, 2)
    ref_edges, quiet_edges, ep_edges, rp_edges, rq, rc = [], [], 0, 0, 0, 0
    changes, first_reset, sampled = [], None, {}
    sample_by_time = {s['ps']: s for s in samples}
    for ps, before, after in rows:
        if before.get(bench+'reset_n') != after.get(bench+'reset_n'):
            changes.append((ps, after[bench+'reset_n']))
        if quiet['ps'] <= ps:
            require(number(after[bench+'phase']) == (1 if ps < assertion['ps'] else 2), 'observed phase')
            require(after['/board/sys_rst_n'] == '1', 'PERST remained released')
            require(after[bench+'reset_n'] == ('1' if ps < assertion['ps'] else '0'), 'optional reset stimulus')
            for n in controls:
                number(after[n])
            if ps < assertion['ps']:
                require(all(after[n] == '1' for n in controls), 'healthy quiet interval')
            elif after[ep+'user_resetn'] == '0' and first_reset is None:
                first_reset = ps
            if first_reset is not None:
                require(after[ep+'user_resetn'] == '0', 'reset re-released while input held')
        if before['/board/ep_sys_clk_p'] == '0' and after['/board/ep_sys_clk_p'] == '1':
            if quiet['ps'] < ps < assertion['ps']:
                quiet_edges.append(ps)
            if ps > assertion['ps']:
                ref_edges.append(ps)
        for domain in (ep, root):
            if before[domain+'user_clk'] != '0' or after[domain+'user_clk'] != '1':
                continue
            if domain == root and ps < quiet['ps']:
                for prefix in ('s_axis_rq', 'm_axis_rc'):
                    if before[root+prefix+'_tvalid'] == '1' and number(before[root+prefix+'_tready']):
                        if prefix == 's_axis_rq': rq += 1
                        else: rc += 1
            if ps >= quiet['ps']:
                if domain == ep:
                    ep_edges += 1
                    require(all(before[n] == '0' for n in valids), 'AXI idle on native clock')
                else:
                    rp_edges += 1
                    require(all(before[root+p+'_tvalid'] == '0' for p in ('s_axis_rq', 'm_axis_rc')), 'root idle on native clock')
        if ps in sample_by_time:
            s = sample_by_time[ps]
            require((s['reset'], s['link'], s['rp_link']) == tuple(number(after[n]) for n in (ep+'user_resetn', ep+'user_lnk_up', root+'user_lnk_up')), 'VCD/marker sample mismatch')
            sampled[ps] = s
    require(changes == [(0, '1'), (assertion['ps'], '0')], 'only one runtime optional-reset assertion, no release')
    require(len(quiet_edges) == len(ref_edges) == 256, 'reference edge windows')
    require(all(b-a == 10000 for a,b in zip(ref_edges,ref_edges[1:])), '100MHz reference')
    require(list(sample_by_time) == [t+1 for t in ref_edges] and len(sampled) == 256, 'sample timing')
    require(rq > 0 and rc > 0, 'native enumeration traffic witnessed')
    require(ep_edges == final['ep_edges'] > 0 and rp_edges == final['rp_edges'] > 0, 'native clock counters')
    require(final['reset_samples'] == sum(s['reset'] == 0 for s in samples), 'reset count')
    require(final['link_down_samples'] == sum(s['link'] == 0 for s in samples), 'link count')
    return dict(status='OBSERVED_OPTIONAL_RESET_ASSERTION_ONLY', enumeration_ps=enum['ps'],
        assertion_ps=assertion['ps'], final_ps=final['ps'], first_axi_reset_ps=first_reset,
        assertion_latency_ps=None if first_reset is None else first_reset-assertion['ps'],
        reset_samples=final['reset_samples'], link_down_samples=final['link_down_samples'],
        endpoint_edges=ep_edges, root_edges=rp_edges, native_enumeration_requests=rq,
        native_enumeration_completions=rc, held_at_finish=True, qualified_reset_recovery=False)
