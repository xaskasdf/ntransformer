"""Full retained-watchdog ledger plus the independent DDR readback boundary."""
from check_ntrpu_retained_long_mig import review as review_original
from check_ntrpu_retained_paths_mig import rows, require, number, address


def review(text, source_period_ps):
    result = review_original(text, source_period_ps)
    rr = rows(text, 'RETAINED_LONG_MIG_FORENSIC_AR ')
    require(len(rr) == 1, 'one independent post-quarantine memory read')
    r = rr[0]
    expected = dict(channel=0, identity=18, address=address(18), calibrated=3,
                    ui_rst=0, admit=0, restart=1, source_reset=1, memory_reset=3,
                    cpl=1, ready=0, cookie=0x10000000f, next=0x100000010,
                    epoch=0x100000013, forced=0, global_ack=0)
    for key, value in expected.items():
        observed = int(r[key]) if key in {'channel', 'identity'} else number(r[key])
        require(observed == value, 'independent read context ' + key)
    ar = [row for row in rows(text, 'MANAGED_MIG_AR ') if row['identity'] == '18']
    require(len(ar) == 1 and ar[0]['ps'] == r['ps'], 'fixture read matches actual MIG handshake')
    result['independent_post_quarantine_read'] = dict(expected, ps=int(r['ps']),
        host_read_qualified=False, memory_fixture_only=True)
    return result
