"""Derive the full checker, retaining the original attention numerical checks."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    path = ROOT / 'tools/check_ntrpu_rope_attention_real_r1.py'
    original = path.read_text()
    source = original.replace('import ntrpu_rope_attention_fixture_r1 as fixture_module',
        'import ntrpu_input_attention_fixture_r1 as fixture_module\nimport ntrpu_input_attention_source_review_r1 as producer_review')
    source = source.replace('BASE = 0x900000000fc0', 'BASE = fixture_module.KV_BASE')
    source = source.replace("'COMPLETE': 'cycle job status append committed tokens bytes heads'", "'COMPLETE': 'cycle job status child committed tokens bytes heads'")
    source = source.replace("HEX = {'sequence'", "CONSUMER_TAGS = set(SCHEMA)\nSCHEMA.update(producer_review.SCHEMA)\nSCHEMA['PRODUCER_SUMMARY'] = 'starts dones acks consumer_starts consumer_acks norm_starts norm_dones mac reads writes'\nHEX = {'tag', 'sequence'", 1)
    source = source.replace('RAP_', 'IAR_').replace("line[4:]", "line[4:]")
    source = source.replace('manifest, samples, refs = fixture_module.load(folder)', 'manifest, producer_fixture, samples, refs = fixture_module.load(folder)')
    source = source.replace("folder/'attention-reference/", "folder/'consumer/attention-reference/")
    source = source.replace('0x2000000000001000', '0x1000')
    source = source.replace('0x100000fc0', 'fixture_module.address(6)').replace('0x8000000080', 'fixture_module.address(7)')
    source = source.replace('0xa000000000', 'fixture_module.address(9)').replace('0xc000000fc0', 'fixture_module.address(8)')
    source = source.replace('4000000', '120000000').replace('600010', '20000010')
    source = source.replace("release['cycle']-ack['cycle'] < 50", "release['cycle']-ack['cycle'] < 100")
    source = source.replace("row['append']", "row['child']")
    marker = '    def job_review(events, spec, ident, memory):\n'
    assert source.count(marker) == 1
    prelude = '''        ordinal = sum(p['mode'] != 1 for p in manifest['plan'][:ident])-1
        producer = producer_review.review_source(events, producer_fixture, spec, ident, ordinal)
        events = [(tag, row) for tag, row in events if tag in CONSUMER_TAGS]
        if spec['mode'] == 1:
            need([tag for tag, _ in events] == ['BEGIN', 'ADMIT', 'COMPLETE', 'ACK', 'RELEASE'], 'Preflight rejects before all arithmetic/AXI')
            start, admit, complete, ack, release = [row for _, row in events]
            seq, epoch = 0x1234567800000000+spec['group'], 0xfedcba9800000000+spec['group']
            need(start == dict(cycle=start['cycle'], job=ident, sample=spec['sample'], mode=1, group=spec['group'], prefix=spec['prefix'], sequence=seq, epoch=epoch), 'Planned rejected command')
            need(admit == dict(cycle=admit['cycle'], job=ident, sequence=seq, epoch=epoch, layer=0, position=spec['sample']), 'Rejected admission identity')
            need(complete == dict(cycle=complete['cycle'], job=ident, status=1, child=0, committed=0, tokens=0, bytes=0, heads=0), 'Rejected command certificate')
            need(ack == dict(cycle=ack['cycle'], job=ident) and release == dict(cycle=release['cycle'], job=ident, prefix=spec['prefix']), 'Rejected ownership release')
            need(start['cycle'] < admit['cycle'] < complete['cycle'] < ack['cycle'] < release['cycle'] and ack['cycle'] >= complete['cycle']+7 and release['cycle']-ack['cycle'] < 100, 'Rejected lifetime')
            return dict(job=ident, mode=1, prefix=spec['prefix'], heads=0, words=0, exact_rope_words=0, kv_reads=0, writes=0, max_output_ulp=0, max_scaled_state_error='0', producer=producer['summary'])
'''
    source = source.replace(marker, marker+prelude)
    old = "                    expected = (source_frames if port == 0 else value_frames)[len(reads[port])]"
    assert source.count(old) == 1
    source = source.replace(old, old+'''
                    if a in producer['memory']:
                        actual = producer['memory'][a]
                        need(actual['cycle'] < c and row['data'] == actual['data'], 'Consumer Q/K/V must equal an earlier successful producer B commit')
                    else:
                        need(port == 0 and fixture_module.address(9) <= a < fixture_module.address(9)+256, 'Only phase coefficients are external immutable consumer inputs')''')
    source = source.replace("max_output_ulp=max_ulp, max_scaled_state_error=str(max_error))", "max_output_ulp=max_ulp, max_scaled_state_error=str(max_error), producer=producer['summary'])")
    source = source.replace("        for tag, row in events[:-1]:", "        for tag, row in events[:-2]:")
    source = source.replace("        metrics = []; memory = {}; cursor = prefix = 0", '''        need(len(events) >= 2 and events[-2][0] == 'PRODUCER_SUMMARY' and counts['PRODUCER_SUMMARY'] == 1, 'Producer summary required')
        producer_summary = events[-2][1]
        need(producer_summary == dict(starts=6, dones=6, acks=6, consumer_starts=6, consumer_acks=6, norm_starts=6, norm_dones=6, mac=18432, reads=665088, writes=1536), 'Full producer summary')
        producer_aliases = dict(starts='PSTART', dones='PDONE', acks='PACK', consumer_starts='CSTART', consumer_acks='CACK', norm_starts='PNSTART', norm_dones='PNDONE', mac='PMAC', reads='PR', writes='PW')
        need(all(producer_summary[k] == counts[t] for k,t in producer_aliases.items()), 'Producer summary matches actual events')
        metrics = []; memory = {}; cursor = prefix = 0''')
    source = source.replace("cursor == len(events)-1", "cursor == len(events)-2")
    source = source.replace('errors=3, rope=8', 'errors=3, rope=6').replace('source=1312', 'source=984').replace('values=256', 'values=192')
    source = source.replace('PASS_ROPE_ATTENTION_REAL_R1_TRANSCRIPT', 'PASS_INPUT_ATTENTION_REAL_R1_TRANSCRIPT')
    source = source.replace('summary=summary, metrics=metrics,', 'summary=summary, producer_summary=producer_summary, metrics=metrics,')
    start = source.index('    def controls(text, seed=1):')
    end = source.index('    return dict(review=review', start)
    source = source[:start]+'''    def controls(text, seed=1):
        events = parse(text); review_events(events, seed); rejected = []
        probes = [('PR','data'), ('PB','response'), ('PDONE','bytes'), ('CSTART','epoch'),
                  ('R','data'), ('ROPE_WRITE','data'), ('STATE','data'), ('OUTPUT','data'),
                  ('RESULT','data'), ('HEAD','lease'), ('COMPLETE','committed'), ('RELEASE','prefix')]
        for tag,key in probes:
            i=next(i for i,(kind,_) in enumerate(events) if kind==tag)
            altered=list(events);row=dict(events[i][1]);row[key]^=1;altered[i]=(tag,row)
            try:review_events(altered,seed)
            except (ValueError,AssertionError,KeyError,IndexError):rejected.append(tag+'.'+key)
            else:raise AssertionError('Accepted corrupted '+tag+'.'+key)
        for tag in ('PACK','CACK','PRODUCER_SUMMARY','PASS'):
            i=next(i for i,(kind,_) in enumerate(events) if kind==tag)
            try:review_events(events[:i]+events[i+1:],seed)
            except (ValueError,AssertionError,KeyError,IndexError):rejected.append(tag+'.missing')
            else:raise AssertionError('Accepted missing '+tag)
        return rejected

''' + source[end:]
    ast.parse(source)
    return source, hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    text, parent = render()
    with (ROOT / 'tools/check_ntrpu_input_attention_real_r1.py').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    print('PARENT_SHA256 '+parent)
