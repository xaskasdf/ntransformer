"""One complete actual attention/O/post-O vector; downstream FFN review is separate."""
import re
import check_ntrpu_attention_projected_root_real_r2 as root_check
import check_ntrpu_attention_post_real_r2 as post
from check_ntrpu_attention_projected_root_real_r2 import ABASE,OBASE,WBASE,bursts,review_root

def bucket(rows,schema):
    e={k:[] for k in schema};previous=-1
    for k,row in rows:
        assert k in schema and set(row)==set(schema[k].split())
        if 'cycle' in row:
            assert previous<=row['cycle'];previous=row['cycle']
        if 'job' in row:assert row['job']==1
        e[k].append(row)
    return e

def review_projection(rows,fixture,seed=1):
    e=bucket(rows,root_check.SCHEMA)
    assert e['CONFIG']==[dict(seed=seed,rows=2048,k_blocks=64,tile_k=32,watchdog=65536)] and not e['PASS']
    j=1;mode=0;sample_id=0;seq=0x1234567800000001;epoch=0xfedcba9800000001;tag=0xffffffc0
    for name in ('BEGIN','COMPLETE','RELEASE'):assert len(e[name])==1
    begin=e['BEGIN'][0];end=e['COMPLETE'][0]
    assert (begin['mode'],begin['sample'],begin['sequence'],begin['epoch'],begin['tag'])==(0,0,seq,epoch,tag)
    assert (end['mode'],end['status'],end['sequence'],end['epoch'],end['valid'],end['bytes'])==(0,0,seq,epoch,1,8192)
    root_metrics=review_root(e,fixture,j,mode,sample_id)
    totals={k:0 for k in ('ar','r','aw','w','b','results')}
    for name in ('MATRIX','MATRIX_DONE','PROJECTION_COMMAND','QUANT_PUBLIC','QACK','PACK','START','DONE'):assert len(e[name])==1,(j,name)
    matrix=e['MATRIX'][0];done=e['MATRIX_DONE'][0]
    assert matrix['tag']==done['tag']==tag and matrix['cycle']==e['PROJECTION_COMMAND'][0]['cycle']+1
    assert (done['status'],done['bytes'],done['rows'])==(0,8192,2048)
    assert done['cycle']>matrix['cycle'] and end['cycle']>=done['cycle']+7
    sample=fixture['samples'][sample_id]
    reads_q=e['READ'];inputs_q=e['INPUT'];words_q=e['WORD']
    assert len(reads_q)>=2048 and len(inputs_q)==len(reads_q)
    assert set(r['index'] for r in reads_q)==set(range(2048))
    requested={(r['cycle']+1,r['index']):sample['input_fp32'][r['index']] for r in reads_q}
    assert len(requested)==len(reads_q)
    for row in inputs_q:assert row['data']==requested[(row['cycle'],row['index'])]
    assert len({(r['cycle'],r['index']) for r in inputs_q})==len(inputs_q)
    assert len(words_q)==512 and [r['index'] for r in words_q]==list(range(512))
    assert [r['data'] for r in words_q]==sample['abi_words']
    qdone=e['DONE'][0];root=e['ROOT_ACK'][0];qp=e['QUANT_PUBLIC'][0];pc=e['PROJECTION_COMMAND'][0]
    assert e['START'][0]['cycle']>e['ROOT_PUBLIC'][0]['cycle'] and qdone['status']==0 and qdone['words']==512
    assert all(e['START'][0]['cycle']<=r['cycle']<=qdone['cycle'] for r in reads_q+inputs_q+words_q)
    assert (root['sequence'],root['epoch'])==(seq,epoch) and root['cycle']>qdone['cycle']
    assert root['cycle']<pc['cycle']
    assert (qp['status'],qp['hls'],qp['sequence'],qp['epoch'],qp['records'],qp['valid'],qp['address'],qp['bytes'])==(0,0,seq,epoch,64,1,ABASE+4032,4096)
    assert (pc['sequence'],pc['epoch'],pc['address'])==(seq,epoch,ABASE+4032) and pc['cycle']>qp['cycle']
    for kind in ('QAW','QW','QB'):
        assert len(e[kind])==64 and [r['record'] for r in e[kind]]==list(range(64))
    for index in range(64):
        aw=e['QAW'][index];w=e['QW'][index];reply=e['QB'][index]
        assert aw['address']==ABASE+4032+index*64 and w['data']==sample['activation'][index] and reply['response']==0
        assert root['cycle']<min(aw['cycle'],w['cycle']) and reply['cycle']>max(aw['cycle'],w['cycle'])
        if index:assert e['QB'][index-1]['cycle']<min(aw['cycle'],w['cycle'])
    assert qp['cycle']>e['QB'][-1]['cycle']
    assert e['QACK'][0]['cycle']==e['PACK'][0]['cycle']==e['RELEASE'][0]['cycle']>end['cycle']
    for kind in ('QACK','PACK'):assert (e[kind][0]['sequence'],e[kind][0]['epoch'])==(seq,epoch)
    ng=64 if mode==0 else 0
    assert len(e['GROUP'])==len(e['GROUP_DONE'])==ng
    assert len(e['TILE'])==(128 if mode==0 else 0)
    for index,row in enumerate(e['GROUP']):
        assert (row['group'],row['tag'])==(index,tag+index)
        assert matrix['cycle']<row['cycle']<done['cycle']
        if index:assert row['cycle']>e['GROUP_DONE'][index-1]['cycle']
    for index,row in enumerate(e['TILE']):
        assert (row['tile'],row['address'])==(index,WBASE+4032+index*34816)
        assert e['GROUP'][index//2]['cycle']<row['cycle']<e['GROUP_DONE'][index//2]['cycle']
    for index,row in enumerate(e['GROUP_DONE']):
        assert (row['group'],row['status'],row['bytes'])==(index,0,128)
        assert e['GROUP'][index]['cycle']<row['cycle']<done['cycle']
    expected_ar=[]
    for group in range(ng):
        for address,count in [(ABASE+4032,4096),(WBASE+4032+group*69632,34816),(WBASE+4032+group*69632+34816,34816)]:
            expected_ar += [(address,beats,group) for address,beats in bursts(address,count)]
    ars=e['AR'];reads=e['R'];cursor=0
    if mode==0:assert len(ars)==1344 and len(reads)==73728
    if mode==1:assert not ars and not reads
    for index,row in enumerate(ars):
        assert index<len(expected_ar)
        address,beats,group=expected_ar[index]
        assert (row['ar'],row['address'],row['beats'],row['group'])==(index,address,beats,group)
        assert e['GROUP'][group]['cycle']<row['cycle']<e['GROUP_DONE'][group]['cycle']
        if index:assert row['cycle']>reads[cursor-1]['cycle'],'More than one read outstanding'
        if address>=ABASE:
            values=sample['activation'];word0=(address-ABASE-4032)//64
        else:
            values=fixture['weight'];word0=(address-WBASE)//64
            tile=(address-WBASE-4032)//34816
            assert e['TILE'][tile]['cycle']<row['cycle']
        for beat in range(beats):
            assert cursor<len(reads),'Undrained read burst'
            r=reads[cursor];cursor+=1
            assert (r['ar'],r['beat'],r['address'],r['data'],r['last'])==(index,beat,address+beat*64,values[word0+beat],int(beat==beats-1))
            assert row['cycle']<r['cycle']<e['GROUP_DONE'][group]['cycle']
            if beat:assert reads[cursor-2]['cycle']<r['cycle']
            if address>=ABASE:
                qi=(r['address']-ABASE-4032)//64
                assert r['data']==e['QW'][qi]['data'] and e['QB'][qi]['cycle']<r['cycle']<e['QACK'][0]['cycle']

    assert cursor==len(reads),'Unrequested read response'
    mac=e['MAC'];assert len(mac)==(2048 if mode==0 else 0)
    for index,row in enumerate(mac):
        assert (row['row'],row['data'])==(index,sample['fp64'][index])
        group=index//32
        assert reads[(group+1)*1152-1]['cycle']<row['cycle']<e['GROUP_DONE'][group]['cycle']
        if index:assert row['cycle']>mac[index-1]['cycle']
    expected_aw=[]
    if mode==0:
        for group in range(64):expected_aw += [(address,beats,group) for address,beats in bursts(OBASE+4032+128*group,128)]
    assert len(e['AW'])==len(e['B'])==len(expected_aw)
    assert len(e['W'])==(128 if mode==0 else 0)
    cursor=0
    for index,(address,beats,group) in enumerate(expected_aw):
        aw=e['AW'][index];b=e['B'][index];words=e['W'][cursor:cursor+beats]
        assert (aw['aw'],aw['address'],aw['beats'])==(index,address,beats)
        assert (b['aw'],b['response'])==(index,0)
        assert mac[(group+1)*32-1]['cycle']<aw['cycle']<b['cycle']<e['GROUP_DONE'][group]['cycle']
        if index:assert e['B'][index-1]['cycle']<aw['cycle']
        assert len(words)==beats
        for beat,row in enumerate(words):
            assert (row['word'],row['data'],row['last'])==(cursor+beat,sample['fp32'][cursor+beat],int(beat==beats-1))
            assert mac[(group+1)*32-1]['cycle']<row['cycle']<b['cycle']
            if index:assert row['cycle']>e['B'][index-1]['cycle']
            if beat:assert row['cycle']>words[beat-1]['cycle']
        cursor+=beats
    assert not e['CANCEL']
    bus_cycles=sorted({r['cycle'] for k in ('AR','R','AW','W','B') for r in e[k]})
    start=pc['cycle']+1;stop=done['cycle']
    assert all(start<=c<=stop for c in bus_cycles)
    gaps=([bus_cycles[0]-start]+[b-a-1 for a,b in zip(bus_cycles,bus_cycles[1:])]+[stop-bus_cycles[-1]]) if bus_cycles else [stop-start+1]
    assert end['max_quiet']==max(gaps)<65536,'Watchdog gap differs from trace'
    for key,kind in [('ar','AR'),('r','R'),('aw','AW'),('w','W'),('b','B'),('results','MAC'),('groups','GROUP'),('tiles','TILE')]:
        assert end[key]==len(e[kind]),(j,key)
        if key in totals:totals[key]+=len(e[kind])

    assert totals==dict(ar=1344,r=73728,aw=66,w=128,b=66,results=2048)
    return dict(root=root_metrics,totals=totals,matrix_cycles=stop-matrix['cycle']+1,max_quiet_cycles=max(gaps),
        quantized_input_words=len(words_q),fp32_output_bytes=8192,upstream_lease_ack_cycle=e['RELEASE'][0]['cycle'])

def review_post(rows,root_rows,values):
    e=bucket(rows,post.SCHEMA);r=bucket(root_rows,root_check.SCHEMA);sample=0
    assert not e['PASS'] and len(e['COMPLETE'])==1
    c=e['COMPLETE'][0];rc=r['COMPLETE'][0]
    assert c['cycle']+7==rc['cycle'] and (c['status'],c['valid'])==(rc['status'],rc['valid'])
    assert (c['ar'],c['r'],c['b'])==tuple(len(e[k]) for k in ('AR','R','B'))
    assert c['hls']==0 and 0<=c['max_quiet']<262144
    assert c['status']==0 and c['bytes']==12288
    assert all(len(e[k])==1 for k in ('HANDOFF','START','DONE'))
    h=e['HANDOFF'][0];start=e['START'][0];done=e['DONE'][0]
    assert (h['sequence'],h['epoch'],h['address'],h['bytes'])==(rc['sequence'],rc['epoch'],root.OBASE+4032,8192)
    assert r['MATRIX_DONE'][0]['cycle']<h['cycle']<start['cycle']<done['cycle']<c['cycle']
    assert done['return']==0 and 0<done['cycle']-start['cycle']<262144
    # Reconstruct the only legal post-O branch from accepted root W and B.
    committed={};cursor=0
    for aw,b in zip(r['AW'],r['B']):
        assert b['response']==0 and b['cycle']<h['cycle']
        for word in r['W'][cursor:cursor+aw['beats']]:
            address=aw['address']+(word['word']-cursor)*64
            assert address not in committed and word['cycle']<b['cycle']
            committed[address]=(word['data'],b['cycle'])
        cursor+=aw['beats']
    assert len(committed)==128
    bases=[0xa100000fc0,0xa000000fc0,0xa200000fc0];cursor=0
    for ar in e['AR']:
        plane=cursor//128;offset=cursor%128
        assert plane<3 and ar['address']==bases[plane]+64*offset
        beats=min(128-offset,(4096-ar['address']%4096)//64)
        assert ar['beats']==beats and h['cycle']<ar['cycle']<start['cycle']
        if cursor: assert e['R'][cursor-1]['cycle']<ar['cycle']
        for beat in range(beats):
            reply=e['R'][cursor];address=ar['address']+beat*64
            assert (reply['index'],reply['address'])==(cursor,address)
            assert ar['cycle']<reply['cycle']<start['cycle']
            if cursor: assert e['R'][cursor-1]['cycle']<reply['cycle']
            if plane==1:
                want,commit_cycle=committed[address]
                assert commit_cycle<reply['cycle']
            else:
                index=sample*6144+plane*2048+(offset+beat)*16
                want=sum(values['source32.hex'][index+lane]<<(32*lane) for lane in range(16))
            assert reply['data']==want
            cursor+=1
    assert cursor==len(e['R'])==384 and len(e['AR'])==9 and len(e['B'])==192
    for i,b in enumerate(e['B']):
        address=0xa300000fc0+64*i if i<128 else 0xa400000fc0+64*(i-128)
        assert (b['index'],b['address'])==(i,address) and done['cycle']<b['cycle']<c['cycle']
        if i: assert e['B'][i-1]['cycle']<b['cycle']
        if i<128:
            want=sum(values['residual32.hex'][sample*2048+i*16+k]<<(32*k) for k in range(16))
        else:
            want=sum(values['abi64.hex'][sample*512+(i-128)*8+k]<<(64*k) for k in range(8))
        assert b['data']==want
    assert r['OUT_ACK'][0]['cycle']>c['cycle']

    return dict(hls_cycles=done['cycle']-start['cycle'],last_B_cycle=e['B'][-1]['cycle'],
        committed_O_bytes_consumed=8192,post_read_bytes=24576,post_written_bytes=12288,
        source_beats=[dict(b) for b in e['B']])

def review_events(root_rows,post_rows,values,seed=1):
    return dict(projected_root=review_projection(root_rows,values['root'],seed),post=review_post(post_rows,root_rows,values['post']),
        scope='Upstream one-vector proof only. Does not qualify the integrated FFN/residual or physical timing.')

def review(text,values,seed=1):
    assert not re.search(r'(?im)^\s*(?:Fatal:|FATAL:|ERROR:|AFREAL_FAIL|PRREAL_FAIL|APREAL_FAIL)',text)
    return review_events(root_check.parse(text),post.parse(text),values,seed)
