"""Independent root/KV materialization checks for the full real projected-root bank."""
import hashlib
from pathlib import Path

KBASE=0xc000000000
ROOT_HASHES={
    'kv_beats.hex':'79135a864a639f34cdd25ac63eadee02e33fc0230d5c73606050d1cde0032b07',
    'query.hex':'187f12b896fe1c47ba2c2f35c5d0c35320919add32fceacc95f4c6a606821f44',
    'root-expected.hex':'8105700f1e9cde8cf4a91ca4b8798bb4354d6baae474bc9d3a9d7a9abf2774aa'}
SCHEMA={
 'COMMAND':'cycle job position sequence epoch lease','Q':'cycle job index data',
 'HEAD':'cycle job head sequence epoch lease','COPY':'cycle job head index data',
 'HEAD_DONE':'cycle job head status reads','ROOT_PUBLIC':'cycle job status sequence epoch valid',
 'BIND':'cycle job base capacity sequence epoch','BIND_DONE':'cycle job',
 'APPEND':'cycle job position','DATA':'cycle job position index data',
 'APPEND_DONE':'cycle job position status bytes committed',
 'KAR':'cycle job ar address beats head','KR':'cycle job ar beat address data last',
 'KAW':'cycle job aw position address beats','KW':'cycle job position index data last',
 'KB':'cycle job aw response','OUT_ACK':'cycle job'}

def load_root_fixture(folder,fixture):
    values={}
    for name,sha in ROOT_HASHES.items():
        raw=(Path(folder)/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha,name
        values[name]=[int(word,16) for word in raw.split()]
    expected=values['root-expected.hex']
    assert len(expected)==8192
    assert expected==[word for sample in fixture['samples'] for word in sample['input_fp32']]
    fixture['root']=values
    return fixture

def bursts(address,count):
    result=[]
    while count:
        beats=min(count//64,(4096-address%4096)//64)
        assert 0<beats<=64
        result.append((address,beats));address+=64*beats;count-=64*beats
    return result

def review_root(e,fixture,j,mode,sample_id):
    for kind in ('BIND','BIND_DONE','COMMAND','ROOT_PUBLIC','ROOT_ACK','BEGIN','COMPLETE','RELEASE'):
        assert len(e[kind])==1,(j,kind)
    begin=e['BEGIN'][0];end=e['COMPLETE'][0];release=e['RELEASE'][0]
    seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j;lease=0xffffffe0+(j-1)*64
    bind=e['BIND'][0];bound=e['BIND_DONE'][0];command=e['COMMAND'][0];public=e['ROOT_PUBLIC'][0];ack=e['ROOT_ACK'][0]
    assert (bind['base'],bind['capacity'],bind['sequence'],bind['epoch'])==(KBASE+4032,4,seq,epoch)
    assert begin['cycle']<=bind['cycle']<bound['cycle']
    assert (command['position'],command['sequence'],command['epoch'],command['lease'])==(sample_id,seq,epoch,lease)
    assert (public['status'],public['sequence'],public['epoch'],public['valid'])==(7 if mode else 0,seq,epoch,int(mode==0))
    assert (ack['sequence'],ack['epoch'])==(seq,epoch)
    assert command['cycle']<public['cycle']<ack['cycle']<=release['cycle'] and end['cycle']<release['cycle']
    assert public['cycle']+7<=end['cycle']
    assert len(e['APPEND'])==len(e['APPEND_DONE'])==4 and len(e['DATA'])==256
    assert len(e['KAW'])==len(e['KB'])==68 and len(e['KW'])==256
    materialized={};aw_cursor=0;w_cursor=0
    for position in range(4):
        start=e['APPEND'][position];done=e['APPEND_DONE'][position]
        assert start['position']==position
        assert (done['position'],done['status'],done['bytes'],done['committed'])==(position,0,4096,position+1)
        assert bound['cycle']<=start['cycle']<done['cycle']<command['cycle']
        if position:assert e['APPEND_DONE'][position-1]['cycle']<start['cycle']
        data=e['DATA'][position*64:(position+1)*64]
        for index,row in enumerate(data):
            assert (row['position'],row['index'],row['data'])==(position,index,fixture['root']['kv_beats.hex'][position*64+index])
            assert start['cycle']<row['cycle']<done['cycle']
            if index:assert row['cycle']>data[index-1]['cycle']
        word=0
        for plane_head in range(16):
            for address,beats in bursts(KBASE+4032+(plane_head*4+position)*256,256):
                aw=e['KAW'][aw_cursor];reply=e['KB'][aw_cursor]
                assert (aw['aw'],aw['position'],aw['address'],aw['beats'])==(aw_cursor,position,address,beats)
                assert (reply['aw'],reply['response'])==(aw_cursor,0)
                assert start['cycle']<aw['cycle']<reply['cycle']<done['cycle']
                if aw_cursor:assert e['KB'][aw_cursor-1]['cycle']<aw['cycle']
                for beat in range(beats):
                    w=e['KW'][w_cursor]
                    assert (w['position'],w['index'],w['data'],w['last'])==(position,word,data[word]['data'],int(beat==beats-1))
                    assert data[word]['cycle']<w['cycle']<reply['cycle']
                    if aw_cursor:assert e['KB'][aw_cursor-1]['cycle']<w['cycle']
                    if beat:assert e['KW'][w_cursor-1]['cycle']<w['cycle']
                    key=address+64*beat;assert key not in materialized
                    materialized[key]=(w['data'],reply['cycle'])
                    word+=1;w_cursor+=1
                aw_cursor+=1
        assert word==64
    assert aw_cursor==68 and w_cursor==256 and len(materialized)==256
    nq=8 if mode else 128
    assert len(e['Q'])==nq
    query=fixture['root']['query.hex'][sample_id*2048:(sample_id+1)*2048]
    for index,row in enumerate(e['Q']):
        value=sum(query[index*16+lane]<<(32*lane) for lane in range(16))
        assert (row['index'],row['data'])==(index,value)
        assert command['cycle']<row['cycle']<public['cycle']
        if index:assert e['Q'][index-1]['cycle']<row['cycle']
    if mode:
        assert len(e['CANCEL'])==1 and e['CANCEL'][0]['qbeats']==8
        assert e['Q'][-1]['cycle']<=e['CANCEL'][0]['cycle']<public['cycle']
        assert ack['cycle']==release['cycle']
        for kind in ('HEAD','HEAD_DONE','COPY','KAR','KR','OUT_ACK'):assert not e[kind],kind
        return dict(heads=0,copies=0,q_beats=8,kv_read_bytes=0,kv_written_bytes=16384)
    assert not e['CANCEL'] and len(e['HEAD'])==len(e['HEAD_DONE'])==32 and len(e['COPY'])==2048
    assert len(e['KAR'])==80 and len(e['KR'])==(sample_id+1)*256
    assert len(e['OUT_ACK'])==1 and e['OUT_ACK'][0]['cycle']==release['cycle']
    ar_cursor=0;r_cursor=0
    for head in range(32):
        start=e['HEAD'][head];done=e['HEAD_DONE'][head]
        assert (start['head'],start['sequence'],start['epoch'],start['lease'])==(head,seq,epoch,lease+head)
        assert (done['head'],done['status'],done['reads'])==(head,0,64)
        assert e['Q'][-1]['cycle']<start['cycle']<done['cycle']<public['cycle']
        if head:assert e['HEAD_DONE'][head-1]['cycle']<start['cycle']
        for plane in range(2):
            base=KBASE+4032+(plane*8+head//4)*4*256
            for address,beats in bursts(base,(sample_id+1)*256):
                ar=e['KAR'][ar_cursor]
                assert (ar['ar'],ar['address'],ar['beats'],ar['head'])==(ar_cursor,address,beats,head)
                assert start['cycle']<ar['cycle']<done['cycle']
                if ar_cursor:assert e['KR'][r_cursor-1]['cycle']<ar['cycle']
                for beat in range(beats):
                    r=e['KR'][r_cursor];key=address+beat*64;value,commit=materialized[key]
                    assert (r['ar'],r['beat'],r['address'],r['data'],r['last'])==(ar_cursor,beat,key,value,int(beat==beats-1))
                    assert commit<ar['cycle']<r['cycle']<done['cycle']
                    if beat:assert e['KR'][r_cursor-1]['cycle']<r['cycle']
                    r_cursor+=1
                ar_cursor+=1
        copies=e['COPY'][head*64:(head+1)*64]
        for index,row in enumerate(copies):
            assert (row['head'],row['index'],row['data'])==(head,index,fixture['samples'][sample_id]['input_fp32'][head*64+index])
            assert e['KR'][r_cursor-1]['cycle']<row['cycle']<done['cycle']
            if index:assert copies[index-1]['cycle']<row['cycle']
    assert ar_cursor==80 and r_cursor==len(e['KR'])
    return dict(heads=32,copies=2048,q_beats=128,kv_read_bytes=r_cursor*64,kv_written_bytes=16384,
                root_completion_cycles=public['cycle']-command['cycle'])
