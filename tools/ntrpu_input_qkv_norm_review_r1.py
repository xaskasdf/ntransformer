"""Reconcile normalization math, actual AXI commits, and every live ABI1 read."""
SCHEMA={
    'NSTART':'cycle job','NDONE':'cycle job status words','NWRITE':'cycle job index data',
    'NAR':'cycle job ar address beats','NR':'cycle job ar beat address data response last',
    'NAW':'cycle job aw address beats','NW':'cycle job word data last','NB':'cycle job aw response',
    'NCANCEL':'cycle job reads',
    'NCOMPLETE':'cycle job status hls reads records starts dones words ar r aw w b',
    'NSUMMARY':'starts dones words ar r w b'}

def norm_review(e,values,sample,mode,begin,public,complete):
    assert len(e['NCOMPLETE'])==1
    diag=e['NCOMPLETE'][0]
    active=mode!=1;hls=mode not in (1,4);valid=mode in (0,2,3)
    assert len(e['NAR'])==(0 if mode==1 else 2 if mode==4 else 6)
    assert len(e['NR'])==(0 if mode==1 else 65 if mode==4 else 256)
    words=values['source32.hex'][sample*4096:(sample+1)*4096]
    abi=values['abi64.hex'][sample*512:(sample+1)*512]
    cursor=0
    for index,ar in enumerate(e['NAR']):
        plane=cursor//128;address=0x6000000000+plane*0x10000000+4032+(cursor%128)*64
        beats=min(128-cursor%128,(4096-address%4096)//64)
        assert (ar['ar'],ar['address'],ar['beats'])==(index,address,beats)
        assert ar['cycle']>begin['cycle']
        if cursor:assert ar['cycle']>e['NR'][cursor-1]['cycle']
        for beat in range(beats):
            row=e['NR'][cursor];lanes=words[cursor*16:(cursor+1)*16].copy()
            if mode==5:
                for lane in range(16):
                    if (cursor*16+lane)%2048==17:lanes[lane]=0x7f7fffff
            value=sum(w<<(32*i) for i,w in enumerate(lanes))
            assert (row['ar'],row['beat'],row['address'],row['data'],row['response'],row['last'])==(
                index,beat,address+beat*64,value,2 if mode==4 and cursor==3 else 0,int(beat==beats-1))
            assert row['cycle']>ar['cycle']
            if cursor:assert row['cycle']>e['NR'][cursor-1]['cycle']
            cursor+=1
    assert cursor==len(e['NR'])
    assert len(e['NSTART'])==len(e['NDONE'])==int(hls)
    assert len(e['NWRITE'])==(512 if hls and mode!=5 else 0)
    if hls:
        start=e['NSTART'][0];done=e['NDONE'][0]
        assert e['NR'][-1]['cycle']<start['cycle']<done['cycle']<diag['cycle']
        assert (done['status'],done['words'])==(4 if mode==5 else 0,len(e['NWRITE']))
        for i,row in enumerate(e['NWRITE']):
            assert row['index']==i and start['cycle']<row['cycle']<=done['cycle']
            if i:assert e['NWRITE'][i-1]['cycle']<row['cycle']
            if mode!=6:assert row['data']==abi[i],'Real norm arithmetic'
            elif i%8<4:assert all(v!=128 for v in row['data'].to_bytes(8,'little'))
            elif i%8==4:assert row['data']<=0x4780204070e1c387
            else:assert row['data']==0
    assert len(e['NCANCEL'])==int(mode==6)
    if mode==6:
        cancel=e['NCANCEL'][0]
        assert cancel['reads']>=32 and e['NSTART'][0]['cycle']<cancel['cycle']<e['NDONE'][0]['cycle']
        assert cancel['cycle']<e['NWRITE'][0]['cycle']
    assert len(e['NAW'])==len(e['NW'])==len(e['NB'])==(64 if valid else 0)
    for index,(aw,w,b) in enumerate(zip(e['NAW'],e['NW'],e['NB'])):
        address=0x8000000000+4032+index*64;data=sum(abi[index*8+i]<<(64*i) for i in range(8))
        assert (aw['aw'],aw['address'],aw['beats'])==(index,address,1)
        assert (w['word'],w['data'],w['last'])==(index,data,1)
        assert (b['aw'],b['response'])==(index,0)
        assert w['data']==sum(e['NWRITE'][index*8+i]['data']<<(64*i) for i in range(8)),'HLS-to-AXI bytes'
        assert e['NDONE'][0]['cycle']<aw['cycle']<b['cycle']<diag['cycle']
        assert e['NDONE'][0]['cycle']<w['cycle']<b['cycle']
        if index:assert e['NB'][index-1]['cycle']<min(aw['cycle'],w['cycle'])
    for row in e['R']:
        if 0x8000000fc0<=row['address']<0x8000001fc0:
            index=(row['address']-0x8000000fc0)//64
            assert valid and len(e['NB'])==64
            assert row['data']==e['NW'][index]['data'],'QKV must read committed norm bytes'
            assert e['NB'][-1]['cycle']<row['cycle'] and e['NB'][index]['cycle']<row['cycle'],'No ABI1 read before commit'
    if e['MATRIX']:assert e['NB'][-1]['cycle']<e['MATRIX'][0]['cycle']
    counters=dict(starts=int(hls),dones=int(hls),words=len(e['NWRITE']),ar=len(e['NAR']),r=len(e['NR']),
                  aw=len(e['NAW']),w=len(e['NW']),b=len(e['NB']))
    assert all(diag[k]==v for k,v in counters.items()),'Norm counters'
    assert (diag['status'],diag['hls'],diag['reads'],diag['records'])==(
        3 if mode==4 else 5 if mode==5 else 7 if mode==6 else 0,4 if mode==5 else 0,
        0 if not active else 3 if mode==4 else 256,64 if valid else 0)
    assert begin['cycle']<diag['cycle']<complete['cycle']<public['cycle']
    if active:assert e['NR'][-1]['cycle']<diag['cycle']
    return dict(counters,exact_words=512 if valid else 0,
        hls_cycles=e['NDONE'][0]['cycle']-e['NSTART'][0]['cycle'] if hls else None,
        last_commit_to_first_matrix_cycles=e['MATRIX'][0]['cycle']-e['NB'][-1]['cycle'] if e['MATRIX'] else None)
