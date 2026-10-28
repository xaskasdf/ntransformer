"""R2 public transfer review plus independent native host CQ/CC packet checking."""
from check_ntrpu_xdma_stop_inventory_r6 import review as base_review, vcd_rows, markers, require


def review(sim, vcd):
    result=base_review(sim,vcd)
    requests=[]; beats=[]; start=result['start_ps']
    for t,before,after in vcd_rows(vcd):
        if t<start or not (before['/board/RP/user_clk']==0 and after['/board/RP/user_clk']==1):
            continue
        for prefix,destination in [('m_axis_cq',requests),('s_axis_cc',beats)]:
            path='/board/RP/'+prefix+'_'
            if before[path+'tvalid']==1 and before[path+'tready'] is not None and before[path+'tready']&1:
                row=[t]+[before[path+x] for x in ['tdata','tkeep','tlast','tuser']]
                require(None not in row,'known public host packet')
                destination.append(row)
    require(len(requests)==2 and len(beats)==3,'native descriptor/data request inventory')
    for row,(address,words) in zip(requests,[(0x100,8),(0x400,16)]):
        _,data,keep,last,_=row
        require(data&((1<<64)-1)==address and (data>>64)&0x7ff==words and (data>>75)&15==0,
                'native host memory request address/length/type')
        require(keep==15 and last==1,'native host memory request framing')
        require((data>>121)&7==0 and (data>>124)&7==2,
                'native host memory request traffic class and attributes')
    desc=bytearray(32)
    for offset,value in [(0,0xad4b0013),(4,64),(8,0x400),(16,0x1000)]:
        desc[offset:offset+4]=value.to_bytes(4,'little')
    data=bytes(((i*13)^0xa5)&255 for i in range(64))
    groups=[beats[:1],beats[1:]]
    for request,group,expected in zip(requests,groups,[bytes(desc),data]):
        qt,qd,*_=request; first=group[0][1]
        tag=(qd>>96)&255;requester=(qd>>80)&0xffff
        cc_tag=((first>>64)&255)|(((first>>31)&1)<<8)|(((first>>47)&1)<<9)
        require(qt<group[0][0] and cc_tag==tag and (first>>48)&0xffff==requester,
                'native host completion credit/tag/requester')
        require((first>>16)&0xfff==len(expected) and (first>>32)&0x7ff==len(expected)//4 and
                (first>>43)&15==0 and (first>>88)&1==1 and first&127==0,
                'native host completion header')
        require((first>>89)&7==(qd>>121)&7 and (first>>92)&7==(qd>>124)&7,
                'completion preserves native traffic class and attributes')
        payload=bytearray()
        for index,(_,word,keep,last,user) in enumerate(group):
            count=keep.bit_count()
            require(keep==(1<<count)-1,'contiguous CC dword mask')
            require(last==int(index==len(group)-1),'CC last boundary')
            require(user==((1 if index==0 else 0)|((64|((count-1)<<8)) if last else 0)),
                    'CC SOP/EOP sideband')
            raw=word.to_bytes(64,'little')
            payload.extend(raw[12 if index==0 else 0:count*4])
        require(bytes(payload)==expected,'complete native descriptor/data payload')
    require(groups[0][-1][0]<requests[1][0],'descriptor before source-data request')
    logs=markers(sim,'NTRPU_HOST_CPL_BEAT')
    require(len(logs)==len(beats),'host stimulus log count')
    for row,actual in zip(logs,beats):
        encoded=[int(row['t']),int(row['data'],16),int(row['keep'],16),int(row['last']),int(row['user'],16)]
        require(encoded==actual,'host stimulus log/public VCD mismatch')
    require([int(r['offset']) for r in logs]==[0,0,52],'CC payload segmentation')
    starts=markers(sim,'NTRPU_HOST_CPL_BEGIN');ends=markers(sim,'NTRPU_HOST_CPL_END')
    require(len(starts)==len(ends)==2,'host stimulus begin/end inventory')
    for begin,end,request,group,count in zip(starts,ends,requests,groups,[32,64]):
        require(int(begin['address'],16)==request[1]&((1<<64)-1) and
                int(begin['bytes'])==int(end['bytes'])==count and
                int(begin['tag'],16)==int(end['tag'],16)==(request[1]>>96)&255,
                'host stimulus request identity')
        require(int(begin['tc'])==0 and int(begin['attr'])==2,
                'host stimulus traffic class and attributes')
        require(request[0]<=int(begin['t'])<group[0][0]<=group[-1][0]<int(end['t']),
                'host stimulus lifetime')
    result.update(public_host_completion_driver=True,native_host_read_requests=len(requests),
                  native_host_completion_beats=len(beats),host_descriptor_bytes=32,host_source_bytes=64,
                  host_completion_review='CQ request credit, CC framing/tag/requester and every descriptor/data byte checked from VCD.')
    return result
