"""Strict 2026.1 HLS TV decoder, derived from the archived generated Output writer.

Binary memory frames: big-endian uint64 depth, then depth big-endian words;
one 5a5aa5a50f0ff0f0 trailer terminates the file. No binary transaction IDs
are serialized, so the external case count/order must also be validated.
"""
import re
import base64
import hashlib

BMARK=bytes.fromhex('5a5aa5a50f0ff0f0')

def transactions(data,count,*,width,depth=None):
    assert isinstance(data,bytes) and count>0 and width>0 and width%8==0
    digits=(width+3)//4
    if data.startswith(b'[[[runtime]]]'):
        lines=[s.strip() for s in data.decode('ascii').splitlines() if s.strip()]
        assert lines[0]=='[[[runtime]]]' and lines[-1]=='[[[/runtime]]]'
        result=[];p=1
        while p<len(lines)-1:
            match=re.fullmatch(r'\[\[transaction\]\]\s+(\d+)',lines[p]);assert match and int(match[1])==len(result)
            p+=1;frame=[]
            while p<len(lines)-1 and lines[p]!='[[/transaction]]':
                assert re.fullmatch(r'0x[0-9a-fA-F]+',lines[p]),'Unknown or malformed scalar word'
                value=int(lines[p],16);assert value<1<<width
                frame.append(f'0x{value:0{digits}x}');p+=1
            assert p<len(lines)-1 and lines[p]=='[[/transaction]]';p+=1
            if depth is not None:assert len(frame)==depth
            result.append(frame)
        assert p==len(lines)-1 and len(result)==count
        return result
    assert depth is not None and depth>0, 'Binary memory depth must come from the interface'
    frame_bytes=8+depth*(width//8)
    assert len(data)==count*frame_bytes+8 and data[-8:]==BMARK,'Binary framing/length mismatch'
    frames=[]
    for index in range(count):
        offset=index*frame_bytes
        assert int.from_bytes(data[offset:offset+8],'big')==depth,('Binary depth mismatch',index)
        values=data[offset+8:offset+frame_bytes]
        frames.append([f'0x{int.from_bytes(values[i:i+width//8],"big"):0{digits}x}' for i in range(0,len(values),width//8)])
    return frames

def text_transactions(frames):
    lines=['[[[runtime]]]']
    for index,words in enumerate(frames):
        lines += [f'[[transaction]] {index}',*words,'[[/transaction]]']
    return '\n'.join([*lines,'[[[/runtime]]]',''])

SCHEMAS={
    'ntrpu_rmsnorm_fp32_r1':(60,{'input_r':(32,2048),'weight':(32,2048)},{'output_r':(32,2048)}),
    'ntrpu_rope_table_fp32_r1':(48,{'input_r':(32,2560),'coefficients':(32,64)},{'output_r':(32,2560)}),
    'ntrpu_residual_fp32_r1':(43,{'residual':(32,2048),'branch':(32,2048)},{'output_r':(32,2048)}),
    'ntrpu_norm_q8_abi1_r1':(61,{'input_r':(32,2048),'weight':(32,2048)},{'output_r':(64,512)}),
    'ntrpu_attention_tile_fp32_r1':(291,{'query':(32,64),'keys':(32,4096),'values':(32,4096),'state_input':(64,104)},
        {'state_output':(64,104),'output_r':(32,64)}),
}

def normalize_trace_files(encoded,top):
    """Decode bounded memory frames and separately verify all read-only echoes.

The established numerical checkers receive only the output files they expect.
Every excluded RTL output must be a declared read-only RAM, with an identical
C input frame. Raw files and hashes remain in the evidence/archive.
"""
    count,readonly,outputs=SCHEMAS[top]
    raw={n:base64.b64decode(v,validate=True) for n,v in encoded.items()}
    prefix='output/project/solution1/'
    latency=prefix+'sim/report/verilog/result.transaction.rpt'
    normalized={latency:raw[latency].decode('ascii')}
    selected_by_side={};facts=[];echoes=[]
    for side,sub in [('c','cdatafile'),('rtl','rtldatafile')]:
        stem=prefix+'sim/tv/'+sub+'/'+side+'.'+top+'.autotvout_'
        selected={n.removeprefix(stem).removesuffix('.dat'):n for n in raw if n.startswith(stem) and n.endswith('.dat')}
        assert set(selected)=={'ap_return',*outputs,*(readonly if side=='rtl' else {})},(top,side,set(selected))
        selected_by_side[side]=selected
        for port in ('ap_return',*outputs):
            n=selected[port];width,depth=(32,1) if port=='ap_return' else outputs[port]
            frames=transactions(raw[n],count,width=width,depth=depth)
            normalized[n]=text_transactions(frames)
            facts.append(dict(path=n,sha256=hashlib.sha256(raw[n]).hexdigest(),bytes=len(raw[n]),
                encoding='text' if raw[n].startswith(b'[[[runtime]]]') else 'binary_be_depth_words',
                width=width,depth=depth,frames=count,normalized_sha256=hashlib.sha256(normalized[n].encode()).hexdigest()))
    for port,(width,depth) in readonly.items():
        c=prefix+'sim/tv/cdatafile/c.'+top+'.autotvin_'+port+'.dat'
        rtl=selected_by_side['rtl'][port]
        left=transactions(raw[c],count,width=width,depth=depth)
        right=transactions(raw[rtl],count,width=width,depth=depth)
        assert left==right,('Read-only input changed',port)
        echoes.append(dict(port=port,c_path=c,rtl_path=rtl,c_sha256=hashlib.sha256(raw[c]).hexdigest(),
            rtl_sha256=hashlib.sha256(raw[rtl]).hexdigest(),frames=count,words_per_frame=depth))
    return normalized,dict(output_frames=facts,readonly_echoes=echoes,
        raw_files={n:dict(sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for n,v in raw.items()},
        schema_top=top,transaction_count=count,binary_unknown_bit_masks_available=False,
        scope='Binary memory words are two-state serialized values; no proof of absence of X/Z from these files alone.')
