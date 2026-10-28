"""Bind actual co-simulation input frames to the independently checked fixture."""
import base64
import hashlib
from read_ntrpu_hls_tv_r2 import transactions,text_transactions

PREFIX='output/project/solution1/sim/tv/cdatafile/c.ntrpu_quantize_abi1_r1.autotvin_'
PORTS={'input_r':(32,2048),'output_r':(64,512),'elements':(32,1)}

def review(encoded,jobs):
    assert len(jobs)==27
    wanted={'input_r':[[f'0x{w:08x}' for w in j['input']] for j in jobs],
            'elements':[[f"0x{j['elements']:08x}"] for j in jobs],
            'output_r':[['0xa55a5aa53cc3c33c']*512 for _ in jobs]}
    facts=[]
    for name,(width,depth) in PORTS.items():
        path=PREFIX+name+'.dat';raw=base64.b64decode(encoded[path],validate=True)
        assert transactions(raw,27,width=width,depth=depth)==wanted[name],('Actual fixture input differs',name)
        facts.append(dict(port=name,sha256=hashlib.sha256(raw).hexdigest(),frames=27,depth=depth))
    return dict(status='PASS_ATTENTION_QUANTIZE_INPUT_BINDING_R1',facts=facts,input_words=55296,
        real_attention_frames=4,output_initializer_words=13824)

def controls(encoded,jobs):
    rejected=[]
    for label,port,index,swap in [('input_bit','input_r',0,False),('real_frame_order','input_r',12,True),
                                ('elements','elements',0,False),('output_initializer','output_r',0,False)]:
        path=PREFIX+port+'.dat';width,depth=PORTS[port]
        frames=transactions(base64.b64decode(encoded[path],validate=True),27,width=width,depth=depth)
        if swap:frames[index],frames[index+1]=frames[index+1],frames[index]
        else:frames[index][0]=f'0x{int(frames[index][0],16)^1:0{width//4}x}'
        changed=dict(encoded);changed[path]=base64.b64encode(text_transactions(frames).encode()).decode()
        try:review(changed,jobs)
        except (AssertionError,ValueError,KeyError):rejected.append(label)
        else:raise AssertionError('Accepted '+label)
    return rejected
