"""Independent final-output transcript review for exact post-O ABI1 -> FFN -> residual."""
import math
import re
import struct


def review(text,expected,mode):
    lines=text.splitlines();position=0;times=[]
    assert mode in ('stdlib','hls') and len(expected)==4
    for sample,raw in enumerate(expected,1):
        assert len(raw)==8192
        for index,(word,) in enumerate(struct.iter_unpack('<I',raw)):
            assert lines[position]==f'NTRPU_POST_FFN_NATIVE_VALUE sample={sample} index={index} bits={word}'
            position+=1
        pattern=rf'NTRPU_POST_FFN_NATIVE_CASE sample={sample} dim=2048 hidden=8192 mac_blocks=1572864 reference_values=30720 activation_records=256 bit_exact=1 cpu_seconds=([0-9.e+-]+)'
        match=re.fullmatch(pattern,lines[position]);assert match
        seconds=float(match[1]);assert math.isfinite(seconds) and seconds>0;times.append(seconds);position+=1
    assert lines[position:]==[f'NTRPU_POST_FFN_NATIVE_PASS samples=4 reference_values=122880 mac_blocks=6291456 math={mode}']
    return dict(math=mode,samples=4,independently_checked_layer_values=8192,layer_bytes=32768,
        kernel_test_reference_values=122880,mac_blocks=6291456,cpu_seconds=times,
        input_abi_requantized=False,rtl_qualified=False,full_inference_qualified=False)


def controls(text,expected,mode):
    first=text.splitlines()[0]
    bad_word=re.sub(r'bits=(\d+)',lambda m:'bits='+str(int(m[1])^1),first)
    variants=dict(wrong_word=text.replace(first,bad_word,1),missing_word=text.replace(first+'\n','',1),
        duplicate_word=text.replace(first+'\n',first+'\n'+first+'\n',1),wrong_sample=text.replace('sample=4 index=2047','sample=3 index=2047',1),
        wrong_coverage=text.replace('reference_values=122880','reference_values=122879'),
        wrong_backend=text.replace('math='+mode,'math=wrong'),extra_diagnostic=text+'\nFATAL: injected\n',
        missing_pass=text.replace('NTRPU_POST_FFN_NATIVE_PASS','REMOVED'))
    rejected=[]
    for name,bad in variants.items():
        assert bad!=text
        try:review(bad,expected,mode)
        except (AssertionError,IndexError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
