"""Directed binary lifetime/reset/sequence cases for the production RC stage."""
import hashlib
import json
from pathlib import Path


def header(tag, count, byte_count, address=0x100, terminal=1, error=0):
    return address | error<<12 | byte_count<<16 | terminal<<30 | count<<32 | 0x01a0<<48 | tag<<64


def packet(tag,count,address=0x100,start_lane=0):
    words=[0]*start_lane+[header(tag,count,4*count,address)>>(32*i)&0xffffffff for i in range(3)]+[0]*count
    beats=[]
    for start in range(0,len(words),16):
        chunk=words[start:start+16];data=sum(w<<(32*i) for i,w in enumerate(chunk));user=0;owned=meta=valid_meta=0
        if start==0:
            q=start_lane//4;user|=1<<64 | q<<68;owned=1<<q;valid_meta=owned
            meta=(4*count | address<<13)<<(30*q)
        if start+16>=len(words):user|=1<<76 | (len(chunk)-1)<<80
        for i in range(len(chunk)):
            if start+i>=start_lane+3:user|=15<<(i*4)
        beats.append(dict(data=data,user=user,owned=owned,metadata=meta,expected_valid=valid_meta))
    return beats


def four_partials(tag,remaining,address):
    data=user=metadata=0
    for q in range(4):
        bc=remaining-4*q;addr=(address+4*q)%4096
        data|=header(tag,1,bc,addr,int(bc==4))<<(128*q)
        metadata|=(bc|addr<<13)<<(30*q)
        user|=q<<(68+2*q) | (4*q+3)<<(80+4*q) | 15<<(16*q+12)
    user|=15<<64 | 15<<76
    return dict(data=data,user=user,owned=15,metadata=metadata,expected_valid=15)


def generate():
    rows=[];phases=[]
    def emit(beat=None,*,rst=1,observing=1,valid=None,ready=1,requester=0x01a0,sequence=None,**changes):
        b=dict(data=0,user=0,owned=0,metadata=0,expected_valid=0)
        if beat:b.update(beat)
        b.update(changes)
        if valid is None:valid=int(beat is not None)
        if sequence is None:sequence=len(rows)
        rows.append(f'{phases[-1]["id"]} {rst} {observing} {valid} {ready} {b["owned"]:x} {requester:04x} {sequence:08x} {b["expected_valid"]:x} {b["metadata"]:030x} {b["data"]:0128x} {b["user"]:041x}')
        phases[-1]['rows']+=1
    def begin(name):
        phases.append(dict(id=len(phases),name=name,start=len(rows),rows=0));emit(rst=0);emit()
    def flush():
        emit();emit();emit()
    begin('cold_reset_cancels_captured_terminal')
    emit(packet(1,1)[0]);emit(rst=0);flush()
    begin('cold_reset_cancels_retained_payload')
    long=packet(2,20);emit(long[0]);emit();emit(rst=0);emit(long[-1]);flush()
    begin('sequence_wrap_and_changed_ownership_during_payload')
    wrapped=packet(3,20,start_lane=12)
    emit(wrapped[0],sequence=0xfffffffe)
    emit(wrapped[1],sequence=0xffffffff,owned=15)
    emit(wrapped[2],sequence=0,owned=0);flush()
    begin('observing_pause_retains_context')
    long=packet(4,20);emit(long[0])
    for _ in range(5):emit(observing=0)
    emit(long[1]);flush()
    begin('stalls_before_and_inside_packet')
    long=packet(5,32)
    for _ in range(3):emit(long[0],ready=0)
    emit(long[0])
    for _ in range(4):emit(long[1],ready=0)
    emit(long[1]);emit(long[2]);flush()
    begin('sustained_four_same_tag_splits')
    for beat in range(256):emit(four_partials(55,4096-16*beat,(16*beat)%4096))
    flush()
    begin('missing_expected_metadata')
    emit(packet(6,1)[0],expected_valid=0);flush()
    begin('wrong_independent_identity_tc_attributes')
    emit(packet(7,1)[0],requester=0x01a1,metadata=4|0x100<<13|2<<25|1<<28);flush()
    begin('cold_reset_cancels_captured_validation_fault')
    emit(packet(8,1)[0],expected_valid=0);emit(rst=0);flush()
    begin('payload_enable_fault_retained_to_eop')
    long=packet(9,20);emit(long[0],user=long[0]['user']^(8<<12));emit(long[1]);flush()
    begin('non_normal_terminal_remains_unresolved')
    emit(dict(data=header(10,0,0,error=4),user=1<<64|1<<76|2<<80,owned=1,metadata=4|0x100<<13,expected_valid=1));flush()
    begin('overlap_preserves_lost_obligation')
    emit(packet(11,20)[0]);emit(packet(12,1)[0]);flush()
    begin('discontinue_keeps_failure_explicit')
    emit(packet(13,20)[0]);emit(dict(data=0,user=1<<96|1<<76,owned=0));flush()
    begin('orphan_eop_after_completed_packet')
    emit(packet(14,1)[0]);emit(dict(data=0,user=1<<76,owned=0));flush()
    begin('final_cold_reset')
    flush()
    text='\n'.join(rows)+'\n'
    return text,dict(rows=len(rows),phases=phases,vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
        binary_inputs_only=True,lanes=4,latency_cycles=2,
        scope='Production RC stage boundaries with explicit metadata/ownership inputs; not the integrated inventory bank or native PCIe endpoint.')


if __name__=='__main__':
    text,manifest=generate();base=Path(__file__).resolve().parents[1]/'fpga/tb/data/pcie_rc_event_boundaries'
    with base.with_suffix('.txt').open('x',newline='\n') as f:f.write(text)
    with base.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=manifest['rows'],phases=len(manifest['phases']),sha256=manifest['vectors_sha256'])))
