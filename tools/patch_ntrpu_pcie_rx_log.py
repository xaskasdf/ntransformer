"""Separate copied AMD root CQ/RC log assembly without changing AXIS traffic.

Apply after the existing logger and TX-concurrency patches. Common tasks have
already been verified free of timing controls by patch_concurrent; each call
may therefore use the original reader/parser scratch image atomically.
"""
import re
from patch_ntrpu_pcie_multi import replace_once


RX_TASKS = r'''
reg [7:0] ntrpu_log_cq [0:5119], ntrpu_log_rc [0:5119];
integer ntrpu_log_cq_idx=0,ntrpu_log_rc_idx=0;
task automatic TSK_NTRPU_RX_DATA;
    input is_rc,last,txrx;
    input [63:0] data;
    input remainder;
    integer idx,count,i;
    begin
        if ((^{is_rc,last,txrx,remainder}) === 1'bx || txrx !== 1'b0)
            $fatal(1,"NTRPU_RX_LOG_ARGUMENT");
        idx=is_rc ? ntrpu_log_rc_idx:ntrpu_log_cq_idx;
        count=last && remainder ? 4:8;
        if (idx<0 || idx+count>5120) $fatal(1,"NTRPU_RX_LOG_OVERFLOW");
        // The reader appends only; no copy of earlier bytes is necessary.
        frame_store_rx_idx=idx;
        TSK_READ_DATA(last,txrx,data,remainder);
        if(frame_store_rx_idx!=idx+count) $fatal(1,"NTRPU_RX_LOG_READER");
        for(i=idx;i<frame_store_rx_idx;i=i+1)
            if(is_rc) ntrpu_log_rc[i]=frame_store_rx[i];
            else ntrpu_log_cq[i]=frame_store_rx[i];
        if(is_rc) ntrpu_log_rc_idx=frame_store_rx_idx;
        else ntrpu_log_cq_idx=frame_store_rx_idx;
    end
endtask
task automatic TSK_NTRPU_RX_PARSE;
    input is_rc,txrx;
    integer idx,i;
    begin
        if ((^{is_rc,txrx}) === 1'bx || txrx !== 1'b0)
            $fatal(1,"NTRPU_RX_LOG_ARGUMENT");
        idx=is_rc ? ntrpu_log_rc_idx:ntrpu_log_cq_idx;
        if(idx<12 || idx>5120) $fatal(1,"NTRPU_RX_LOG_SIZE");
        frame_store_rx_idx=idx;
        for(i=0;i<idx;i=i+1)
            frame_store_rx[i]=is_rc ? ntrpu_log_rc[i]:ntrpu_log_cq[i];
        $display("NTRPU_PCIE_RX_LOG_PACKET channel=%0d bytes=%0d head=%08h time_ps=%0.0f",
            is_rc,idx,{frame_store_rx[0],frame_store_rx[1],frame_store_rx[2],frame_store_rx[3]},$realtime/1ps);
        TSK_PARSE_FRAME(txrx);
        if(frame_store_rx_idx!=0) $fatal(1,"NTRPU_RX_LOG_PARSER_DRAIN");
        if(is_rc) ntrpu_log_rc_idx=0;else ntrpu_log_cq_idx=0;
    end
endtask
'''


def patch_rx(rx, com):
    for body in re.findall(r'task (?:automatic )?\w+;.*?endtask',com,re.S):
        code=re.sub(r'/\*.*?\*/|//[^\n]*','',body,flags=re.S)
        if re.search(r'@|#|\bwait\s*\(',code):
            raise ValueError('Common parser gained a timing control')
    # Boundaries from the generated source, including all data-width branches.
    cq_start=rx.index('  case (cq_rx_state)')
    rc_start=rx.index('  case (rc_rx_state)')
    builder_start=rx.index('task TSK_BUILD_RC_TO_PCIE_PKT;')
    assert cq_start<rc_start<builder_start
    records=[]
    def calls(body,is_rc,label):
        before=body
        for name in ['DATA','PARSE']:
            old='board.RP.com_usrapp.TSK_'+('READ_DATA' if name=='DATA' else 'PARSE_FRAME')+'('
            new='board.RP.com_usrapp.TSK_NTRPU_RX_'+name+"(1'b"+str(is_rc)+', '
            count=body.count(old)
            body=body.replace(old,new)
            records.append(dict(section=label,operation=name,calls=count))
        restored=re.sub(r"TSK_NTRPU_RX_DATA\(1'b[01], ",'TSK_READ_DATA(',body)
        restored=re.sub(r"TSK_NTRPU_RX_PARSE\(1'b[01], ",'TSK_PARSE_FRAME(',restored)
        assert restored==before
        return body
    cq=calls(rx[cq_start:rc_start],0,'CQ state machine')
    rc=calls(rx[rc_start:builder_start],1,'RC state machine')
    suffix=rx[builder_start:]
    def builder(match):
        return calls(match.group(),int(match.group(1)=='RC'),match.group(1)+' builder')
    suffix=re.sub(r'task TSK_BUILD_(RC|CQ)_TO_PCIE_PKT;.*?endtask',builder,suffix,flags=re.S)
    fixed=rx[:cq_start]+cq+rc+suffix
    if 'com_usrapp.TSK_READ_DATA(' in fixed or 'com_usrapp.TSK_PARSE_FRAME(' in fixed:
        raise ValueError('Unclassified RX log call')
    com=replace_once(com,'task TSK_READ_DATA;',RX_TASKS+'\ntask TSK_READ_DATA;')
    return fixed,com,records
