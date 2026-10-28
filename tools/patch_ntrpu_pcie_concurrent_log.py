"""Separate RQ/CC logging state in an already logger-patched AMD example.

No AXIS assignment, protected IP or header decoder is changed. Each channel
keeps its own packet image and partial frame. Completed frames use the original
parser atomically; its tasks must contain no timing controls. The completion
callback is additionally limited to RX so logging outbound DMA completions
cannot wake a pending MMIO read with an unrelated zero value.
"""
import re
from patch_ntrpu_pcie_multi import replace_once


CHANNEL_STATE = '''
reg [7:0] ntrpu_log_rq [0:5119];
reg [7:0] ntrpu_log_cc [0:5119];
integer ntrpu_log_rq_idx = 0;
integer ntrpu_log_cc_idx = 0;
'''

CHANNEL_TASK = r'''
task automatic TSK_NTRPU_LOG_CHANNEL;
    input first, last, is_cc;
    input [511:0] data;
    input [3:0] remainder;
    integer idx, i, bytes_in;
    begin
        if ((^{first,last,is_cc}) === 1'bx)
            $fatal(1, "NTRPU_LOG_CHANNEL_UNKNOWN");
        idx = is_cc ? ntrpu_log_cc_idx : ntrpu_log_rq_idx;
        if ((first && idx != 0) || (!first && idx == 0))
            $fatal(1, "NTRPU_LOG_CHANNEL_SEQUENCE");
        if ((^remainder) === 1'bx) $fatal(1, "NTRPU_LOG_UNKNOWN_REMAINDER");
        bytes_in = is_cc ? 64-4*remainder : 32;
        if (idx < 0 || idx+bytes_in > 5120) $fatal(1, "NTRPU_LOG_CHANNEL_OVERFLOW");
        // The original reader/parser has no timing controls. Its scratch
        // frame is used only during this atomic call, never across a wait.
        frame_store_tx_idx = idx;
        for (i=0; i<idx; i=i+1)
            frame_store_tx[i] = is_cc ? ntrpu_log_cc[i] : ntrpu_log_rq[i];
        TSK_NTRPU_LOG_TX(first, last, is_cc, data, remainder);
        idx = frame_store_tx_idx;
        if (last) begin
            $display("NTRPU_PCIE_LOG_PACKET channel=%0d bytes=%0d head=%08h time_ps=%0.0f",
                is_cc,idx,{frame_store_tx[0],frame_store_tx[1],frame_store_tx[2],frame_store_tx[3]},$realtime/1ps);
            TSK_PARSE_FRAME(`TX_LOG);
            if (frame_store_tx_idx != 0) $fatal(1, "NTRPU_LOG_PARSER_DRAIN");
            idx = 0;
        end else begin
            for (i=0; i<idx; i=i+1)
                if (is_cc) ntrpu_log_cc[i] = frame_store_tx[i];
                else ntrpu_log_rq[i] = frame_store_tx[i];
        end
        if (is_cc) ntrpu_log_cc_idx = idx;
        else ntrpu_log_rq_idx = idx;
    end
endtask
'''


def patch_concurrent(tx, com):
    # Refuse timing-dependent parser implementations rather than assuming an
    # atomic scratch-buffer call. These source files retain AMD headers.
    tasks = re.findall(r'task (?:automatic )?\w+;.*?endtask', com, re.S)
    for task in tasks:
        code = re.sub(r'/\*.*?\*/|//[^\n]*', '', task, flags=re.S)
        if re.search(r'@|#|\bwait\s*\(', code):
            raise ValueError('Common parser gained a timing control')
    changes = []
    def split(match):
        body, name = match.group(), match.group(1)
        if name == 'TSK_TX_SYNCHRONIZE' or not re.search(r'\bpcie_tlp_(data|rem)\b', body):
            return body
        selectors = set(re.findall(r'`SYNC_(RQ|CC)_RDY', body))
        if len(selectors) != 1:
            raise ValueError('Unexpected logger channel in '+name)
        channel = selectors.pop().lower()
        result = re.sub(r'\bpcie_tlp_(data|rem)\b', r'ntrpu_'+channel+r'_tlp_\1', body)
        assert re.sub(r'\bntrpu_'+channel+r'_tlp_(data|rem)\b', r'pcie_tlp_\1', result) == body
        changes.append(dict(task=name, channel=channel))
        return result
    tx = re.sub(r'task (?:automatic )?(\w+);.*?endtask', split, tx, flags=re.S)
    if len(changes) != 14:
        raise ValueError('Unexpected number of packet producers')
    tx = replace_once(tx,
        'board.RP.com_usrapp.TSK_NTRPU_LOG_TX(first_, last_call_, tready_sw_ == `SYNC_CC_RDY, pcie_tlp_data, pcie_tlp_rem);',
        'board.RP.com_usrapp.TSK_NTRPU_LOG_CHANNEL(first_, last_call_, tready_sw_ == `SYNC_CC_RDY,\n'
        '                    tready_sw_ == `SYNC_CC_RDY ? ntrpu_cc_tlp_data : ntrpu_rq_tlp_data,\n'
        '                    tready_sw_ == `SYNC_CC_RDY ? ntrpu_cc_tlp_rem : ntrpu_rq_tlp_rem);')
    tx = replace_once(tx, 'if (last_call_)\n                 board.RP.com_usrapp.TSK_PARSE_FRAME(`TX_LOG);',
                      '// Per-channel logger parses complete packets atomically.')
    # Remaining original declarations/initialization become the RQ pair.
    tx = re.sub(r'\bpcie_tlp_(data|rem)\b', r'ntrpu_rq_tlp_\1', tx)
    declarations = 'reg [511:0] ntrpu_cc_tlp_data = 0;\nreg [3:0] ntrpu_cc_tlp_rem = 0;\n'
    pos = tx.index('reg        [(C_DATA_WIDTH - 1):0]')
    tx = tx[:pos]+declarations+tx[pos:]
    com = replace_once(com, 'task TSK_NTRPU_LOG_TX;', CHANNEL_STATE+CHANNEL_TASK+'\ntask TSK_NTRPU_LOG_TX;')
    com = replace_once(com,
        '`TX_TASKS.TSK_SET_READ_DATA(4\'hf,dword_data); // send the data to the tx_app',
        'if (!txrx) `TX_TASKS.TSK_SET_READ_DATA(4\'hf,dword_data); // RX completion only')
    return tx, com, changes
