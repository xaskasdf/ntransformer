"""Repair public AMD example logging for the bounded host64 DMA fixture.

RQ control accesses are single-beat, <=32-byte TLPs packed in bits 255:0.
CC completions use all 512 bits. This is not a general RQ traffic generator.
The DUT and AXIS traffic are unchanged; invalid frames still stop simulation.
"""
import re

from patch_ntrpu_pcie_multi import replace_once


LOG_TX_TASK = r"""
task TSK_NTRPU_LOG_TX;
    input first;
    input last;
    input is_cc;
    input [511:0] data;
    input [3:0] remainder;
    integer count;
    reg [3:0] rq_remainder;
    begin
        if (is_cc) begin
            TSK_READ_DATA_512(first, last, 1'b1, data, remainder);
        end else begin
            // Generated control tasks zero-extend a 256-bit log image.
            // Infer its actual byte count from the TLP, not their stale rem.
            if (first !== 1'b1 || last !== 1'b1)
                $fatal(1, "NTRPU_LOG_RQ_MULTIBEAT_UNSUPPORTED");
            if (data[511:256] !== 256'b0 || (^data[255:224]) === 1'bx)
                $fatal(1, "NTRPU_LOG_RQ_LAYOUT");
            count = data[253] ? 16 : 12;
            if (data[254]) count = count + 4 * ((data[233:224] == 0) ? 1024 : data[233:224]);
            if (count > 32) $fatal(1, "NTRPU_LOG_RQ_SIZE_UNSUPPORTED");
            rq_remainder = 16 - count / 4;
            TSK_READ_DATA_512(first, last, 1'b1, data << 256, rq_remainder);
        end
    end
endtask
"""


def patch_text(tx, rx, com):
    # Completion AXIS and logging branches call this concurrently. Verilog
    # static task arguments otherwise overwrite one another while waiting.
    tx = replace_once(tx, 'task TSK_TX_SYNCHRONIZE;', 'task automatic TSK_TX_SYNCHRONIZE;')
    tx = replace_once(tx,
        'board.RP.com_usrapp.TSK_READ_DATA_512(first_, last_call_,`TX_LOG,pcie_tlp_data,pcie_tlp_rem);',
        'board.RP.com_usrapp.TSK_NTRPU_LOG_TX(first_, last_call_, tready_sw_ == `SYNC_CC_RDY, pcie_tlp_data, pcie_tlp_rem);')
    rx = replace_once(rx, "((cq_data[63:32] == 32'h0) ? 3'b010 : 3'b001)",
                      "((cq_data[63:32] == 32'h0) ? 3'b010 : 3'b011)")
    start = com.index('task TSK_DECIPHER_FRAME;')
    end = com.index('endtask', start)
    decoder = com[start:end]
    decoder, count = re.subn(r"frame_store_tx\['h2([0-3])\]", r"frame_store_tx['h\1]", decoder)
    if count != 8:
        raise ValueError('Unexpected transmitted header offset uses')
    com = com[:start] + decoder + com[end:]
    start = com.index('task TSK_READ_DATA_512;')
    end = com.index('endtask', start)
    reader = com[start:end]
    reader, count = re.subn(r'casex \(trn_rem\).*?endcase',
        'if ((^trn_rem) === 1\'bx) $fatal(1, "NTRPU_LOG_UNKNOWN_REMAINDER");\n'
        '       _rem = 64 - 4 * trn_rem;', reader, flags=re.S)
    if count != 1:
        raise ValueError('Unexpected log remainder decoder')
    com = com[:start] + LOG_TX_TASK + '\n' + reader + com[end:]
    return tx, rx, com
