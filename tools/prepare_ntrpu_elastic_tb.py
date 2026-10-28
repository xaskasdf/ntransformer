"""Derive an independent AW/W scoreboard while preserving the original bench."""

import argparse
from pathlib import Path


def prepare(source, netlist=False, reserve=4):
    text = source.replace('ntrpu_peer_multibuffer #(', 'ntrpu_peer_multibuffer_elastic #(')
    text = text.replace('dut.', 'dut.core.')
    text = text.replace('    integer aw_tag', '''    integer issued[0:CHANNELS-1], aw_cursor[0:CHANNELS-1];
    integer w_cursor[0:CHANNELS-1], cpl_cursor[0:CHANNELS-1];
    integer expect_tag[0:CHANNELS*512-1], expect_beats[0:CHANNELS*512-1];
    integer expect_status[0:CHANNELS*512-1];
    integer w_before_aw=0, aw_ahead=0, jobs=0, bytes=0, simultaneous_w=0;
    integer completed=0, foreign_b=0, error_b=0, w_this_cycle;
    integer aw_tag''')
    text = text.replace('aw_tag[p]=-1; w_count[p]=0;', '''aw_tag[p]=-1; w_count[p]=0;
                issued[p]=0; aw_cursor[p]=0; w_cursor[p]=0; cpl_cursor[p]=0;''')
    start = text.index('            if(cmd_valid && cmd_ready)')
    end = text.index('            if(data_valid && data_ready) $display', start)
    text = text[:start] + '''            w_this_cycle=0;
            if(cmd_valid && cmd_ready) begin
                q=cmd_channel*512+issued[cmd_channel];
                if(issued[cmd_channel]>=512) $fatal(1,"scoreboard capacity");
                expect_tag[q]=cmd_tag; expect_beats[q]=cmd_len+1; expect_status[q]=-1;
                issued[cmd_channel]=issued[cmd_channel]+1; jobs=jobs+1;
                $display("EXT CMD %0d %0d %0d %h %0d",cycle,cmd_channel,cmd_tag,cmd_addr,cmd_len+1);
            end
''' + text[end:]
    # Public checks stay separate from internal lifetime/credit assertions.
    internal_start = text.index('            if(data_valid && data_ready && dut.core.')
    internal_end = text.index('            for(p=0;', internal_start)
    text = text[:internal_start]+'`ifndef NTRPU_ELASTIC_NETLIST\n'+text[internal_start:internal_end]+'`endif\n'+text[internal_end:]
    internal_start = text.index('                if(dut.core.fifo_count')
    internal_end = text.index('                if(m_awvalid', internal_start)
    text = text[:internal_start]+'`ifndef NTRPU_ELASTIC_NETLIST\n'+text[internal_start:internal_end]+'`endif\n'+text[internal_end:]
    start = text.index('                if(m_awvalid[p] && m_awready[p]) begin')
    end = text.index('\n            end\n        end\n    end', start)
    text = text[:start] + '''                if(m_awvalid[p] && m_awready[p]) begin
                    q=p*512+aw_cursor[p];
                    if(aw_cursor[p]>=issued[p] ||
                       m_awaddr[p*64 +:64]!==BASE+(64'(expect_tag[q])<<12) ||
                       int'(m_awlen[p*8 +:8])+1!=expect_beats[q]) $fatal(1,"AW order/length mismatch");
                    if(m_awsize[p*3 +:3]!=6 || m_awburst[p*2 +:2]!=1 || m_awid[p*4 +:4]!=0 || m_awcache[p*4 +:4]!=0)
                        $fatal(1,"AXI attributes");
                    if(aw_cursor[p]>w_cursor[p]) aw_ahead=aw_ahead+1;
                    aw_cursor[p]=aw_cursor[p]+1;
                    $display("EXT AW %0d %0d %h %0d",cycle,p,m_awaddr[p*64 +:64],m_awlen[p*8 +:8]+1);
                end
                if(m_wvalid[p] && m_wready[p]) begin
                    q=p*512+w_cursor[p];
                    if(w_cursor[p]>=issued[p] || m_wstrb[p*64 +:64]!==64'hffffffffffffffff)
                        $fatal(1,"unexpected W or partial strobe");
                    if(m_wdata[p*512 +:512]!==pattern(expect_tag[q],w_count[p])) $fatal(1,"payload mismatch");
                    if(m_wlast[p] !== (w_count[p]==expect_beats[q]-1)) $fatal(1,"WLAST mismatch");
                    if(w_cursor[p]>=aw_cursor[p]) w_before_aw=w_before_aw+1;
                    $display("EXT W %0d %0d %0d %h",cycle,p,m_wlast[p],m_wdata[p*512 +:512]);
                    w_count[p]=w_count[p]+1; w_total[p]=w_total[p]+1; bytes=bytes+64;
                    w_this_cycle=w_this_cycle+1;
                    if(m_wlast[p]) begin w_cursor[p]=w_cursor[p]+1; w_count[p]=0; end
                end
                // B is eligible only after BOTH accepted AW and accepted WLAST.
                while(response_tail[p]<aw_cursor[p] && response_tail[p]<w_cursor[p]) begin
                    q=p*512+response_tail[p];
                    response_tag[q]=expect_tag[q]; response_due[q]=cycle+160+(expect_tag[q]%23);
                    response_tail[p]=response_tail[p]+1;
                end
                if(m_bvalid[p] && m_bready[p]) begin
                    $display("EXT B %0d %0d %0d %0d",cycle,p,m_bid[p*4 +:4],m_bresp[p*2 +:2]);
                    if(m_bid[p*4 +:4]==0) begin
                        expect_status[p*512+response_head[p]]=m_bresp[p*2 +:2];
                        response_head[p]=response_head[p]+1;
                        if(m_bresp[p*2 +:2]!=0) error_b=error_b+1;
                    end else begin foreign_pending=0; foreign_b=foreign_b+1; end
                end
                if(cpl_valid[p] && cpl_ready[p]) begin
                    q=p*512+cpl_cursor[p];
                    if(cpl_cursor[p]>=response_head[p] || cpl_tag[p*64 +:64]!==64'(expect_tag[q]) ||
                       int'(cpl_status[p*2 +:2])!=expect_status[q]) $fatal(1,"completion order/status mismatch");
                    cpl_cursor[p]=cpl_cursor[p]+1; completed=completed+1;
                    $display("TRACE CPL %0d %0d %0d %0d",cycle,p,cpl_tag[p*64 +:64],cpl_status[p*2 +:2]);
                end
''' + text[end:]
    text = text.replace('            end\n        end\n    end\n\n    task send_job;',
                        '            end\n            if(w_this_cycle>1) simultaneous_w=simultaneous_w+1;\n        end\n    end\n\n    task send_job;')
    text = text.replace('if(response_head[k]!=response_tail[k] || aw_tag[k]!=-1)',
                        'if(response_head[k]!=response_tail[k] || cpl_cursor[k]!=issued[k] || aw_cursor[k]!=issued[k] || w_cursor[k]!=issued[k])')
    text = text.replace('        repeat(4) @(negedge clk); rst_n=1;', '''`ifdef NTRPU_ELASTIC_NETLIST
        #200;
`endif
        repeat(4) @(negedge clk); rst_n=1;
        $display("ELASTIC_CASE data_before_address");
        hold_aw0=1;
        send_job(0,1); send_job(0,1);
        repeat(40) @(negedge clk);
        if(w_before_aw==0) $fatal(1,"no data before address coverage");
        stop=1; probe_closed_admission;
        if(drained) $fatal(1,"drained with buffered AW");
        hold_aw0=0; drain_all; stop=0;
        $display("ELASTIC_CASE bank_return_with_queued_data");
        hold_w0=1;
        send_job(0,1); send_job(0,1); send_job(0,1);
        repeat(40) @(negedge clk);
        if(drained) $fatal(1,"drained with buffered W");
        hold_w0=0; drain_all;''')
    text = text.replace('        $display("PASS seed=', '''        if(jobs!=completed || foreign_b==0 || error_b==0 || w_before_aw==0) $fatal(1,"missing external coverage");
        if(MAX_OUTSTANDING>1 && aw_ahead==0) $fatal(1,"no AW ahead of W coverage");
        if(CHANNELS>1 && simultaneous_w==0) $fatal(1,"no parallel W coverage");
        $display("ELASTIC_SCOREBOARD jobs=%0d bytes=%0d completed=%0d w_before_aw=%0d aw_ahead=%0d parallel=%0d foreign_b=%0d error_b=%0d",jobs,bytes,completed,w_before_aw,aw_ahead,simultaneous_w,foreign_b,error_b);
        $display("PASS seed=''')
    monitor = '''
`ifndef NTRPU_ELASTIC_NETLIST
    integer x, lifetime_reuse=0;
    integer last_bank_tag[0:BANKS-1];
    reg external_done[0:4095];
    always @(posedge clk) begin
        if(!rst_n) begin
            for(x=0;x<BANKS;x=x+1) last_bank_tag[x]=0;
            for(x=0;x<4096;x=x+1) external_done[x]=0;
        end else begin
            if(cmd_valid && cmd_ready) begin
                $display("TRACE CMD %0d %0d %0d %h %0d %0d %0d",cycle,cmd_channel,cmd_tag,cmd_addr,cmd_len+1,dut.core.alloc_bank,dut.core.alloc_generation);
                if(last_bank_tag[dut.core.alloc_bank]!=0 && !external_done[last_bank_tag[dut.core.alloc_bank]])
                    lifetime_reuse=lifetime_reuse+1;
                last_bank_tag[dut.core.alloc_bank]=cmd_tag;
            end
            for(x=0;x<CHANNELS;x=x+1) begin
                if(dut.i_awvalid[x] && dut.i_awready[x])
                    $display("TRACE AW %0d %0d %h %0d",cycle,x,dut.i_awaddr[x*64 +:64],dut.i_awlen[x*8 +:8]+1);
                if(dut.i_wvalid[x] && dut.i_wready[x])
                    $display("TRACE W %0d %0d %0d %h",cycle,x,dut.i_wlast[x],dut.i_wdata[x*512 +:512]);
                if(dut.i_bvalid[x] && dut.i_bready[x])
                    $display("TRACE B %0d %0d %0d %0d",cycle,x,dut.i_bid[x*4 +:4],dut.i_bresp[x*2 +:2]);

            end
        end
    end
    final $display("ELASTIC_LIFETIME queued_bank_reuses=%0d",lifetime_reuse);
`endif
'''
    # Track completed external tags directly in the public W scoreboard.
    text=text.replace('if(m_wlast[p]) begin w_cursor[p]', '''if(m_wlast[p]) begin
`ifndef NTRPU_ELASTIC_NETLIST
                        external_done[expect_tag[q]]=1;
`endif
                        w_cursor[p]''')
    # Capture cycle consistently from simulation time, independent of always scheduling.
    monitor=monitor.replace(',cycle,',',($time/10)+1,')
    text=text.replace('endmodule',monitor+'\nendmodule')
    if netlist:
        old='''    ntrpu_peer_multibuffer_elastic #(.CHANNELS(CHANNELS),.RESERVE_BANKS(RESERVE_BANKS),
        .BANK_BEATS(BANK_BEATS),.DESCRIPTORS(DESCRIPTORS),.MAX_OUTSTANDING(MAX_OUTSTANDING)) dut(.*);'''
        assert old in text
        text=text.replace(old,'    ntrpu_peer_multibuffer_compact dut(.*);')
        text=text.replace('parameter integer CHANNELS=2, RESERVE_BANKS=4,',f'parameter integer CHANNELS=2, RESERVE_BANKS={reserve},')
        text='`define NTRPU_ELASTIC_NETLIST\n'+text
    return text


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--netlist',action='store_true')
    parser.add_argument('--reserve',type=int,default=4)
    args=parser.parse_args()
    args.output.write_text(prepare(args.source.read_text(),args.netlist,args.reserve))
