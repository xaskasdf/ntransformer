`timescale 1ns/1ps
// Passive observer for the three-job Brandon async fixture. No drives, delays,
// forces or random calls: the functional trace must match the unprofiled run.
`define FP_CORE tb_ntrpu_ffn_async_transfer.dut.device.fabric.device.core
`define FP_MAC `FP_CORE.matrix.engine.engine.core
`define FP_ACT `FP_CORE.activation
module ntrpu_ffn_profile;
    wire clk=tb_ntrpu_ffn_async_transfer.clk;
    wire rst_n=tb_ntrpu_ffn_async_transfer.rst_n;
    wire accept=(`FP_CORE.m_job_valid && `FP_CORE.m_job_ready) ||
                (`FP_CORE.a_job_valid && `FP_CORE.a_job_ready);
    wire finish=(`FP_CORE.m_done_valid && `FP_CORE.m_done_ready) ||
                (`FP_CORE.a_done_valid && `FP_CORE.a_done_ready);
    bit active=0;
    longint unsigned clock_count=0,start_clock=0;
    int ident,phase,cycles,ar,r,aw,w,b,write_bytes;
    int ar_wait,r_wait,r_backpressure,aw_wait,w_wait,b_wait;
    int read_pending,write_pending;
    int lane0,lane1,both_lanes,mac_blocks,mac_done0,mac_done1;
    int run_cycles,block_fire,block_backpressure,block_empty,block_neither;
    int act_read,act_hls,act_write,act_other,act_starts,act_dones;
    always @(posedge clk) begin
        clock_count++;
        if(!rst_n) begin
            if(active) $fatal(1,"FAIL FFN PROFILE reset during measured stage");
        end else begin
            if(accept) begin
                if(active) $fatal(1,"FAIL FFN PROFILE overlapping stages");
                active=1;start_clock=clock_count;
                ident=tb_ntrpu_ffn_async_transfer.ident;
                phase=int'(`FP_CORE.controller.phase);
                cycles=0;ar=0;r=0;aw=0;w=0;b=0;write_bytes=0;
                ar_wait=0;r_wait=0;r_backpressure=0;aw_wait=0;w_wait=0;b_wait=0;
                read_pending=0;write_pending=0;
                lane0=0;lane1=0;both_lanes=0;mac_blocks=0;mac_done0=0;mac_done1=0;
                run_cycles=0;block_fire=0;block_backpressure=0;block_empty=0;block_neither=0;
                act_read=0;act_hls=0;act_write=0;act_other=0;act_starts=0;act_dones=0;
            end
            if(active) begin
                if(phase!=int'(`FP_CORE.controller.phase)) $fatal(1,"FAIL FFN PROFILE phase changed while active");
                cycles++;
                ar_wait+=int'(`FP_CORE.m_axi_arvalid && !`FP_CORE.m_axi_arready);
                r_wait+=int'(read_pending>0 && !`FP_CORE.m_axi_rvalid);
                r_backpressure+=int'(`FP_CORE.m_axi_rvalid && !`FP_CORE.m_axi_rready);
                aw_wait+=int'(`FP_CORE.m_axi_awvalid && !`FP_CORE.m_axi_awready);
                w_wait+=int'(`FP_CORE.m_axi_wvalid && !`FP_CORE.m_axi_wready);
                b_wait+=int'(write_pending>0 && !`FP_CORE.m_axi_bvalid);
                if(`FP_CORE.m_axi_arvalid && `FP_CORE.m_axi_arready) begin ar++;read_pending++;end
                if(`FP_CORE.m_axi_rvalid && `FP_CORE.m_axi_rready) begin
                    r++;if(`FP_CORE.m_axi_rlast) read_pending--;
                end
                if(`FP_CORE.m_axi_awvalid && `FP_CORE.m_axi_awready) begin aw++;write_pending++;end
                if(`FP_CORE.m_axi_wvalid && `FP_CORE.m_axi_wready) begin
                    w++;write_bytes+=$countones(`FP_CORE.m_axi_wstrb);
                end
                if(`FP_CORE.m_axi_bvalid && `FP_CORE.m_axi_bready) begin b++;write_pending--;end
                if(read_pending<0 || write_pending<0) $fatal(1,"FAIL FFN PROFILE response without request");
                if(phase!=2) begin
                    lane0+=int'(`FP_MAC.lane_state[0]!=`FP_MAC.MAC_IDLE);
                    lane1+=int'(`FP_MAC.lane_state[1]!=`FP_MAC.MAC_IDLE);
                    both_lanes+=int'(`FP_MAC.lane_state[0]!=`FP_MAC.MAC_IDLE && `FP_MAC.lane_state[1]!=`FP_MAC.MAC_IDLE);
                    mac_blocks+=int'(`FP_MAC.block_fire);
                    mac_done0+=int'(`FP_MAC.lane_state[0]==`FP_MAC.MAC_WAIT && `FP_MAC.lane_done[0]);
                    mac_done1+=int'(`FP_MAC.lane_state[1]==`FP_MAC.MAC_WAIT && `FP_MAC.lane_done[1]);
                    if(`FP_MAC.state==`FP_MAC.RUN) begin
                        run_cycles++;
                        case({`FP_MAC.block_valid,`FP_MAC.block_ready})
                            2'b11:block_fire++;
                            2'b10:block_backpressure++;
                            2'b01:block_empty++;
                            2'b00:block_neither++;
                            default:$fatal(1,"FAIL FFN PROFILE unknown block handshake");
                        endcase
                    end
                end else begin
                    case(`FP_ACT.state)
                        `FP_ACT.G_ISSUE,`FP_ACT.G_READ,`FP_ACT.U_ISSUE,`FP_ACT.U_READ:act_read++;
                        `FP_ACT.H_START,`FP_ACT.H_WAIT:act_hls++;
                        `FP_ACT.W_ISSUE,`FP_ACT.W_WAIT:act_write++;
                        default:act_other++;
                    endcase
                    act_starts+=int'(`FP_ACT.state==`FP_ACT.H_START);
                    act_dones+=int'(`FP_ACT.state==`FP_ACT.H_WAIT && `FP_ACT.hls_done);
                end
                if(finish) begin
                    if(read_pending!=0 || write_pending!=0 || aw!=b) $fatal(1,"FAIL FFN PROFILE unfinished memory transfer");
                    $display("NTRPU_FFN_PROFILE id=%0d phase=%0d start=%0d end=%0d cycles=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d bytes=%0d ar_wait=%0d r_wait=%0d r_backpressure=%0d aw_wait=%0d w_wait=%0d b_wait=%0d lane0=%0d lane1=%0d both=%0d blocks=%0d done0=%0d done1=%0d run=%0d block_fire=%0d block_backpressure=%0d block_empty=%0d block_neither=%0d act_read=%0d act_hls=%0d act_write=%0d act_other=%0d act_starts=%0d act_dones=%0d",
                        ident,phase,start_clock,clock_count,cycles,ar,r,aw,w,b,write_bytes,
                        ar_wait,r_wait,r_backpressure,aw_wait,w_wait,b_wait,
                        lane0,lane1,both_lanes,mac_blocks,mac_done0,mac_done1,
                        run_cycles,block_fire,block_backpressure,block_empty,block_neither,
                        act_read,act_hls,act_write,act_other,act_starts,act_dones);
                    active=0;
                end
            end else if(finish) $fatal(1,"FAIL FFN PROFILE finish without stage");
        end
    end
endmodule
`undef FP_CORE
`undef FP_MAC
`undef FP_ACT
