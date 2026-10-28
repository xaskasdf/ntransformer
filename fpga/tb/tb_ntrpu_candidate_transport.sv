`timescale 1ns/1ps
module tb_ntrpu_candidate_transport;
    reg clk=0;always #2.5 clk=~clk;
    reg rst_n=0,stop=0,fault_in=0;
    reg [1:0] submit_valid=0;
    wire [1:0] submit_ready;
    reg [463:0] submit_desc=0;
    wire fetch_valid,fetch_channel,payload_ready;
    wire [231:0] fetch_desc;
    reg fetch_ready=0,payload_valid=0;
    reg [511:0] payload_data=0;
    reg fence_valid=0,fence_done_ready=0;
    reg [63:0] fence_id=0;
    wire fence_ready,fence_done_valid;
    wire [63:0] fence_done_id;
    wire [1:0] cpl_valid,m_awvalid,m_wvalid,m_wlast,m_bready;
    reg [1:0] cpl_ready=0,m_awready=0,m_wready=0,m_bvalid=0;
    wire [127:0] cpl_tag,m_awaddr,m_wstrb;
    wire [3:0] cpl_status,m_awburst;
    wire [15:0] m_awlen;
    wire [7:0] m_awid,m_awcache;
    wire [5:0] m_awsize;
    wire [1023:0] m_wdata;
    reg [7:0] m_bid=0;
    reg [3:0] m_bresp=0;
    wire fault,drained;
    wire [7:0] banks_used,queued;
    wire [9:0] pending_counts;
    ntrpu_candidate_transport dut(.*);

    integer cycle=0,scenario=0,src_tag=0,src_index=0,src_beats=0;
    reg src_active=0,hold_fetch=0,hold_payload=0,unsolicited=0;
    reg payload_sent=0,payload_probe=0,input_held=0;
    reg [511:0] saved_input;
    reg [1:0] hold_aw=0,hold_w=0,hold_b=0,hold_cpl=0;
    integer error_tag=-1;
    integer submitted[0:1],fetched[0:1],aws[0:1],ws[0:1],bs[0:1],cpls[0:1],b_tail[0:1];
    integer order[0:2047],lengths[0:1023],input_beats[0:1023],output_beats[0:1023];
    integer b_due[0:2047],status[0:1023];
    reg [1:0] b_sent=0,aw_held=0,w_held=0,cpl_held=0;
    reg fetch_held=0,fence_held=0,barrier_pending=0;
    reg [169:0] saved_aw;
    reg [1153:0] saved_w;
    reg [131:0] saved_cpl;
    reg [232:0] saved_fetch;
    reg [63:0] saved_fence,barrier_id;
    integer p,t,k;

    function [231:0] descriptor(input integer tag,input integer beats);
        descriptor={32'h76543210,64'(tag),64'h100000000+64'(tag)*4096,
                    64'h800000000+64'(tag)*4096,8'(beats-1)};
    endfunction
    function [511:0] pattern(input integer tag,input integer beat);
        for(integer lane=0;lane<16;lane++)
            pattern[lane*32+:32]=(32'(tag)*32'h9e3779b9)^(32'(beat)*32'h10203)^(32'(lane)*32'h01010101);
    endfunction
    task need(input bit ok,input string why);
        if(!ok)$fatal(1,"TRANSPORT_%s case=%0d cycle=%0d",why,scenario,cycle);
    endtask
    task step;@(negedge clk);#0.2;endtask

    always @(negedge clk)begin
        fetch_ready=rst_n&&!hold_fetch;
        // A legal producer retains VALID/data while stalled. Only the explicit
        // unsolicited-data negative probe is allowed to violate that contract.
        if(!rst_n||unsolicited||payload_probe||!payload_valid||payload_sent)begin
            payload_valid=rst_n&&(src_active||unsolicited)&&!hold_payload&&(cycle%11!=4);
            payload_data=pattern(src_tag,src_index);
        end
        payload_probe=unsolicited;
        for(integer ch=0;ch<2;ch++)begin
            m_awready[ch]=rst_n&&!hold_aw[ch]&&cycle%5!=0;
            m_wready[ch]=rst_n&&!hold_w[ch]&&cycle%7!=0;
            cpl_ready[ch]=rst_n&&!hold_cpl[ch]&&cycle%13!=0;
            if(!m_bvalid[ch]||b_sent[ch]||!rst_n)begin
                m_bvalid[ch]=rst_n&&bs[ch]<b_tail[ch]&&cycle>=b_due[ch*1024+bs[ch]]&&!hold_b[ch];
                m_bid[ch*4+:4]=0;
                m_bresp[ch*2+:2]=order[ch*1024+bs[ch]]==error_tag ? 2 : 0;
            end
        end
    end

    always @(posedge clk)begin
        cycle++;
        b_sent=m_bvalid&m_bready;
        payload_sent=payload_valid&&payload_ready;
        if(!rst_n)begin
            src_active=0;src_index=0;aw_held=0;w_held=0;cpl_held=0;
            fetch_held=0;fence_held=0;barrier_pending=0;input_held=0;
            for(p=0;p<2;p++)begin
                submitted[p]=0;fetched[p]=0;aws[p]=0;ws[p]=0;bs[p]=0;cpls[p]=0;b_tail[p]=0;
            end
            for(p=0;p<1024;p++)begin input_beats[p]=0;output_beats[p]=0;lengths[p]=0;status[p]=0;end
        end else begin
            $display("TE S %0d %0d %b %b %b %h %h %0d %0d %0d %b %b %0d %b %b",
                cycle,scenario,stop,fault,drained,queued,banks_used,dut.fence_state,
                dut.ingress_count[0],dut.ingress_count[1],dut.completion_empty,dut.core_drained,
                dut.payload_budget,submit_ready,fence_done_valid);
            need(!fetch_held || (fetch_valid&&{fetch_channel,fetch_desc}===saved_fetch),"FETCH_STABLE");
            need(!input_held || (payload_valid&&payload_data===saved_input),"PRODUCER_PAYLOAD_STABLE");
            input_held=payload_valid&&!payload_ready&&!payload_probe;
            saved_input=payload_data;
            need(!fence_held || (fence_done_valid&&fence_done_id===saved_fence),"FENCE_STABLE");
            fetch_held=fetch_valid&&!fetch_ready;saved_fetch={fetch_channel,fetch_desc};
            fence_held=fence_done_valid&&!fence_done_ready;saved_fence=fence_done_id;
            for(p=0;p<2;p++)begin
                need(!aw_held[p] || (m_awvalid[p]&&
                    {m_awaddr[p*64+:64],m_awlen[p*8+:8],m_awid[p*4+:4],m_awcache[p*4+:4],m_awsize[p*3+:3],m_awburst[p*2+:2]}===saved_aw[p*85+:85]),"AW_STABLE");
                need(!w_held[p] || (m_wvalid[p]&&{m_wdata[p*512+:512],m_wstrb[p*64+:64],m_wlast[p]}===saved_w[p*577+:577]),"W_STABLE");
                need(!cpl_held[p] || (cpl_valid[p]&&{cpl_tag[p*64+:64],cpl_status[p*2+:2]}===saved_cpl[p*66+:66]),"CPL_STABLE");
                aw_held[p]=m_awvalid[p]&&!m_awready[p];
                saved_aw[p*85+:85]={m_awaddr[p*64+:64],m_awlen[p*8+:8],m_awid[p*4+:4],m_awcache[p*4+:4],m_awsize[p*3+:3],m_awburst[p*2+:2]};
                w_held[p]=m_wvalid[p]&&!m_wready[p];saved_w[p*577+:577]={m_wdata[p*512+:512],m_wstrb[p*64+:64],m_wlast[p]};
                cpl_held[p]=cpl_valid[p]&&!cpl_ready[p];saved_cpl[p*66+:66]={cpl_tag[p*64+:64],cpl_status[p*2+:2]};
                if(submit_valid[p]&&submit_ready[p])begin
                    t=int'(submit_desc[p*232+136+:64]);need(t<1024&&submitted[p]<1024,"TAG_BOUNDS");
                    need(!barrier_pending,"NO_PUBLICATION_DURING_BARRIER");
                    order[p*1024+submitted[p]]=t;submitted[p]++;lengths[t]=int'(submit_desc[p*232+:8])+1;
                    $display("TE SUB %0d %0d %0d %h",cycle,p,t,submit_desc[p*232+:232]);
                end
                if(dut.i_submit_valid[p]&&dut.i_submit_ready[p])
                    $display("TE ADMIT %0d %0d %0d",cycle,p,dut.i_submit_desc[p*232+136+:64]);
                if(dut.core.reserve_take[p])
                    $display("TE RES %0d %0d %0d",cycle,p,dut.core.selected_desc[199:136]);
            end
            if(fetch_valid&&fetch_ready)begin
                t=int'(fetch_desc[199:136]);p=fetch_channel;
                need(!src_active&&fetched[p]<submitted[p]&&order[p*1024+fetched[p]]==t,"FETCH_OWNER");
                need(fetch_desc===descriptor(t,lengths[t]),"FETCH_DESCRIPTOR");
                fetched[p]++;src_active=1;src_tag=t;src_beats=lengths[t];src_index=0;
                $display("TE FETCH %0d %0d %0d %h",cycle,p,t,fetch_desc);
            end
            if(payload_valid&&payload_ready)begin
                need(src_active&&src_index<src_beats&&payload_data===pattern(src_tag,src_index),"INPUT_BUDGET_DATA");
                $display("TE IN %0d %0d %0d %h",cycle,src_tag,src_index,payload_data);
                input_beats[src_tag]++;src_index++;if(src_index==src_beats)src_active=0;
            end
            if(fence_valid&&fence_ready)begin
                need(!barrier_pending,"ONE_BARRIER");barrier_pending=1;barrier_id=fence_id;
                $display("TE FENCE %0d %0d",cycle,fence_id);
            end
            for(p=0;p<2;p++)begin
                if(m_awvalid[p]&&m_awready[p])begin
                    t=order[p*1024+aws[p]];need(aws[p]<fetched[p],"AW_AFTER_FETCH");
                    need(m_awaddr[p*64+:64]==64'h100000000+64'(t)*4096&&int'(m_awlen[p*8+:8])+1==lengths[t],"AW_METADATA");
                    need(m_awsize[p*3+:3]==6&&m_awburst[p*2+:2]==1&&m_awid[p*4+:4]==0&&m_awcache[p*4+:4]==0,"AW_ATTRIBUTES");
                    aws[p]++;$display("TE AW %0d %0d %0d %h %0d",cycle,p,t,m_awaddr[p*64+:64],lengths[t]);
                end
                if(m_wvalid[p]&&m_wready[p])begin
                    t=order[p*1024+ws[p]];k=output_beats[t];
                    need(ws[p]<fetched[p]&&k<input_beats[t],"W_AFTER_INPUT");
                    need(m_wdata[p*512+:512]===pattern(t,k)&&m_wstrb[p*64+:64]==64'hffffffffffffffff,"OUTPUT_DATA");
                    need(m_wlast[p]==(k==lengths[t]-1),"WLAST");
                    $display("TE W %0d %0d %0d %0d %b %h",cycle,p,t,k,m_wlast[p],m_wdata[p*512+:512]);
                    output_beats[t]++;if(m_wlast[p])ws[p]++;
                end
                while(b_tail[p]<aws[p]&&b_tail[p]<ws[p])begin b_due[p*1024+b_tail[p]]=cycle+20+order[p*1024+b_tail[p]]%7;b_tail[p]++;end
                if(m_bvalid[p]&&m_bready[p])begin
                    t=order[p*1024+bs[p]];need(bs[p]<b_tail[p],"B_CAUSAL");
                    status[t]=m_bresp[p*2+:2];bs[p]++;$display("TE B %0d %0d %0d %0d",cycle,p,t,status[t]);
                end
                if(dut.i_cpl_valid[p]&&dut.i_cpl_ready[p])
                    $display("TE CORE_CPL %0d %0d %0d",cycle,p,dut.i_cpl_tag[p*64+:64]);
                if(cpl_valid[p]&&cpl_ready[p])begin
                    t=order[p*1024+cpls[p]];need(cpls[p]<bs[p]&&cpl_tag[p*64+:64]==64'(t)&&cpl_status[p*2+:2]==status[t],"CPL_CAUSAL_ORDER");
                    cpls[p]++;$display("TE CPL %0d %0d %0d %0d",cycle,p,t,status[t]);
                end
            end
            if(fence_done_valid)begin
                need(barrier_pending&&fence_done_id==barrier_id&&cpls[0]==submitted[0]&&cpls[1]==submitted[1],"FENCE_AFTER_EXTERNAL_COMPLETIONS");
                if(fence_done_ready)begin $display("TE FENCED %0d %0d",cycle,fence_done_id);barrier_pending=0;end
            end
            if(drained)need(!barrier_pending&&!src_active&&cpls[0]==submitted[0]&&cpls[1]==submitted[1],"DRAIN_INCLUDES_BOUNDARY");
        end
    end
    task reset_case(input integer number);
        rst_n=0;stop=0;fault_in=0;submit_valid=0;fence_valid=0;fence_done_ready=0;
        hold_fetch=0;hold_payload=0;unsolicited=0;hold_aw=0;hold_w=0;hold_b=0;hold_cpl=0;error_tag=-1;
        repeat(4)step;scenario=number;rst_n=1;$display("TE CASE %0d %0d",cycle,scenario);step;
    endtask
    task push_one(input integer ch,input integer tag,input integer beats);
        submit_valid[ch]=1;submit_desc[ch*232+:232]=descriptor(tag,beats);
        begin: waiting
            for(integer tries=0;tries<5000;tries++)begin @(posedge clk);if(submit_ready[ch])begin step;submit_valid[ch]=0;disable waiting;end step;end
            need(0,"SUBMIT_TIMEOUT");
        end
    endtask
    task wait_cpl(input integer ch,input integer count);
        begin: waiting
            for(integer tries=0;tries<10000;tries++)begin step;if(cpls[ch]==count)disable waiting;end
            need(0,"CPL_TIMEOUT");
        end
    endtask
    task wait_fetch;
        begin: waiting
            for(integer tries=0;tries<1000;tries++)begin step;if(fetch_valid)disable waiting;end
            need(0,"FETCH_TIMEOUT");
        end
    endtask
    task drain_all;
        begin: waiting
            for(integer tries=0;tries<10000;tries++)begin step;if(drained)disable waiting;end
            need(0,"DRAIN_TIMEOUT");
        end
        need(!fault&&cpls[0]==submitted[0]&&cpls[1]==submitted[1],"CLEAN_DRAIN");
        $display("TE DRAIN %0d",cycle);
    endtask
    task send_fence(input integer id);
        fence_id=id;fence_valid=1;
        begin: waiting
            for(integer tries=0;tries<1000;tries++)begin @(posedge clk);if(fence_ready)begin step;fence_valid=0;disable waiting;end step;end
            need(0,"FENCE_ADMISSION_TIMEOUT");
        end
    endtask
    task wait_fence;
        begin: waiting
            for(integer tries=0;tries<5000;tries++)begin step;if(fence_done_valid)disable waiting;end
            need(0,"FENCE_RESULT_TIMEOUT");
        end
    endtask
    task pass_case;$display("TE CASE_PASS %0d %0d %0d %0d %0d",scenario,submitted[0],submitted[1],cpls[0],cpls[1]);endtask

    initial begin
        reset_case(1);
        for(integer j=0;j<24;j++)push_one(j%2,j+1,j%3==0?1:j%3==1?8:64);
        drain_all;pass_case;

        reset_case(2);hold_fetch=1;push_one(0,100,64);
        need(dut.ingress_count[0]==1,"STOP_WITH_STAGED_DESCRIPTOR");stop=1;step;stop=0;
        wait_fetch;repeat(8)begin step;need(fetch_valid&&!drained&&submit_ready==0,"STOP_RETAINS_FETCH");end
        hold_fetch=0;submit_valid=1;submit_desc[231:0]=descriptor(101,1);drain_all;
        need(submitted[0]==1&&submit_ready==0,"STOP_IS_LATCHED");pass_case;

        reset_case(3);hold_cpl=3;push_one(0,200,1);push_one(1,201,8);
        need(dut.ingress_count[1]==1,"FENCE_WITH_STAGED_DESCRIPTOR");send_fence(33);repeat(150)step;
        need(dut.i_fence_done_valid&&!fence_done_valid&&cpl_valid==3&&!drained,"FENCE_CANNOT_OVERTAKE_BUFFERED_CPL");
        hold_cpl=2;wait_cpl(0,1);repeat(10)step;need(!fence_done_valid,"FENCE_WAITS_BOTH_LANES");
        hold_cpl=0;wait_fence;repeat(5)begin step;need(fence_done_valid&&submit_ready==0,"FENCE_RESULT_HELD");end
        fence_done_ready=1;step;fence_done_ready=0;drain_all;pass_case;

        reset_case(4);hold_fetch=1;push_one(0,300,64);push_one(1,301,8);wait_fetch;
        fault_in=1;step;fault_in=0;repeat(8)begin step;need(fetch_valid&&fault&&!drained,"FAULT_RETAINS_COMMITTED_FETCH");end
        hold_fetch=0;wait_cpl(0,1);repeat(8)step;
        need(queued[7:4]==1&&cpls[1]==0&&fault&&!drained,"FAULT_QUARANTINE");pass_case;

        reset_case(5);hold_b=1;push_one(0,400,1);
        begin: first_write
            for(integer j=0;j<1000;j++)begin step;if(ws[0]==1)disable first_write;end
            need(0,"FIRST_WRITE_TIMEOUT");
        end
        hold_fetch=1;push_one(1,401,64);push_one(0,402,8);wait_fetch;error_tag=400;hold_b=0;
        begin: fault_wait
            for(integer j=0;j<1000;j++)begin step;if(fault)disable fault_wait;end
            need(0,"BRESP_TIMEOUT");
        end
        need(fetch_valid,"ERROR_RETAINS_FETCH");hold_fetch=0;wait_cpl(1,1);wait_cpl(0,1);repeat(8)step;
        need(queued[3:0]==1&&!drained&&status[400]==2&&status[401]==0,"BRESP_QUARANTINE");pass_case;

        reset_case(6);hold_w=1;
        for(integer j=0;j<8;j++)push_one(0,500+j,64);
        repeat(700)step;need(fetched[0]==6&&queued[3:0]==2,"BANK_LIMIT");
        for(integer j=0;j<8;j++)push_one(1,520+j,64);
        wait_cpl(1,8);need(cpls[0]==0,"BLOCKED_DESTINATION_INDEPENDENCE");hold_w=0;stop=1;drain_all;pass_case;

        reset_case(7);hold_cpl=1;
        for(integer j=0;j<28;j++)push_one(0,600+j,1);
        repeat(200)step;
        need(dut.core.receiver.core.occupied[0]==16&&fetched[0]==18&&queued[3:0]==10&&
             dut.ingress_count[0]==2&&banks_used==0,"RESULT_BUFFER_PLUS_DESCRIPTOR_LIMIT");
        submit_valid=1;submit_desc[231:0]=descriptor(628,1);
        repeat(8)begin step;need(!submit_ready[0]&&submitted[0]==28,"FULL_INGRESS_BACKPRESSURE");end
        // Keep the blocked lane's VALID/descriptor stable through peer traffic
        // and stop, until reset. Its unaccepted tag must never enter the DUT.
        for(integer j=0;j<8;j++)push_one(1,630+j,8);
        wait_cpl(1,8);need(cpls[0]==0,"OTHER_COMPLETION_LANE_PROGRESS");
        stop=1;step;stop=0;hold_cpl=0;drain_all;
        need(submit_ready==0&&submitted[0]==28,"STOP_DRAINS_FULL_INGRESS");pass_case;

        reset_case(8);hold_aw=1;push_one(0,700,1);push_one(0,701,1);
        repeat(100)step;need(ws[0]==2&&aws[0]==0&&!drained,"W_BEFORE_AW");hold_aw=0;drain_all;pass_case;

        reset_case(9);unsolicited=1;repeat(8)begin step;need(!payload_ready&&banks_used==0,"NO_UNSOLICITED_PAYLOAD");end
        unsolicited=0;hold_payload=1;hold_cpl=1;push_one(0,750,8);repeat(20)step;stop=1;repeat(8)step;
        need(dut.payload_budget==8&&banks_used!=0&&!drained,"STOP_WITH_PAYLOAD_BUDGET");hold_payload=0;
        begin: buffered_result
            for(integer j=0;j<1000;j++)begin step;if(dut.core_drained&&cpl_valid[0])disable buffered_result;end
            need(0,"BUFFERED_RESULT_TIMEOUT");
        end
        unsolicited=1;repeat(8)begin step;need(!payload_ready&&!drained&&dut.core_drained,"DRAIN_WAITS_EXTERNAL_CPL");end
        hold_cpl=0;drain_all;pass_case;

        reset_case(10);hold_b=1;push_one(0,800,8);send_fence(100);repeat(100)step;
        need(!fence_done_valid&&ws[0]==1,"FENCE_WAITS_B");fault_in=1;step;fault_in=0;hold_b=0;wait_cpl(0,1);
        repeat(20)begin step;need(fault&&!fence_done_valid&&!drained,"FAULT_QUARANTINES_PENDING_FENCE");end pass_case;

        reset_case(11);push_one(0,850,1);drain_all;send_fence(110);wait_fence;
        fault_in=1;step;fault_in=0;repeat(5)begin step;need(fault&&fence_done_valid&&fence_done_id==110,"OFFERED_FENCE_SURVIVES_FAULT");end
        fence_done_ready=1;step;fence_done_ready=0;repeat(5)step;need(drained&&fault,"COMPLETED_FENCE_RELEASES_BOUNDARY");pass_case;

        reset_case(12);hold_fetch=1;push_one(0,900,64);send_fence(120);stop=1;step;stop=0;repeat(30)step;
        need(fetch_valid&&!fence_done_valid&&!drained,"STOP_DRAINS_BEFORE_FENCE");hold_fetch=0;wait_fence;
        fence_done_ready=1;step;fence_done_ready=0;drain_all;need(submit_ready==0,"STOP_AFTER_FENCE");pass_case;

        reset_case(13);push_one(0,950,8);need(dut.ingress_count[0]==1,"FAULT_WITH_STAGED_DESCRIPTOR");
        fault_in=1;step;fault_in=0;repeat(12)step;
        need(fault&&dut.ingress_count[0]==1&&queued==1&&banks_used==0&&!fetch_valid&&!drained,"INGRESS_QUARANTINE_NO_RESERVATION");pass_case;
        $display("TE SUITE_PASS cases=13 cycle=%0d",cycle);$finish;
    end
    initial begin #1000000;$fatal(1,"TRANSPORT_WATCHDOG");end
endmodule
