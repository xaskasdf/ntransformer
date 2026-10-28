`timescale 1ns/1ps
module tb_ntrpu_reserved_dispatch;
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
    ntrpu_reserved_dispatch dut(.*);

    integer cycle=0,case_id=0,src_tag=0,src_beats=0,src_index=0;
    reg src_active=0,hold_fetch=0,hold_payload=0,hold_w0=0,hold_aw0=0,hold_b0=0;
    reg [1:0] hold_cpl=0;
    integer error_tag=-1;
    integer submitted[0:1],issued[0:1],aws[0:1],ws[0:1],bs[0:1],cpls[0:1];
    integer submit_tags[0:1023],tags[0:1023],job_beats[0:1023];
    integer wbeat[0:1],b_due[0:1023],b_tail[0:1],job_status[0:1023];
    integer p,k,t,idx;
    reg [1:0] held_aw=0,held_w=0,held_cpl=0;
    reg [1:0] b_sent=0;
    reg [63:0] saved_aw[0:1],saved_cpl[0:1];
    reg [7:0] saved_len[0:1];
    reg [511:0] saved_w[0:1];
    reg saved_last[0:1];
    reg [1:0] saved_status[0:1];

    function [231:0] descriptor(input integer tag,input integer beats);
        descriptor={32'h76543210,64'(tag),64'h100000000+64'(tag)*4096,
                    64'h800000000+64'(tag)*4096,8'(beats-1)};
    endfunction
    function [511:0] pattern(input integer tag,input integer beat);
        for(integer lane=0;lane<16;lane++)
            pattern[lane*32+:32]=(32'(tag)*32'h9e3779b9)^(32'(beat)*32'h10203)^(32'(lane)*32'h01010101);
    endfunction
    task need(input bit ok,input string why);
        if(!ok)$fatal(1,"RESERVED_%s case=%0d cycle=%0d",why,case_id,cycle);
    endtask

    // Source owns actual data generation; completion comes from DUT beat count.
    always @(negedge clk)begin
        fetch_ready=rst_n&&!hold_fetch;
        payload_valid=rst_n&&src_active&&!hold_payload&&(cycle%11!=4);
        payload_data=pattern(src_tag,src_index);
        for(integer ch=0;ch<2;ch++)begin
            m_awready[ch]=rst_n&&cycle%5!=0&&!(ch==0&&hold_aw0);
            m_wready[ch]=rst_n&&cycle%7!=0&&!(ch==0&&hold_w0);
            cpl_ready[ch]=rst_n&&!hold_cpl[ch]&&cycle%13!=0;
            // Once BVALID is exposed, retain response until its handshake.
            if(!m_bvalid[ch] || b_sent[ch] || !rst_n)begin
                m_bvalid[ch]=rst_n&&bs[ch]<b_tail[ch]&&cycle>=b_due[ch*512+bs[ch]]&&!(ch==0&&hold_b0);
                m_bid[ch*4+:4]=0;
                m_bresp[ch*2+:2]=tags[ch*512+bs[ch]]==error_tag ? 2 : 0;
            end
        end
    end

    always @(posedge clk)begin
        cycle++;
        b_sent=m_bvalid&m_bready;
        if(!rst_n)begin
            src_active=0;src_index=0;held_aw=0;held_w=0;held_cpl=0;
            for(p=0;p<2;p++)begin submitted[p]=0;issued[p]=0;aws[p]=0;ws[p]=0;bs[p]=0;cpls[p]=0;wbeat[p]=0;b_tail[p]=0;end
        end else begin
            $display("RS S %0d %0d %b %b %b %b %h %h %b %b %b %b %b %b %b %b %b %b %b %b %b %b %b %b %b %b",
                cycle,case_id,stop,fault_in,fault,drained,banks_used,queued,dut.reserve_can,dut.reserve_take,
                dut.engine_cmd_ready,dut.lease_valid,fetch_valid,fetch_ready,payload_valid,payload_ready,
                dut.done_pending,dut.done_ready,fence_valid,fence_ready,fence_done_valid,fence_done_ready,
                hold_w0,hold_b0,hold_cpl,hold_payload);
            for(p=0;p<2;p++)if(submit_valid[p]&&submit_ready[p])begin
                t=int'(submit_desc[p*232+136+:64]);need(submitted[p]<512,"SUBMIT_CAPACITY");
                submit_tags[p*512+submitted[p]]=t;submitted[p]++;
                job_beats[t]=int'(submit_desc[p*232+:8])+1;
                $display("RS SUB %0d %0d %0d %h",cycle,p,t,submit_desc[p*232+:232]);
            end
            for(p=0;p<2;p++)if(dut.receiver.core.finish_w[p])begin
                need(dut.receiver.core.return_ready[p]&&!dut.receiver.core.return_error[p],"VALID_BANK_RETURN");
                $display("RS RETURN %0d %0d %0d %0d",cycle,p,
                    dut.receiver.core.return_bank[p*3+:3],dut.receiver.core.return_generation[p*64+:64]);
            end
            if(|dut.reserve_take)begin
                p=dut.selected_channel;t=int'(dut.selected_desc[199:136]);
                need(dut.engine_cmd_ready&&dut.reserve_take!=3&&!dut.lease_valid,"ATOMIC_RESERVATION");
                need(issued[p]<submitted[p]&&submit_tags[p*512+issued[p]]==t,"RESERVATION_FIFO");
                tags[p*512+issued[p]]=t;issued[p]++;
                $display("RS RES %0d %0d %0d %h %0d %0d %0d",cycle,p,t,dut.selected_desc[135:72],
                    int'(dut.selected_desc[7:0])+1,dut.receiver.core.alloc_bank,dut.receiver.core.alloc_generation);
            end
            if(dut.command_valid&&dut.command_ready)$display("RS ECHO %0d %0d %0d",cycle,dut.command_channel,dut.command_desc[199:136]);
            if(fetch_valid&&fetch_ready)begin
                need(!src_active,"ONE_PAYLOAD_STREAM");
                src_tag=int'(fetch_desc[199:136]);src_beats=int'(fetch_desc[7:0])+1;src_index=0;src_active=1;
                need(fetch_desc===descriptor(src_tag,src_beats),"FETCH_METADATA");
                $display("RS FETCH %0d %0d %0d %h",cycle,fetch_channel,src_tag,fetch_desc);
            end
            if(payload_valid&&payload_ready)begin
                need(src_active&&src_index<src_beats,"INPUT_OWNERSHIP");
                need(payload_data===pattern(src_tag,src_index),"SOURCE_PATTERN");
                $display("RS IN %0d %0d %0d %h",cycle,src_tag,src_index,payload_data);
                src_index++;if(src_index==src_beats)src_active=0;
            end
            if(dut.done_pending&&dut.done_ready)begin
                need(!src_active&&dut.received==src_beats,"DONE_AFTER_ACTUAL_BYTES");
                $display("RS DONE %0d %0d %0d",cycle,dut.lease_desc[199:136],dut.received);
            end
            if(fence_valid&&fence_ready)$display("RS FENCE %0d %0d",cycle,fence_id);
            if(fence_done_valid&&fence_done_ready)$display("RS FENCED %0d %0d",cycle,fence_done_id);
            for(p=0;p<2;p++)begin
                need(!(held_aw[p]&&(!m_awvalid[p]||m_awaddr[p*64+:64]!==saved_aw[p]||m_awlen[p*8+:8]!==saved_len[p])),"AW_STABLE");
                need(!(held_w[p]&&(!m_wvalid[p]||m_wdata[p*512+:512]!==saved_w[p]||m_wlast[p]!==saved_last[p])),"W_STABLE");
                need(!(held_cpl[p]&&(!cpl_valid[p]||cpl_tag[p*64+:64]!==saved_cpl[p]||cpl_status[p*2+:2]!==saved_status[p])),"CPL_STABLE");
                held_aw[p]=m_awvalid[p]&&!m_awready[p];saved_aw[p]=m_awaddr[p*64+:64];saved_len[p]=m_awlen[p*8+:8];
                held_w[p]=m_wvalid[p]&&!m_wready[p];saved_w[p]=m_wdata[p*512+:512];saved_last[p]=m_wlast[p];
                held_cpl[p]=cpl_valid[p]&&!cpl_ready[p];saved_cpl[p]=cpl_tag[p*64+:64];saved_status[p]=cpl_status[p*2+:2];
                need(pending_counts[p*5+:5]<=4&&dut.receiver.core.occupied[p]<=16,"CREDIT_BOUNDS");
                if(m_awvalid[p]&&m_awready[p])begin
                    idx=p*512+aws[p];t=tags[idx];need(aws[p]<issued[p],"AW_OWNER");
                    need(m_awaddr[p*64+:64]==64'h100000000+64'(t)*4096&&int'(m_awlen[p*8+:8])+1==job_beats[t],"AW_FIELDS");
                    need(m_awsize[p*3+:3]==6&&m_awburst[p*2+:2]==1&&m_awid[p*4+:4]==0&&m_awcache[p*4+:4]==0,"AW_ATTRIBUTES");
                    $display("RS AW %0d %0d %0d %h %0d",cycle,p,t,m_awaddr[p*64+:64],int'(m_awlen[p*8+:8])+1);aws[p]++;
                end
                if(m_wvalid[p]&&m_wready[p])begin
                    t=tags[p*512+ws[p]];need(ws[p]<issued[p],"W_OWNER");
                    need(m_wstrb[p*64+:64]==64'hffffffffffffffff&&m_wdata[p*512+:512]===pattern(t,wbeat[p]),"W_BYTES");
                    need(m_wlast[p]==(wbeat[p]==job_beats[t]-1),"WLAST");
                    $display("RS W %0d %0d %0d %0d %b %h",cycle,p,t,wbeat[p],m_wlast[p],m_wdata[p*512+:512]);
                    wbeat[p]++;if(m_wlast[p])begin ws[p]++;wbeat[p]=0;end
                end
                while(b_tail[p]<aws[p]&&b_tail[p]<ws[p])begin
                    b_due[p*512+b_tail[p]]=cycle+20+(tags[p*512+b_tail[p]]%7);b_tail[p]++;
                end
                if(m_bvalid[p]&&m_bready[p])begin
                    t=tags[p*512+bs[p]];need(bs[p]<b_tail[p],"B_CAUSAL");
                    job_status[t]=m_bresp[p*2+:2];
                    $display("RS B %0d %0d %0d %0d",cycle,p,t,m_bresp[p*2+:2]);bs[p]++;
                end
                if(cpl_valid[p]&&cpl_ready[p])begin
                    t=tags[p*512+cpls[p]];need(cpls[p]<bs[p]&&cpl_tag[p*64+:64]==64'(t)&&cpl_status[p*2+:2]==job_status[t],"CPL_ORDER_STATUS");
                    $display("RS CPL %0d %0d %0d %0d",cycle,p,t,cpl_status[p*2+:2]);cpls[p]++;
                end
            end
        end
    end

    task step;@(negedge clk);#0.2;endtask
    task reset_case(input integer number);
        rst_n=0;stop=0;fault_in=0;submit_valid=0;fence_valid=0;fence_done_ready=0;
        hold_fetch=0;hold_payload=0;hold_w0=0;hold_aw0=0;hold_b0=0;hold_cpl=0;error_tag=-1;
        repeat(4)step;case_id=number;rst_n=1;
        $display("RS CASE %0d %0d",cycle,case_id);step;
    endtask
    task push_one(input integer ch,input integer tag,input integer beats);
        submit_valid=1<<ch;submit_desc[ch*232+:232]=descriptor(tag,beats);
        begin: waiting
            for(integer tries=0;tries<5000;tries++)begin
                @(posedge clk);if(submit_ready[ch])begin step;submit_valid=0;disable waiting;end
                step;
            end
            need(0,"SUBMIT_TIMEOUT");
        end
    endtask
    task drain_all;
        begin: waiting
            for(integer tries=0;tries<10000;tries++)begin step;if(drained)disable waiting;end
            need(0,"DRAIN_TIMEOUT");
        end
        for(integer ch=0;ch<2;ch++)need(cpls[ch]==submitted[ch]&&aws[ch]==issued[ch]&&ws[ch]==issued[ch]&&bs[ch]==issued[ch],"FULL_DRAIN");
        need(!src_active&&!fault,"CLEAN_DRAIN");
        $display("RS DRAIN %0d",cycle);
    endtask
    task wait_transport;
        begin: waiting
            for(integer tries=0;tries<10000;tries++)begin step;if(dut.engine_drained&&!dut.lease_valid)disable waiting;end
            need(0,"TRANSPORT_TIMEOUT");
        end
    endtask
    task pass_case;
        $display("RS CASE_PASS %0d submitted0=%0d submitted1=%0d issued0=%0d issued1=%0d completed0=%0d completed1=%0d",
            case_id,submitted[0],submitted[1],issued[0],issued[1],cpls[0],cpls[1]);
    endtask

    initial begin
        reset_case(1);
        for(integer j=0;j<24;j++)push_one(j%2,1+j,j%3==0 ? 1 : j%3==1 ? 8 : 64);
        drain_all;pass_case;

        reset_case(2);hold_w0=1;
        for(integer j=0;j<8;j++)push_one(0,100+j,64);
        repeat(700)step;need(issued[0]==6&&banks_used[7:4]==15,"REAL_BANK_LIMIT");
        for(integer j=0;j<12;j++)push_one(1,120+j,64);
        begin: peer_wait
            for(integer tries=0;tries<3000;tries++)begin step;if(cpls[1]==12)disable peer_wait;end
            need(0,"PEER_PROGRESS_TIMEOUT");
        end
        need(cpls[0]==0&&queued[3:0]==2,"PEER_ISOLATION");
        stop=1;hold_w0=0;drain_all;need(submit_ready==0,"STOP_CLOSES_NEW");pass_case;

        reset_case(3);hold_cpl=1;
        for(integer j=0;j<24;j++)push_one(0,200+j,1);
        repeat(200)step;
        need(dut.receiver.core.occupied[0]==16&&banks_used==0&&!dut.reserve_can[0]&&queued[3:0]==8,"REAL_DESCRIPTOR_LIMIT");
        for(integer j=0;j<8;j++)push_one(1,230+j,8);
        begin: other_wait
            for(integer tries=0;tries<1000;tries++)begin step;if(cpls[1]==8)disable other_wait;end
            need(0,"DESCRIPTOR_PEER_TIMEOUT");
        end
        need(cpls[0]==0,"RESULT_HELD");hold_cpl=0;drain_all;pass_case;

        reset_case(4);hold_b0=1;hold_cpl=2;
        push_one(0,260,8);push_one(1,261,8);
        fence_valid=1;fence_id=55;
        do begin @(posedge clk);end while(!fence_ready);
        step;fence_valid=0;repeat(120)step;
        need(!fence_done_valid&&ws[0]==1&&ws[1]==1&&!drained,"FENCE_PENDING_B");
        hold_b0=0;repeat(80)step;need(!fence_done_valid&&cpls[0]==1&&cpls[1]==0,"FENCE_PENDING_RESULT");
        hold_cpl=0;
        begin: fence_wait
            for(integer tries=0;tries<1000;tries++)begin step;if(fence_done_valid)disable fence_wait;end
            need(0,"FENCE_TIMEOUT");
        end
        repeat(5)begin step;need(fence_done_valid&&fence_done_id==55&&submit_ready==0,"FENCE_HELD");end
        fence_done_ready=1;step;fence_done_ready=0;drain_all;pass_case;

        reset_case(5);hold_fetch=1;push_one(0,300,64);push_one(1,301,8);
        repeat(10)step;need(dut.lease_valid&&fetch_valid&&banks_used!=0,"RESERVED_BEFORE_FAULT");
        fault_in=1;step;fault_in=0;repeat(5)step;
        need(fault&&fetch_valid,"COMMIT_SURVIVES_FAULT");hold_fetch=0;wait_transport;
        need(fault&&!drained&&issued[0]==1&&issued[1]==0&&cpls[0]==1&&queued[7:4]==1,"EXTERNAL_FAULT_QUARANTINE");pass_case;

        reset_case(6);hold_b0=1;push_one(0,400,1);
        begin: first_write
            for(integer tries=0;tries<1000;tries++)begin step;if(ws[0]==1)disable first_write;end
            need(0,"FIRST_WRITE_TIMEOUT");
        end
        hold_fetch=1;push_one(1,401,64);push_one(0,402,8);
        repeat(10)step;need(dut.lease_valid&&fetch_valid,"SECOND_RESERVED");
        error_tag=400;hold_b0=0;
        begin: fault_wait
            for(integer tries=0;tries<1000;tries++)begin step;if(fault)disable fault_wait;end
            need(0,"BRESP_FAULT_TIMEOUT");
        end
        need(fetch_valid,"FETCH_RETAINS_ON_B_ERROR");hold_fetch=0;wait_transport;
        need(fault&&!drained&&cpls[0]==1&&cpls[1]==1&&job_status[400]==2&&job_status[401]==0&&queued[3:0]==1,"BRESP_QUARANTINE");pass_case;

        reset_case(7);hold_payload=1;push_one(0,450,64);push_one(1,451,8);
        repeat(15)step;stop=1;repeat(30)step;
        need(dut.lease_valid&&dut.received==0&&banks_used!=0&&submit_ready==0&&!drained,"STOP_RETAINS_BANK");
        hold_payload=0;drain_all;pass_case;

        reset_case(8);hold_aw0=1;push_one(0,480,1);push_one(0,481,1);
        repeat(100)step;need(ws[0]==2&&aws[0]==0&&!drained,"W_BEFORE_AW");
        hold_aw0=0;drain_all;pass_case;
        $display("RS SUITE_PASS cases=8 cycle=%0d",cycle);$finish;
    end
    initial begin #500000;$fatal(1,"RESERVED_WATCHDOG");end
endmodule
