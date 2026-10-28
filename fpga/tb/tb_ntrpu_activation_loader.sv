`timescale 1ns/1ps
module tb_ntrpu_activation_loader;
    logic clk=0,rst_n=0;
    always #2.5 clk=~clk;
    logic cmd_valid=0,cmd_ready,cancel_valid=0,cancel_ready;
    logic [63:0] cmd_addr=0,cmd_region_base=0,cmd_region_bytes=0;
    logic [8:0] cmd_blocks=0;
    logic [31:0] cmd_sequence=0,cmd_epoch=0,expected_epoch=0;
    logic activation_valid,activation_ready=0;
    logic [7:0] activation_index;
    logic [255:0] activation_values;
    logic [63:0] activation_scale;
    logic [31:0] activation_epoch;
    logic done_valid,done_ready=0;
    logic [3:0] done_status;
    logic [8:0] done_blocks;
    logic [31:0] done_sequence,done_epoch;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize,m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid,m_axi_arcache;
    logic m_axi_arlock,m_axi_arvalid,m_axi_arready=0;
    logic [511:0] m_axi_rdata=0;
    logic [1:0] m_axi_rresp=0;
    logic [3:0] m_axi_rid=0;
    logic m_axi_rlast=0,m_axi_rvalid=0,m_axi_rready;
    ntrpu_activation_loader #(.WATCHDOG_CYCLES(32)) dut(.*);
    localparam logic [63:0] BASE=64'h8000000000;
    int seed=1,random_state,case_id=0,total_bursts=0,total_outputs=0;
    int output_stalls=0,ar_stalls=0,done_stalls=0,reset_cases=0;
    function automatic logic [511:0] record(input int index,input int scenario,input int n);
        record=0;
        for(int lane=0;lane<32;lane++) record[lane*8+:8]=8'(index*13+lane*7+case_id);
        record[319:256]=64'h3ff0000000000000;
        if ((scenario==7 || scenario==20) && index==n-1) record[511]=1;
        if (scenario==8 && index==n-1) record[319:256]=64'h7ff8000000000000;
        if (scenario==9 && index==0) record[319:256]=64'hbff0000000000000;
        if (scenario==10 && index==0) record[319:256]=64'h4780204070e1c388;
        if (scenario==23) record[319:256]=64'h8000000000000000;
        if (scenario==24) record[319:256]=64'h0000000000000001;
        if (scenario==25) record[319:256]=64'h4780204070e1c387;
        if (scenario==26) record[319:256]=0;
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL ACTIVATION case=%0d: %s",case_id,why);
    endtask
    task automatic run_case(input int n,input int offset,input int scenario,input int status);
        bit memory_active,accepted,finished,cancelled,reset_done;
        bit ar_stalled,out_stalled,done_stalled,ar_hs,r_hs,out_hs,cmd_hs,cancel_hs;
        logic [71:0] saved_ar;
        logic [359:0] saved_output;
        logic [76:0] saved_done;
        logic [511:0] want;
        logic [63:0] next_address,burst_address;
        int cycles,outputs,bursts,returned,beat,beat_count,done_wait,remaining,index;
        case_id++;memory_active=0;accepted=0;finished=0;cancelled=0;reset_done=0;
        ar_stalled=0;out_stalled=0;done_stalled=0;
        cycles=0;outputs=0;bursts=0;returned=0;beat=0;beat_count=0;done_wait=0;remaining=n;
        @(negedge clk);
        require(cmd_ready,"previous command not released");
        cmd_addr=BASE+64'(offset);cmd_region_base=BASE;cmd_region_bytes=64'h100000;
        cmd_blocks=9'(n);cmd_sequence=32'(case_id);cmd_epoch=32'(case_id+100);expected_epoch=cmd_epoch;
        if(scenario==2) cmd_addr++;
        if(scenario==3) cmd_region_bytes=64;
        if(scenario==4) expected_epoch++;
        if(scenario==5) cmd_region_base=cmd_addr+64;
        if(scenario==6) begin cmd_addr=64'hffffffffffffffc0;cmd_region_base=cmd_addr;cmd_region_bytes=4096;end
        next_address=cmd_addr;cmd_valid=1;cancel_valid=0;m_axi_rvalid=0;
        while(!finished) begin
            m_axi_arready=$urandom_range(0,3)!=0;
            if(scenario==15 && cycles<20) m_axi_arready=0;
            activation_ready=cycles>=15 && $urandom_range(0,3)!=0;
            if((scenario==18 || scenario==22) && cycles<50) activation_ready=0;
            done_ready=done_wait>=5 && $urandom_range(0,2)!=0;
            if(!cancelled && cancel_ready &&
               ((scenario==14 && cycles==2) || (scenario==15 && m_axi_arvalid && cycles>=8) ||
                (scenario==16 && returned>=4) || (scenario==18 && activation_valid))) cancel_valid=1;
            if(!m_axi_rvalid && memory_active && $urandom_range(0,3)!=0 && !(scenario==17 && cycles<120)) begin
                index=int'((burst_address-cmd_addr)/64)+beat;
                m_axi_rdata=record(index,scenario,n);m_axi_rresp=0;m_axi_rid=0;
                if(scenario==11 && index==n-1) m_axi_rresp=2;
                if(scenario==12 && index==2) m_axi_rid=1;
                m_axi_rlast=beat==beat_count-1;
                if(scenario==13 && beat==2) m_axi_rlast=1;
                m_axi_rvalid=1;
            end
            @(posedge clk);
            cycles++;require(cycles<6000,"timeout");
            cmd_hs=cmd_valid&&cmd_ready;cancel_hs=cancel_valid&&cancel_ready;
            ar_hs=m_axi_arvalid&&m_axi_arready;r_hs=m_axi_rvalid&&m_axi_rready;
            out_hs=activation_valid&&activation_ready;
            if(ar_stalled) require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"AR stability");
            if(out_stalled) require(activation_valid && {activation_index,activation_values,activation_scale,activation_epoch}===saved_output,"output stability through cancel/error");
            if(done_stalled) require(done_valid && {done_status,done_blocks,done_sequence,done_epoch}===saved_done,"completion stability");
            ar_stalled=m_axi_arvalid&&!m_axi_arready;out_stalled=activation_valid&&!activation_ready;done_stalled=done_valid&&!done_ready;
            saved_ar={m_axi_araddr,m_axi_arlen};saved_output={activation_index,activation_values,activation_scale,activation_epoch};
            saved_done={done_status,done_blocks,done_sequence,done_epoch};
            ar_stalls+=int'(ar_stalled);output_stalls+=int'(out_stalled);done_stalls+=int'(done_stalled);
            if(cmd_hs) begin require(!accepted,"duplicate command");accepted=1;end
            if(cancel_hs) cancelled=1;
            if(ar_hs) begin
                require(accepted && !memory_active,"unexpected AXI request");
                if(scenario==15 || scenario==16 || scenario==17 || scenario==18) require(bursts==0,"new burst after cancellation/timeout");
                require(m_axi_araddr==next_address,"address/high bits");
                require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"4KiB crossing");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AXI attributes");
                beat_count=int'(m_axi_arlen)+1;
                require(beat_count==(remaining<(4096-int'(next_address[11:0]))/64 ? remaining:(4096-int'(next_address[11:0]))/64),"burst length");
                burst_address=next_address;next_address+=64'(beat_count*64);remaining-=beat_count;
                memory_active=1;beat=0;bursts++;
            end
            if(r_hs) begin require(memory_active,"unsolicited R");beat++;returned++;if(m_axi_rlast) memory_active=0;end
            if(out_hs) begin
                want=record(outputs,scenario>=23 ? scenario:0,n);
                require(activation_index==8'(outputs) && activation_epoch==cmd_epoch,"activation index/epoch");
                require({activation_scale,activation_values}===want[319:0],"activation bytes/scale");
                if(scenario==7 || scenario==8 || scenario==20) require(outputs<n-1,"malformed last record forwarded");
                if(scenario==9 || scenario==10) require(0,"invalid scale forwarded");
                outputs++;
            end
            if(done_valid) begin
                done_wait++;
                require(!memory_active && !activation_valid,"completion before draining AXI/output");
                require(done_status==4'(status),$sformatf("status=%0d expected=%0d",done_status,status));
                require(done_blocks==9'(outputs) && done_sequence==cmd_sequence && done_epoch==cmd_epoch,"completion count/identity");
                if(status==0) require(outputs==n,"incomplete successful vector");
                if(status==1 || status==2 || scenario==14) require(bursts==0 && outputs==0,"invalid command touched memory");
                if(done_ready) finished=1;
            end
            #1;
            if(cmd_hs) cmd_valid=0;
            if(cancel_hs) cancel_valid=0;
            if(r_hs) m_axi_rvalid=0;
            if((scenario==21 && memory_active && returned>=2) || (scenario==22 && out_stalled)) begin
                @(negedge clk);rst_n=0;cmd_valid=0;cancel_valid=0;m_axi_rvalid=0;memory_active=0;
                repeat(4) @(negedge clk);
                rst_n=1;repeat(8) @(negedge clk);
                require(cmd_ready && !done_valid && !activation_valid && !m_axi_arvalid,"stale state after coordinated reset");
                reset_done=1;finished=1;reset_cases++;
            end
            if(!finished) @(negedge clk);
        end
        if(reset_done) $display("NTRPU_ACTIVATION_RESET case=%0d scenario=%0d outputs=%0d",case_id,scenario,outputs);
        else $display("NTRPU_ACTIVATION_CASE id=%0d blocks=%0d offset=%0d scenario=%0d status=%0d cycles=%0d bursts=%0d returned=%0d outputs=%0d",case_id,n,offset,scenario,status,cycles,bursts,returned,outputs);
        total_bursts+=bursts;total_outputs+=outputs;
        repeat(3) @(negedge clk);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_ACTIVATION_SEED seed=%0d",seed);random_state=$urandom(seed);
        repeat(4) @(negedge clk);rst_n=1;@(posedge clk);#1;
        run_case(1,0,0,0);run_case(23,4032,0,0);run_case(256,4032,0,0);
        run_case(64,0,0,0);run_case(33,64,0,0);
        run_case(0,0,1,1);run_case(257,0,1,1);run_case(2,0,2,1);
        run_case(2,0,3,1);run_case(2,0,4,2);run_case(2,0,5,1);run_case(2,0,6,1);
        run_case(23,4032,7,4);run_case(23,4032,8,4);run_case(23,0,9,4);run_case(23,0,10,4);
        run_case(130,4032,11,3);run_case(64,0,12,3);run_case(64,0,13,3);
        run_case(128,0,14,7);run_case(128,0,15,7);run_case(128,0,16,7);
        run_case(128,0,17,8);run_case(128,0,18,7);run_case(130,4032,20,4);
        run_case(128,0,21,0);run_case(128,0,22,0);
        run_case(1,0,23,0);run_case(1,0,24,0);run_case(1,0,25,0);run_case(1,0,26,0);
        run_case(3,4032,0,0);
        require(ar_stalls>0 && output_stalls>0 && done_stalls>0 && reset_cases==2,"coverage counters");
        $display("NTRPU_ACTIVATION_PASS cases=%0d bursts=%0d outputs=%0d ar_stalls=%0d output_stalls=%0d done_stalls=%0d resets=%0d",case_id,total_bursts,total_outputs,ar_stalls,output_stalls,done_stalls,reset_cases);
        $finish;
    end
endmodule
