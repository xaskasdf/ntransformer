`timescale 1ns/1ps
module tb_ntrpu_read_tracker;
    parameter integer DEPTH=4;
    reg clk=0,rst_n=0;
    always #1.111 clk=~clk;
    reg p_arvalid=0,p_rready=0,h_arready=1,h_rvalid=0,h_rlast=0;
    reg [33:0] p_araddr=0;
    reg [3:0] p_arlen=15,p_arid=0;
    reg [2:0] p_arsize=5;
    reg [1:0] p_arburst=1,h_rresp=0;
    reg [255:0] h_rdata=0;
    reg [5:0] h_rid=0;
    wire p_arready,p_rvalid,h_arvalid,h_rready,pending,fault,validated_last;
    wire [33:0] h_araddr;
    wire [3:0] h_arlen;
    wire [5:0] h_arid,validated_rid;
    wire [2:0] h_arsize;
    wire [1:0] h_arburst,validated_resp;
    wire [255:0] validated_data;
    ntrpu_hbm_read_tracker #(.MAX_PENDING(DEPTH)) dut(.*);
    integer scenario=0,received=0,addresses=0,first_length=16;
    function automatic [255:0] pattern(input integer index);
        for(integer j=0;j<8;j++)pattern[j*32+:32]=32'h57a1b000+index*8+j;
    endfunction
    task automatic need(input bit ok,input string label);
        if(!ok)$fatal(1,"TRACKER_UNIT_FAIL case=%0d depth=%0d %s",scenario,DEPTH,label);
    endtask
    always @(posedge clk)begin
        if(!rst_n)begin received=0;addresses=0;end
        else begin
            if(h_arvalid&&h_arready)addresses++;
            if(p_rvalid&&p_rready)begin
                need(validated_data===pattern(received)&&validated_rid==0&&validated_resp==0,"DATA_OR_METADATA");
                need(validated_last==(received==first_length-1||
                    (received>=first_length&&(received-first_length)%16==15)),"VALIDATED_LAST");
                received++;
            end
        end
    end
    reg held=0;
    reg [264:0] saved;
    always @(posedge clk)begin
        if(!rst_n)held<=0;
        else begin
            need(!held||(p_rvalid&&{validated_rid,validated_resp,validated_last,validated_data}===saved),"OFFER_DROPPED");
            held<=p_rvalid&&!p_rready;
            saved<={validated_rid,validated_resp,validated_last,validated_data};
        end
    end
    task automatic boot(input integer number,length0=16);
        @(negedge clk);rst_n=0;p_arvalid=0;p_rready=0;h_arready=1;h_rvalid=0;
        h_rid=0;h_rresp=0;h_rlast=0;scenario=number;first_length=length0;
        repeat(8)@(negedge clk);rst_n=1;
    endtask
    task automatic issue(input integer index,length0=16);
        @(negedge clk);p_araddr=34'h10000+index*512;p_arlen=(index==0?length0:16)-1;p_arvalid=1;
        do @(posedge clk);while(!p_arready);
        @(negedge clk);p_arvalid=0;
    endtask
    task automatic fill(input integer length0=16);
        for(integer i=0;i<DEPTH;i++)issue(i,length0);
        repeat(4)@(negedge clk);
        need(addresses==DEPTH&&dut.reserved_count==DEPTH&&dut.accepted_count==DEPTH&&!p_arready,"FULL_CREDITS");
    endtask
    task automatic beat(input integer index,id,resp,last);
        @(negedge clk);h_rdata=pattern(index);h_rid=id;h_rresp=resp;h_rlast=last;h_rvalid=1;
        do @(posedge clk);while(!h_rready);
        @(negedge clk);h_rvalid=0;
    endtask
    task automatic held_fault(input integer number,id,resp,last,length0=16);
        boot(number,length0);fill(length0);
        beat(0,0,0,0);beat(1,0,0,0);beat(2,id,resp,last);
        repeat(8)@(negedge clk);
        need(fault&&pending&&!h_rready&&!p_arready,"QUARANTINE_FAULT");
        need(dut.reserved_count==DEPTH&&dut.accepted_count==DEPTH,"QUARANTINE_CREDITS");
        need(dut.validated_cut.count==2&&p_rvalid&&received==0,"VALIDATED_HELD");
        p_rready=1;repeat(8)@(negedge clk);
        need(received==2&&!p_rvalid&&fault&&pending,"VALIDATED_DRAIN_WITH_FAULT");
        $display("TU PASS case=%0d depth=%0d delivered=%0d reserved=%0d",scenario,DEPTH,received,dut.reserved_count);
    endtask
    initial begin
        boot(1);fill();p_rready=1;
        #0.05;h_rid=32;h_rresp=1;h_rlast=1;#0.05;
        need(h_rready,"READY_DEPENDENCE");h_rid=0;h_rresp=0;h_rlast=0;
        for(integer f=0;f<DEPTH;f++)begin
            for(integer i=0;i<16;i++)beat(f*16+i,0,0,i==15);
            need(dut.reserved_count==DEPTH-f,"EARLY_CREDIT_RELEASE");
            repeat(8)@(negedge clk);
            need(dut.reserved_count==DEPTH-f-1,"MISSING_CREDIT_RELEASE");
        end
        need(received==DEPTH*16&&!pending&&!fault,"CLEAN_DRAIN");
        $display("TU PASS case=1 depth=%0d delivered=%0d reserved=0",DEPTH,received);
        held_fault(2,32,0,0);
        held_fault(3,0,0,1);
        held_fault(4,0,0,0,3);
        held_fault(5,0,1,0);
        boot(6);h_arready=0;issue(0);repeat(4)@(negedge clk);
        need(h_arvalid&&h_araddr==34'h10000,"HELD_AR");
        h_rvalid=1;h_rid=32;repeat(4)@(negedge clk);
        need(fault&&h_arvalid&&h_araddr==34'h10000&&dut.accepted_count==0,"FAULT_RETAIN_AR");
        h_arready=1;repeat(4)@(negedge clk);h_rvalid=0;
        need(addresses==1&&dut.accepted_count==1&&dut.reserved_count==1&&pending&&!h_rready,"FAULT_AR_ACCEPTANCE");
        $display("TU PASS case=6 depth=%0d delivered=0 reserved=1",DEPTH);
        boot(7);repeat(4)@(negedge clk);need(!pending&&!fault,"COLD_RESET");
        $display("TU ALL_PASS depth=%0d cases=6",DEPTH);$finish;
    end
    initial begin #100000;$fatal(1,"TRACKER_UNIT_WATCHDOG");end
endmodule
