`timescale 1ns/1ps
module tb_ntrpu_bridge_fifo_rev2;
    parameter integer WIDTH=514;
    integer seed=0,ratio=0,writes=0,reads=0,total_writes=0,total_reads=0;
    reg wclk=0,rclk=0,we=1,re=1,wrst_n=0,rrst_n=0,wvalid=0,rready=0;
    reg [WIDTH-1:0] wdata=0;
    wire wready,rvalid;
    wire [WIDTH-1:0] rdata;
    reg [WIDTH-1:0] expected[0:4095];
    reg held=0;
    reg [WIDTH-1:0] held_data;
    real wh=2.5,rh=2.0;
    integer target,counter;
    ntrpu_bridge_async_fifo #(.WIDTH(WIDTH)) dut(.*);
    initial begin
        if($value$plusargs("seed=%d",seed))begin end
        ratio=seed%3;
        if(ratio==1)begin wh=2.0;rh=3.5;end
        if(ratio==2)begin wh=3.5;rh=2.0;end
        forever #(wh)if(we)wclk=~wclk;
    end
    initial begin #0.001;#(0.1*(seed%10));forever #(rh)if(re)rclk=~rclk;end
    initial begin #1000000;$fatal(1,"FIFO_REV2 WATCHDOG width=%0d seed=%0d",WIDTH,seed);end
    function automatic [WIDTH-1:0] word(input integer n);
        reg [WIDTH-1:0] v;
        begin
            for(integer i=0;i<WIDTH;i=i+1)v[i]=1'((n*37+seed*13+i*19)^(n>>(i%11))^(i>>2));
            word=v;
        end
    endfunction
    always @(posedge wclk)begin
        if(!wrst_n)writes=0;
        else if(wvalid && wready)begin
            expected[writes]=wdata;writes=writes+1;total_writes=total_writes+1;
        end
    end
    always @(posedge rclk)begin
        if(!rrst_n)begin reads=0;held=0;end
        else begin
            if(held && (!rvalid || rdata!==held_data))$fatal(1,"FIFO_REV2 UNSTABLE");
            if(rvalid && rready)begin
                if(reads>=writes || rdata!==expected[reads])$fatal(1,"FIFO_REV2 ORDER_DATA reads=%0d writes=%0d",reads,writes);
                reads=reads+1;total_reads=total_reads+1;
            end
            held=rvalid && !rready;held_data=rdata;
        end
    end
    task automatic reset_both;
        begin
            @(negedge wclk);wrst_n=0;rrst_n=0;wvalid=0;rready=0;
            repeat(3)@(negedge wclk);repeat(3)@(negedge rclk);
            @(negedge wclk);wrst_n=1;@(negedge rclk);rrst_n=1;
        end
    endtask
    task automatic send(input integer n);
        begin
            @(negedge wclk);wdata=word(n);wvalid=1;
            @(posedge wclk);while(!wready)@(posedge wclk);
            @(negedge wclk);wvalid=0;
        end
    endtask
    initial begin
        reset_both();
        for(integer i=0;i<5;i=i+1)send(i);
        wait(rvalid);@(negedge rclk);re=0;
        @(negedge wclk);wdata=word(5);wvalid=1;
        repeat(12)@(negedge wclk);
        if(writes!=5 || reads || wready || !rvalid || rdata!==word(0))$fatal(1,"FIFO_REV2 CAPACITY_OR_OVERWRITE");
        re=1;@(negedge rclk);rready=1;@(negedge rclk);rready=0;
        wait(writes==6);@(negedge wclk);wvalid=0;
        repeat(6)@(negedge wclk);
        if(writes!=6 || reads!=1 || wready)$fatal(1,"FIFO_REV2 CREDIT");
        @(negedge rclk);rready=1;wait(reads==6);@(negedge rclk);rready=0;
        repeat(4)@(negedge rclk);if(rvalid)$fatal(1,"FIFO_REV2 EMPTY");

        for(integer i=0;i<5;i=i+1)send(100+i);
        repeat(8)@(negedge rclk);@(negedge wclk);we=0;
        @(negedge rclk);rready=1;
        repeat(5)begin @(posedge rclk);if(!rvalid)$fatal(1,"FIFO_REV2 PREFILLED_BUBBLE");end
        @(negedge rclk);rready=0;we=1;
        if(reads!=11)$fatal(1,"FIFO_REV2 THROUGHOUT_COUNT");

        fork
            begin
                for(integer i=0;i<240;i=i+1)begin
                    @(negedge wclk);wdata=word(1000+writes);wvalid=((i+seed)%5!=0);
                end
                @(negedge wclk);wvalid=0;
            end
            begin
                for(integer i=0;i<420;i=i+1)begin
                    @(negedge rclk);rready=((i+seed)%4!=0 && (i%37)<25);
                end
                @(negedge rclk);rready=1;
            end
        join
        wait(reads==writes);@(negedge rclk);rready=0;
        if(reads<60)$fatal(1,"FIFO_REV2 INSUFFICIENT_TRAFFIC");
        for(integer i=0;i<5;i=i+1)send(5000+i);
        wait(rvalid);reset_both();
        repeat(12)@(negedge rclk);if(rvalid)$fatal(1,"FIFO_REV2 STALE_AFTER_RESET");
        send(6000);@(negedge rclk);rready=1;wait(reads==1);@(negedge rclk);rready=0;
        repeat(4)@(negedge rclk);
        if(rvalid || total_writes-total_reads!=5)$fatal(1,"FIFO_REV2 RESET_ACCOUNTING");
        $display("FIFO_REV2 PASS width=%0d seed=%0d written=%0d read=%0d discarded=5",WIDTH,seed,total_writes,total_reads);
        $finish;
    end
endmodule
