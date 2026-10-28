`timescale 1ns / 1ps
// Boundary-mechanism bank with actual AMD AXI4 and AXI-Lite CDCs.
// Reset/loss are controlled here; no PCIe loss detector or NTRD core claim.
module tb_ntrpu_host_boundary_cut_r2;
    reg host_clk=0,clk=0,host_run=1;
    initial begin #0.731;forever begin #3.65;if(host_run) host_clk=~host_clk;end end
    always #2.5 clk=~clk;
    reg cold_rst_n=0,boundary_rst_n=0,host_rst_n=0,loss=0;
    reg [7:0] hv=0,hr=0,core_ready=0,core_valid=0,response_pause=0;
    wire [7:0] hready,hvalid,in_valid,in_ready,out_valid,out_ready,cdc_response_ready;
    reg [576:0] hdata[0:7],core_data[0:7];
    wire [576:0] hreturn[0:7],raw_request[0:7],raw_response[0:7],core_request[0:7];
    wire invalid;
    wire [7:0] fenced;
    wire [15:0] pending,fault;
    wire [511:0] accepted,delivered,abandoned;
    integer scenario=-1,hcycle=0,cycle=0;
    localparam [7:0] RESP=8'ha4,REQ=8'h5b;
    for(genvar c=0;c<8;c=c+1) begin:wiring
        if(c==2 || c==5 || c==7) begin
            assign in_valid[c]=core_valid[c];
            assign out_ready[c]=cdc_response_ready[c] && !response_pause[c];
            assign hready[c]=0;
        end else begin
            assign out_ready[c]=core_ready[c];
            assign hvalid[c]=0;
            assign hreturn[c]=0;
            assign cdc_response_ready[c]=0;
        end
    end
    ntrpu_host_boundary_cut dut (
        .clk(clk),.cold_rst_n(cold_rst_n),.boundary_rst_n(boundary_rst_n),.loss(loss),
        .in_valid(in_valid),.in_ready(in_ready),.out_valid(out_valid),.out_ready(out_ready),
        .data_aw_in(raw_request[0][81:0]),.data_aw_out(core_request[0][81:0]),
        .data_w_in(raw_request[1]),.data_w_out(core_request[1]),
        .data_b_in(core_data[2][5:0]),.data_b_out(raw_response[2][5:0]),
        .ctl_aw_in(raw_request[3][31:0]),.ctl_aw_out(core_request[3][31:0]),
        .ctl_w_in(raw_request[4][35:0]),.ctl_w_out(core_request[4][35:0]),
        .ctl_b_in(core_data[5][1:0]),.ctl_b_out(raw_response[5][1:0]),
        .ctl_ar_in(raw_request[6][31:0]),.ctl_ar_out(core_request[6][31:0]),
        .ctl_r_in(core_data[7][33:0]),.ctl_r_out(raw_response[7][33:0]),
        .invalid(invalid),.fenced(fenced),.pending(pending),.fault(fault),
        .accepted(accepted),.delivered(delivered),.abandoned(abandoned));
    `include "ntrpu_host_boundary_cdc_wiring.vh"
    function automatic integer width(input integer c);
        case(c) 0:width=82;1:width=577;2:width=6;3,6:width=32;4:width=36;5:width=2;7:width=34;endcase
    endfunction
    function automatic [576:0] value(input integer c,n);
        reg [576:0] v;reg [63:0] addr;
        begin
            v=0;addr=64'h1234000100001000+64*n;
            case(c)
                0:v={addr,4'(n),8'd0,3'd6,2'd1,1'b0};
                1:begin for(integer b=0;b<64;b=b+1) v[65+b*8+:8]=8'(n*37+b*19);v[64:1]=64'hffffffffffffffff;v[0]=1;end
                2:v={4'(n),2'(n%2==0 ? 0:2)};
                3:v=32'h00000020+4*n;
                4:v={32'h87654320+32'(n),4'hf};
                5:v=n%2==0 ? 577'd0:577'd3;
                6:v=32'h00000080+4*n;
                7:v={32'h5a1c0000+32'(n),2'(n%2==0 ? 0:3)};
            endcase
            value=v;
        end
    endfunction
    always @(posedge host_clk) begin
        hcycle=hcycle+1;
        if(scenario>=0) begin
            $display("BOUND_HOST case=%0d cycle=%0d ps=%0d rst=%b hv=%h hr=%h ready=%h valid=%h",scenario,hcycle,longint'($realtime*1000),host_rst_n,hv,hr,hready,hvalid);
            for(integer c=0;c<8;c=c+1) if(hv[c] || hvalid[c])
                $display("BOUND_HOST_DATA case=%0d cycle=%0d c=%0d request=%h response=%h",scenario,hcycle,c,hdata[c],hreturn[c]);
        end
    end
    always @(posedge clk) begin
        cycle=cycle+1;
        if(scenario>=0) begin
            $display("BOUND_CUT case=%0d cycle=%0d ps=%0d cold=%b bound=%b loss=%b invalid=%b sv=%h sr=%h mv=%h mr=%h pending=%h fenced=%h fault=%h",scenario,cycle,longint'($realtime*1000),cold_rst_n,boundary_rst_n,loss,invalid,in_valid,in_ready,out_valid,out_ready,pending,fenced,fault);
            for(integer c=0;c<8;c=c+1) if(in_valid[c] || out_valid[c])
                $display("BOUND_CUT_DATA case=%0d cycle=%0d c=%0d input=%h output=%h",scenario,cycle,c,RESP[c] ? core_data[c]:raw_request[c],RESP[c] ? raw_response[c]:core_request[c]);
        end
    end
    task automatic host_send(input integer c,n);
        integer timeout;
        begin
            @(negedge host_clk);hdata[c]=value(c,n);hv[c]=1;timeout=0;
            do begin @(posedge host_clk);timeout++;if(timeout>2000) $fatal(1,"BOUND_HOST_TIMEOUT case=%0d c=%0d",scenario,c);end while(!hready[c]);
            @(negedge host_clk);hv[c]=0;
        end
    endtask
    task automatic core_send(input integer c,n);
        integer timeout;
        begin
            @(negedge clk);core_data[c]=value(c,n);core_valid[c]=1;timeout=0;
            do begin @(posedge clk);timeout++;if(timeout>2000) $fatal(1,"BOUND_CORE_TIMEOUT case=%0d c=%0d",scenario,c);end while(!in_ready[c]);
            @(negedge clk);core_valid[c]=0;
        end
    endtask
    task automatic begin_case(input integer which);
        begin
            @(negedge clk);scenario=-1;cold_rst_n=0;boundary_rst_n=0;loss=0;core_ready=REQ;core_valid=0;response_pause=0;
            @(negedge host_clk);host_rst_n=0;hv=0;hr=RESP;
            repeat(64) @(negedge host_clk);
            @(negedge clk);scenario=which;cold_rst_n=1;boundary_rst_n=1;
            @(negedge host_clk);host_rst_n=1;
            repeat(40) @(negedge host_clk);
            $display("BOUND_BEGIN case=%0d",scenario);
        end
    endtask
    task automatic lose_and_reset(input bit stop_clock,input integer guard_cycles=5);
        begin
            if(stop_clock) begin @(negedge host_clk);host_run=0;end
            @(negedge clk);loss=1;
            repeat(guard_cycles) @(negedge clk);
            // The core invalidation acknowledgement is represented by this
            // ordered test sequence; no detector or automatic guard is claimed.
            boundary_rst_n=0;host_rst_n=0;hv=0;
            repeat(70) @(negedge clk);
            host_run=1;
            repeat(64) @(negedge host_clk);
            @(negedge clk);loss=0;
            boundary_rst_n=1;
            @(negedge host_clk);host_rst_n=1;
            repeat(40) @(negedge host_clk);
            if(!invalid) $fatal(1,"BOUND_LOSS_FORGOTTEN");
        end
    endtask
    task automatic end_case;
        begin
            repeat(20) @(negedge clk);
            if(fault!=0) $fatal(1,"BOUND_FAULT case=%0d fault=%h",scenario,fault);
            for(integer c=0;c<8;c=c+1)
                $display("BOUND_SUMMARY case=%0d c=%0d accepted=%0d delivered=%0d abandoned=%0d pending=%0d invalid=%b",scenario,c,accepted[c*64+:64],delivered[c*64+:64],abandoned[c*64+:64],pending[c*2+:2],invalid);
            $display("BOUND_CASE_PASS case=%0d",scenario);
        end
    endtask
    integer k,target;
    initial begin
        for(integer c=0;c<8;c=c+1) begin hdata[c]=0;core_data[c]=0;end
        #250;
        begin_case(0);
        for(integer n=0;n<3;n=n+1) begin
            host_send(0,n);host_send(1,n);host_send(3,n);host_send(4,n);host_send(6,n);
            repeat(30) @(negedge clk);
            core_send(2,n);core_send(5,n);core_send(7,n);
            repeat(40) @(negedge host_clk);
        end
        end_case();
        // Two captured requests remain exposed across the facing CDC reset.
        for(k=1;k<=5;k=k+1) begin
            case(k) 1:target=0;2:target=1;3:target=3;4:target=4;5:target=6;endcase
            begin_case(k);core_ready[target]=0;
            host_send(target,0);host_send(target,1);
            wait(pending[target*2+:2]==2);
            lose_and_reset(0);
            if(pending[target*2+:2]!=2 || !out_valid[target]) $fatal(1,"BOUND_REQUEST_ERASED");
            @(negedge clk);core_ready[target]=1;
            repeat(10) @(negedge clk);
            // A restored upstream can offer another request, but source
            // admission stays closed. Do not invent its host acknowledgement.
            @(negedge host_clk);hdata[target]=value(target,2);hv[target]=1;
            repeat(40) @(negedge host_clk);
            end_case();
        end
        // Fill the response cut, then accept a third retained response at
        // the reset fence: all three are abandoned and none delivered to CDC.
        for(k=6;k<=8;k=k+1) begin
            target=k==6 ? 2:k==7 ? 5:7;
            begin_case(k);response_pause[target]=1;
            for(integer n=0;n<3;n=n+1) begin
                if(target==2) begin host_send(0,n);host_send(1,n);end
                if(target==5) begin host_send(3,n);host_send(4,n);end
                if(target==7) host_send(6,n);
            end
            repeat(30) @(negedge clk);
            core_send(target,0);core_send(target,1);
            fork core_send(target,2);begin repeat(5) @(negedge clk);lose_and_reset(0);end join
            if(abandoned[target*64+:64]!=3 || delivered[target*64+:64]!=0) $fatal(1,"BOUND_RESPONSE_FENCE");
            end_case();
        end
        // A response already accepted by CDC is not necessarily delivered to
        // the host. Freeze its clock and reset both sides; retain this distinction.
        for(k=9;k<=11;k=k+1) begin
            target=k==9 ? 2:k==10 ? 5:7;
            begin_case(k);hr[target]=0;
            if(target==2) begin host_send(0,0);host_send(1,0);end
            if(target==5) begin host_send(3,0);host_send(4,0);end
            if(target==7) host_send(6,0);
            repeat(30) @(negedge clk);core_send(target,0);
            wait(hvalid[target]);
            lose_and_reset(1);
            if(target==2) begin
                if(abandoned[target*64+:64]!=0 || delivered[target*64+:64]!=1) $fatal(1,"BOUND_FIFO_RECEIPT");
            end else if(abandoned[target*64+:64]!=1 || delivered[target*64+:64]!=0)
                $fatal(1,"BOUND_AXIL_RETURN_ACK");
            end_case();
        end
        // A completion arriving from the retained core after the reset fence
        // must be consumed locally and cannot reach the reopened host CDC.
        begin_case(12);host_send(0,0);host_send(1,0);
        repeat(30) @(negedge clk);lose_and_reset(0);core_send(2,0);end_case();
        // AXI-Lite destination delivery can precede the host AW/W acknowledgements.
        begin_case(13);
        @(negedge host_clk);hdata[3]=value(3,0);hdata[4]=value(4,0);hv[3]=1;hv[4]=1;
        wait(accepted[3*64+:64]==1 && accepted[4*64+:64]==1);
        lose_and_reset(1);
        end_case();
        // Conversely, a host B receipt can precede the producer acknowledgement.
        begin_case(14);host_send(3,0);host_send(4,0);
        repeat(30) @(negedge clk);core_send(5,0);
        wait(hvalid[5]);@(posedge host_clk);
        if(!hvalid[5] || !hr[5] || delivered[5*64+:64]!=0) $fatal(1,"BOUND_ACK_WINDOW");
        lose_and_reset(1,1);
        if(abandoned[5*64+:64]!=1 || delivered[5*64+:64]!=0) $fatal(1,"BOUND_EARLY_HOST_RECEIPT");
        end_case();
        $display("BOUND_SUITE_PASS cases=15");$finish;
    end
    initial begin #200000;$fatal(1,"BOUND_BANK_TIMEOUT");end
endmodule
