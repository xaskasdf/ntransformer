`timescale 1ns/1ps
module tb_ntrpu_xdma_completion_guard;
    reg clk=0;
    always #2 clk=~clk;
    reg cold_rst_n=0,core_ready=1,exclusive_owner=1,external_fault=0,wait_expired=0,reset_request=0;
    reg start_valid=0;
    reg [63:0] start_epoch=0;
    reg [1:0] native_run=3,descriptor_load=0,native_done=0;
    wire start_ready,active;
    wire [1:0] admit_descriptor,validated_done,dma_error;
    wire [15:0] faults;
    wire [63:0] published_count,observed_count,delivered_count,control_snapshot,status_snapshot,count_snapshot;
    wire m_admit,m_req_valid,m_req_ready,m_req_write,m_rsp_valid,m_rsp_ready,m_rsp_timeout,m_timeout_latched;
    wire [31:0] m_req_address,m_req_data,m_rsp_data;
    wire [3:0] m_req_strobe;
    wire [127:0] m_req_identity,m_rsp_identity,master_identity;
    wire [1:0] m_rsp_code;
    reg corrupt_identity=0;
    assign m_rsp_identity=master_identity ^ (corrupt_identity ? 128'd1 : 128'd0);
    ntrpu_xdma_completion_guard #(.WAIT_LIMIT(1000)) dut(.*);

    wire [31:0] awaddr,wdata,araddr;
    wire [2:0] awprot,arprot;
    wire [3:0] wstrb;
    wire awvalid,wvalid,bready,arvalid,rready,busy;
    reg arready=0,rvalid=0;
    reg [31:0] rdata=0;
    reg [1:0] rresp=0;
    ntrpu_xdma_local_master #(.WAIT_LIMIT(40)) master(
        .clk(clk),.cold_rst_n(cold_rst_n),.admit(m_admit),
        .req_valid(m_req_valid),.req_ready(m_req_ready),.req_write(m_req_write),
        .req_address(m_req_address),.req_data(m_req_data),.req_strobe(m_req_strobe),.req_identity(m_req_identity),
        .rsp_valid(m_rsp_valid),.rsp_ready(m_rsp_ready),.rsp_data(m_rsp_data),.rsp_code(m_rsp_code),
        .rsp_identity(master_identity),.rsp_timeout(m_rsp_timeout),.timeout_latched(m_timeout_latched),.busy(busy),
        .m_axil_awaddr(awaddr),.m_axil_awprot(awprot),.m_axil_awvalid(awvalid),.m_axil_awready(1'b0),
        .m_axil_wdata(wdata),.m_axil_wstrb(wstrb),.m_axil_wvalid(wvalid),.m_axil_wready(1'b0),
        .m_axil_bresp(2'b00),.m_axil_bvalid(1'b0),.m_axil_bready(bready),
        .m_axil_araddr(araddr),.m_axil_arprot(arprot),.m_axil_arvalid(arvalid),.m_axil_arready(arready),
        .m_axil_rdata(rdata),.m_axil_rresp(rresp),.m_axil_rvalid(rvalid),.m_axil_rready(rready));

    integer phase=0,seed=0,ticks=0,cases=0,credits0=0,credits1=0,read_count=0;
    reg [31:0] controls[0:1],statuses[0:1],counts[0:1];
    reg stall_ar=0,stall_r=0,pending=0;
    reg [31:0] address_q=0;
    reg [1:0] response_code=0;
    integer delay_q=0;
    // Behavioral register target. Independent AR/R backpressure; no fake B.
    always @(negedge clk) begin
        if(!cold_rst_n) begin
            arready=0;rvalid=0;pending=0;delay_q=0;ticks=0;
        end else begin
            ticks=ticks+1;
            arready=!stall_ar && !pending && !rvalid && (ticks+seed)%4!=0;
            if(pending && !rvalid) begin
                if(delay_q>0) delay_q=delay_q-1;
                else if(!stall_r) begin
                    case(address_q[7:0])
                        8'h04:rdata=controls[address_q[8]];
                        8'h40:rdata=statuses[address_q[8]];
                        8'h48:rdata=counts[address_q[8]];
                        default:$fatal(1,"Unexpected register address %h",address_q);
                    endcase
                    rresp=response_code;rvalid=1;
                end
            end
        end
    end
    always @(posedge clk) if(cold_rst_n) begin
        if(awvalid || wvalid || m_req_write || m_req_strobe!=0) $fatal(1,"Guard must never write");
        if(arvalid && arready) begin
            if(pending) $fatal(1,"Second outstanding AR");
            address_q<=araddr;pending<=1;delay_q<=1+(ticks+seed)%5;read_count=read_count+1;
        end
        if(rvalid && rready) begin pending<=0;rvalid<=0;end
        if(validated_done[0]) credits0=credits0+1;
        if(validated_done[1]) credits1=credits1+1;
        if(|validated_done && (faults || !exclusive_owner || !core_ready || reset_request || wait_expired))
            $fatal(1,"Credit with revoked context");
    end
    task cycles(input integer n); repeat(n) @(negedge clk); endtask
    task require(input bit ok,input string message);
        if(!ok) $fatal(1,"GUARD_ASSERT phase=%0d %s",phase,message);
    endtask
    task boot(input integer n);
        begin
            @(negedge clk);cold_rst_n=0;phase=n;start_valid=0;
            core_ready=1;exclusive_owner=1;external_fault=0;wait_expired=0;reset_request=0;
            native_run=3;descriptor_load=0;native_done=0;corrupt_identity=0;
            stall_ar=0;stall_r=0;response_code=0;credits0=0;credits1=0;read_count=0;
            for(integer c=0;c<2;c=c+1) begin controls[c]='h00f83e7f;statuses[c]=1;counts[c]=0;end
            cycles(3);cold_rst_n=1;start_epoch=n;cycles(1);
            require(start_ready,"configuration admission");start_valid=1;cycles(1);start_valid=0;
        end
    endtask
    task armed;
        begin
            while(admit_descriptor!=3 && !faults) cycles(1);
            require(admit_descriptor==3 && !faults && read_count==8,"eight initial readbacks before admission");
        end
    endtask
    task submit_two;
        begin
            require(admit_descriptor==3,"first descriptor admitted");descriptor_load=3;cycles(1);
            require(admit_descriptor==3,"second descriptor admitted");cycles(1);descriptor_load=0;
            require(admit_descriptor==0,"two-credit bound");
        end
    endtask
    task done_two;
        begin native_done=3;cycles(2);native_done=0;end
    endtask
    task success(input integer n);
        begin
            while(credits0<n || credits1<n) begin require(!faults,"success must remain clean");cycles(1);end
            cycles(4);require(credits0==n && credits1==n && !faults,"exact credit count");
        end
    endtask
    task failure(input integer bit_index);
        reg [63:0] held_published,held_observed,held_delivered;
        begin
            while(!faults) cycles(1);
            require(faults[bit_index],"expected fault category");
            held_published=published_count;held_observed=observed_count;held_delivered=delivered_count;
            // Restore all reversible context inputs: sticky fault must persist.
            core_ready=1;exclusive_owner=1;external_fault=0;wait_expired=0;reset_request=0;native_run=3;
            cycles(6);
            require(faults[bit_index] && admit_descriptor==0 && validated_done==0 && dma_error==3,"sticky fail closed");
            require(published_count==held_published && observed_count==held_observed && delivered_count==held_delivered,
                    "fault retains inventory; does not invent retirement");
        end
    endtask
    task finish_case;
        begin cases=cases+1;$display("GUARD_CASE phase=%0d faults=%h credits0=%0d credits1=%0d reads=%0d",phase,faults,credits0,credits1,read_count);end
    endtask
    integer j;
    initial begin
        if(!$value$plusargs("seed=%d",seed)) seed=0;
        $dumpfile("guard.vcd");$dumpvars(0,tb_ntrpu_xdma_completion_guard);
        boot(1);armed;submit_two;counts[0]=2;counts[1]=2;done_two;success(2);finish_case;
        // Counter visibility can lag a Done pulse: retry, never early credit.
        boot(2);armed;submit_two;done_two;cycles(160);
        require(!faults && credits0==0 && credits1==0,"wait for counter visibility");
        counts[0]=2;counts[1]=2;success(2);finish_case;
        for(j=3;j<=4;j=j+1) begin
            boot(j);armed;submit_two;counts[0]=2;counts[1]=2;
            stall_ar=j==3;stall_r=j==4;done_two;failure(3);
            require(busy && !m_rsp_valid,"timeout cannot fabricate a response");
            if(j==3) require(arvalid,"AR retained after timeout");
            else require(rready,"late R remains accepted");
            stall_ar=0;stall_r=0;cycles(30);
            require(!busy && credits0==0 && credits1==0,"late response consumed without credits");finish_case;
        end
        for(j=5;j<=6;j=j+1) begin
            boot(j);armed;submit_two;counts[0]=2;counts[1]=2;response_code=j-3;done_two;failure(9);finish_case;
        end
        boot(7);armed;submit_two;counts[0]=2;counts[1]=2;corrupt_identity=1;done_two;failure(10);finish_case;
        boot(8);armed;submit_two;controls[1]=controls[1]&~32'h80000;done_two;failure(11);finish_case;
        boot(9);armed;submit_two;controls[0]=controls[0]|32'h08000000;done_two;failure(11);finish_case;
        // Every bit outside the allowed Busy/Completed set is rejected.
        for(j=0;j<32;j=j+1) if(j!=0 && j!=2) begin
            boot(10+j);armed;submit_two;counts[0]=2;counts[1]=2;
            statuses[j%2]=32'd1<<j;done_two;failure(12);finish_case;
        end
        boot(42);armed;submit_two;counts[0]=3;done_two;failure(13);finish_case;
        boot(43);armed;submit_two;counts[0]=2;counts[1]=2;done_two;success(2);
        while(admit_descriptor!=3) cycles(1);
        submit_two;counts[0]=1;done_two;failure(13);finish_case;
        boot(44);armed;native_done=1;cycles(1);native_done=0;failure(7);finish_case;
        boot(45);armed;submit_two;descriptor_load=1;cycles(1);descriptor_load=0;failure(5);finish_case;
        for(j=46;j<=51;j=j+1) begin
            boot(j);armed;submit_two;
            case(j)
                46:core_ready=0;
                47:exclusive_owner=0;
                48:reset_request=1;
                49:wait_expired=1;
                50:external_fault=1;
                51:native_run=2;
            endcase
            failure(j==46 || j==51 ? 0 : j==47 ? 1 : j==49 ? 3 : 2);finish_case;
        end
        boot(52);armed;submit_two;counts[0]=2;counts[1]=2;done_two;
        while(validated_done==0) cycles(1);
        exclusive_owner=0;#0.1;require(validated_done==0,"same-cycle revoke suppresses credit");failure(1);finish_case;
        boot(53);counts[0]=1;failure(13);finish_case;
        boot(54);armed;submit_two;controls[0]=controls[0]&~32'd1;done_two;failure(11);finish_case;
        boot(55);armed;submit_two;failure(3);finish_case;
        // Ownership lost while AR is stalled: must not cancel the bus offer.
        boot(56);armed;submit_two;stall_ar=1;done_two;
        while(!arvalid) cycles(1);
        exclusive_owner=0;failure(1);require(arvalid,"revocation retained AR");
        stall_ar=0;cycles(30);require(!busy,"real late response completed");finish_case;
        $display("GUARD_PASS cases=%0d seed=%0d",cases,seed);$finish;
    end
    initial begin #2000000;$fatal(1,"guard watchdog");end
endmodule
