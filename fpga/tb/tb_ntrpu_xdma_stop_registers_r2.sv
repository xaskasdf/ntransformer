`timescale 1ns/1ps
// Public-port integration with the actual retained AXI-Lite master.
// The slave is a register-response fixture, not native XDMA or a host driver.
module tb_ntrpu_xdma_stop_registers;
    reg clk=0; always #2 clk=~clk;
    reg cold_rst_n=0,core_ready=1,exclusive_owner=1,external_fault=0,wait_expired=0;
    reg start_valid=0,close_request=1,host_closed=1,ledger_active=1,ledger_closed=1;
    reg ledger_accounted=1,release_valid=0;
    reg [63:0] start_generation=41,start_tag=1,host_generation=41,host_tag=1;
    reg [63:0] ledger_generation=41,ledger_tag=1,ledger_published=0,ledger_completed=0;
    reg [8:0] ledger_faults=0;
    wire start_ready,active,snapshot_valid,transaction_pending;
    wire [63:0] generation,tag,expected_published,control_snapshot,status_snapshot,count_snapshot;
    wire [13:0] faults;
    wire [1:0] stop_responses;
    wire [5:0] read_responses;
    wire m_admit,m_req_valid,m_req_ready,m_req_write,m_rsp_valid,m_rsp_ready,m_rsp_timeout,m_timeout_latched;
    wire [31:0] m_req_address,m_req_data,m_rsp_data;
    wire [3:0] m_req_strobe;
    wire [1:0] m_rsp_code;
    wire [127:0] m_req_identity,m_rsp_identity;
    reg allow_request=1,wrong_identity=0,unsolicited=0;
    wire master_req_ready,master_rsp_valid;
    wire [127:0] master_rsp_identity;
    wire master_busy;
    assign m_req_ready=master_req_ready && allow_request;
    assign m_rsp_valid=master_rsp_valid || unsolicited;
    assign m_rsp_identity=master_rsp_identity ^ (wrong_identity?128'd1:128'd0);
    ntrpu_xdma_stop_registers dut(.*);
    wire [31:0] awaddr,wdata,araddr;
    wire [3:0] wstrb;
    wire [2:0] awprot,arprot;
    wire awvalid,wvalid,bready,arvalid,rready;
    reg awready=0,wready=0,bvalid=0,arready=0,rvalid=0;
    reg [1:0] bresp=0,rresp=0;
    reg [31:0] rdata=0;
    ntrpu_xdma_local_master #(.WAIT_LIMIT(32)) master(
        .clk(clk),.cold_rst_n(cold_rst_n),.admit(m_admit),
        .req_valid(m_req_valid && allow_request),.req_ready(master_req_ready),
        .req_write(m_req_write),.req_address(m_req_address),.req_data(m_req_data),
        .req_strobe(m_req_strobe),.req_identity(m_req_identity),
        .rsp_valid(master_rsp_valid),.rsp_ready(m_rsp_ready),.rsp_data(m_rsp_data),
        .rsp_code(m_rsp_code),.rsp_identity(master_rsp_identity),.rsp_timeout(m_rsp_timeout),
        .timeout_latched(m_timeout_latched),.busy(master_busy),
        .m_axil_awaddr(awaddr),.m_axil_awprot(awprot),.m_axil_awvalid(awvalid),.m_axil_awready(awready),
        .m_axil_wdata(wdata),.m_axil_wstrb(wstrb),.m_axil_wvalid(wvalid),.m_axil_wready(wready),
        .m_axil_bresp(bresp),.m_axil_bvalid(bvalid),.m_axil_bready(bready),
        .m_axil_araddr(araddr),.m_axil_arprot(arprot),.m_axil_arvalid(arvalid),.m_axil_arready(arready),
        .m_axil_rdata(rdata),.m_axil_rresp(rresp),.m_axil_rvalid(rvalid),.m_axil_rready(rready));

    reg hold_b=0,hold_r=0;
    reg [1:0] h_bcode=0,c_bcode=0,read_code=0;
    reg [31:0] h_control=0,c_control=0,h_status=0,c_status=0,h_count=0,c_count=0;
    integer phase=0,cycle=0,write_count=0,read_count=0,case_count=0;
    reg seen_aw=0,seen_w=0,read_pending=0;
    reg [31:0] saved_aw=0,saved_ar=0;
    integer read_delay=0;
    function [31:0] fixture_value(input [31:0] address);
        case(address)
            'h4:fixture_value=h_control;
            'h40:fixture_value=h_status;
            'h48:fixture_value=h_count;
            'h1004:fixture_value=c_control;
            'h1040:fixture_value=c_status;
            'h1048:fixture_value=c_count;
            default:begin fixture_value=32'hdeadbeef;$fatal(1,"Unexpected CSR read %h",address);end
        endcase
    endfunction
    // AW and W can arrive in either order. All response beats are retained.
    always @(negedge clk) begin
        awready=cold_rst_n && !seen_aw && cycle%3!=0;
        wready=cold_rst_n && !seen_w && cycle%4!=0;
        arready=cold_rst_n && !read_pending && !rvalid && cycle%3!=1;
    end
    always @(posedge clk) begin
        if(!cold_rst_n) begin
            cycle<=0;seen_aw<=0;seen_w<=0;bvalid<=0;rvalid<=0;read_pending<=0;
            write_count<=0;read_count<=0;read_delay<=0;bresp<=0;rresp<=0;rdata<=0;
        end else begin
            cycle<=cycle+1;
            if(awvalid && awready) begin
                if(awaddr !== 32'h000c && awaddr !== 32'h100c) $fatal(1,"Unexpected W1C address");
                seen_aw<=1;saved_aw<=awaddr;
            end
            if(wvalid && wready) begin
                if(wdata !== 32'd1 || wstrb !== 4'hf) $fatal(1,"Unexpected W1C payload");
                seen_w<=1;
            end
            if(seen_aw && seen_w && !bvalid && !hold_b) begin
                bvalid<=1;bresp<=(saved_aw == 32'h000c) ? h_bcode : c_bcode;
            end
            if(bvalid && bready) begin
                bvalid<=0;seen_aw<=0;seen_w<=0;write_count<=write_count+1;
            end
            if(arvalid && arready) begin
                if(write_count<2) $fatal(1,"Read before both real B responses");
                saved_ar<=araddr;read_pending<=1;read_delay<=2;
            end
            if(read_pending && !hold_r) begin
                if(read_delay!=0) read_delay<=read_delay-1;
                else begin rvalid<=1;rdata<=fixture_value(saved_ar);rresp<=read_code;read_pending<=0;end
            end
            if(rvalid && rready) begin rvalid<=0;read_count<=read_count+1;end
        end
    end
    always @(posedge clk) if(cold_rst_n) begin
        if($isunknown({active,snapshot_valid,faults,transaction_pending,m_req_valid,m_rsp_ready}))
            $fatal(1,"Unknown control output");
        if(m_req_valid && $isunknown({m_req_write,m_req_address,m_req_data,m_req_strobe,m_req_identity}))
            $fatal(1,"Unknown offered master request");
        if(awvalid && $isunknown({awaddr,awprot})) $fatal(1,"Unknown AW");
        if(wvalid && $isunknown({wdata,wstrb})) $fatal(1,"Unknown W");
        if(arvalid && $isunknown({araddr,arprot})) $fatal(1,"Unknown AR");
    end
    task tick; begin @(negedge clk);#0.1;end endtask
    task ticks(input integer n);begin repeat(n) tick();end endtask
    task reset_case(input integer p);
        begin
            tick();cold_rst_n=0;phase=p;start_valid=0;release_valid=0;
            core_ready=1;exclusive_owner=1;external_fault=0;wait_expired=0;
            close_request=1;host_closed=1;ledger_active=1;ledger_closed=1;ledger_accounted=1;
            start_generation=41;start_tag=p+1;host_generation=41;host_tag=p+1;
            ledger_generation=41;ledger_tag=p+1;ledger_published=0;ledger_completed=0;ledger_faults=0;
            allow_request=1;wrong_identity=0;unsolicited=0;hold_b=0;hold_r=0;
            h_bcode=0;c_bcode=0;read_code=0;h_control=0;c_control=0;
            h_status=0;c_status=0;h_count=0;c_count=0;
            ticks(3);cold_rst_n=1;tick();
        end
    endtask
    task start;
        begin
            #0.01;
            if(!start_ready) $fatal(1,"Not ready to start phase %0d",phase);
            start_valid=1;tick();start_valid=0;
            if(!active) $fatal(1,"Start not accepted");
        end
    endtask
    task await_snapshot;
        integer n;begin
            n=0;while(!snapshot_valid && n<500) begin tick();n=n+1;end
            if(!snapshot_valid || faults || stop_responses!=3 || read_responses!=63)
                $fatal(1,"Missing snapshot phase %0d faults=%h",phase,faults);
        end
    endtask
    task await_fault(input [13:0] expected);
        integer n;begin
            n=0;while(faults==0 && n<500) begin tick();n=n+1;end
            ticks(40);
            if(faults !== expected || snapshot_valid) $fatal(1,"Wrong fault phase %0d expected=%h actual=%h",phase,expected,faults);
        end
    endtask
    task checkpoint(input [13:0] expected,input bit qualified);
        begin
            if(faults !== expected || snapshot_valid !== qualified) $fatal(1,"Checkpoint mismatch phase %0d",phase);
            $display("STOP_REG_CASE phase=%0d t=%0t faults=%h valid=%b active=%b writes=%0d reads=%0d pending=%b",phase,$realtime,faults,snapshot_valid,active,write_count,read_count,transaction_pending);
            case_count=case_count+1;
        end
    endtask
    task good_case(input integer p);begin reset_case(p);start();await_snapshot();end endtask
    integer i,ch;reg [31:0] mask,bit_value;reg [13:0] expected_fault;
    initial begin
        #121;
        reset_case(1);ledger_published=64'h0000000100000002;ledger_completed=0;ledger_accounted=0;
        h_control=1;c_control=1;h_status=1;c_status=1;start();
        while(read_count<12) tick();
        if(snapshot_valid || write_count!=2) $fatal(1,"Live channels qualified");
        h_control=0;c_control=0;h_count=1;
        while(read_count<18) tick();
        if(snapshot_valid) $fatal(1,"Outstanding descriptor cancelled");
        h_status='h46;c_status='h46;h_count=2;c_count=1;
        ledger_completed=ledger_published;ledger_accounted=1;await_snapshot();checkpoint(0,1);
        release_valid=1;tick();release_valid=0;if(active) $fatal(1,"Release rejected");
        // A second fresh session without cold reset must work.
        phase=2;start_tag=3;host_tag=3;ledger_tag=3;ledger_published=0;ledger_completed=0;
        h_count=0;c_count=0;h_status=0;c_status=0;start();await_snapshot();checkpoint(0,1);
        reset_case(3);h_bcode=2;start();await_fault(14'h20);
        if(write_count!=2 || read_count) $fatal(1,"Both stops required despite first B error");checkpoint('h20,0);
        reset_case(4);c_bcode=3;start();await_fault('h20);checkpoint('h20,0);
        reset_case(5);h_count=1;start();await_fault('h200);checkpoint('h200,0);
        reset_case(6);ledger_published=2;ledger_accounted=0;h_count=1;start();
        while(read_count<6) tick();h_count=0;await_fault('h200);checkpoint('h200,0);
        reset_case(7);wrong_identity=1;start();await_fault('h40);checkpoint('h40,0);
        reset_case(8);read_code=2;start();await_fault('h20);checkpoint('h20,0);
        reset_case(9);hold_b=1;start();while(!m_timeout_latched) tick();
        if(write_count || m_rsp_valid || !transaction_pending) $fatal(1,"Fabricated timeout response");
        hold_b=0;await_fault('h10);
        if(write_count!=1 || !m_req_valid || m_req_address!='h100c || m_req_ready) $fatal(1,"Second stop lost after timeout");
        checkpoint('h10,0);
        reset_case(10);hold_r=1;start();while(!m_timeout_latched) tick();
        if(read_count || !transaction_pending) $fatal(1,"Fabricated R after timeout");
        hold_r=0;await_fault('h10);checkpoint('h10,0);
        reset_case(11);allow_request=0;start();core_ready=0;ticks(12);
        if(!m_req_valid || m_req_address!='hc || m_req_identity!={generation,tag}) $fatal(1,"Revocation retracted request");
        allow_request=1;await_fault(1);checkpoint(1,0);
        good_case(12);host_closed=0;#0.01;if(snapshot_valid) $fatal(1,"Delayed host revoke");tick();host_closed=1;await_fault(4);checkpoint(4,0);
        good_case(13);ledger_completed=1;#0.01;if(snapshot_valid) $fatal(1,"Delayed counter revoke");tick();ledger_completed=0;await_fault('h2000);checkpoint('h2000,0);
        reset_case(14);allow_request=0;start();release_valid=1;tick();release_valid=0;allow_request=1;await_fault('h800);checkpoint('h800,0);
        good_case(15);release_valid=1;tick();release_valid=0;start_valid=1;tick();start_valid=0;await_fault(2);checkpoint(2,0);
        reset_case(16);allow_request=0;start();ledger_published=1;ticks(12);allow_request=1;await_fault(8);checkpoint(8,0);
        good_case(17);core_ready=0;tick();core_ready=1;await_fault(1);checkpoint(1,0);
        reset_case(18);allow_request=0;start();wait_expired=1;ticks(12);allow_request=1;await_fault('h400);checkpoint('h400,0);
        reset_case(19);allow_request=0;start();ledger_tag=99;ticks(12);allow_request=1;await_fault(2);checkpoint(2,0);
        reset_case(20);allow_request=0;start();ledger_faults=1;ticks(12);allow_request=1;await_fault(8);checkpoint(8,0);
        good_case(21);unsolicited=1;#0.01;if(snapshot_valid) $fatal(1,"Unexpected response qualified");tick();unsolicited=0;await_fault('h1000);checkpoint('h1000,0);
        good_case(22);start_valid=1;tick();start_valid=0;await_fault('h1000);checkpoint('h1000,0);
        good_case(23);ledger_accounted=0;tick();ledger_accounted=1;await_fault('h2000);checkpoint('h2000,0);
        good_case(24);external_fault=1;tick();external_fault=0;await_fault(1);checkpoint(1,0);
        good_case(25);exclusive_owner=0;tick();exclusive_owner=1;await_fault(1);checkpoint(1,0);
        // Exercise every CSR bit independently for both different encodings.
        for(ch=0;ch<2;ch=ch+1) begin
            mask=ch==0?32'h00f8fe38:32'h00f80638;
            for(i=0;i<32;i=i+1) begin
                reset_case(100+ch*100+i);bit_value=32'd1<<i;
                if(ch==0) h_status=bit_value;else c_status=bit_value;
                start();
                if(i==0) begin
                    while(read_count<12) tick();
                    if(snapshot_valid) $fatal(1,"Busy qualified");
                    wait_expired=1;expected_fault='h400;await_fault(expected_fault);checkpoint(expected_fault,0);
                end else if(i==1 || i==2 || i==6) begin await_snapshot();checkpoint(0,1);end
                else begin
                    expected_fault=(mask & bit_value)!=0?'h80:'h100;
                    await_fault(expected_fault);checkpoint(expected_fault,0);
                end
            end
        end
        $display("STOP_REG_PASS cases=%0d master_wait_limit=32 native_xdma=0 global_drain=0 reset_permission=0",case_count);
        $finish;
    end
    initial begin #200000;$fatal(1,"Stop-register test watchdog");end
endmodule
