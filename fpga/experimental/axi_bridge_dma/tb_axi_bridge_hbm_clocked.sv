`timescale 1ns/1ps
`default_nettype none
// Control/queue contract test. Memory data are explicit fixtures, not HBM IP.
module tb_axi_bridge_hbm_clocked;
    integer test_id=0,seed=0;
    reg p_clk=0,r_clk=0,mgmt_clk=0,mem_clk=0;
    reg p_run=1,r_run=1,mem_run=1,cold=1,native_ok=0;
    reg [1:0] init_complete=0,adapter_fault=0;
    initial forever begin #2;if(p_run)p_clk=~p_clk;end
    initial begin #0.7;forever begin #2.5;if(r_run)r_clk=~r_clk;end end
    initial begin #0.3;forever #5 mgmt_clk=~mgmt_clk;end
    initial begin #0.1;forever begin #1.111;if(mem_run)mem_clk=~mem_clk;end end
    wire p_reset,m_reset,r_reset,mem_reset,pcie_link,pcie_stop,guard_fault;
    wire ref_reset,ref_link,ref_stop,ref_alive,ref_armed,ref_fault;
    wire [1:0] guard_code,ref_code;
    wire guard_alive,guard_armed;
    ntrpu_axi_bridge_guard_v3_reset_export #(.SHARED_NATIVE_RESET(1)) guard(
        .mgmt_clk(mgmt_clk),.axi_clk(p_clk),.cold_rst_n(cold),
        .native_link_up(native_ok),.native_axi_reset_n(native_ok),.native_ctl_reset_n(native_ok),.perst_n(native_ok),
        .client_cold_rst_n(p_reset),.management_cold_rst_n(m_reset),.client_link_up(pcie_link),.client_stop(pcie_stop),
        .clock_alive(guard_alive),.armed(guard_armed),.fault(guard_fault),.fault_code(guard_code));
    ntrpu_axi_bridge_guard_v3 #(.SHARED_NATIVE_RESET(1)) reference(
        .mgmt_clk(mgmt_clk),.axi_clk(p_clk),.cold_rst_n(cold),
        .native_link_up(native_ok),.native_axi_reset_n(native_ok),.native_ctl_reset_n(native_ok),.perst_n(native_ok),
        .client_cold_rst_n(ref_reset),.client_link_up(ref_link),.client_stop(ref_stop),
        .clock_alive(ref_alive),.armed(ref_armed),.fault(ref_fault),.fault_code(ref_code));
    wire startup_started,startup_fault;
    ntrpu_hbm_startup startup(.clk(r_clk),.mem_clk(mem_clk),.cold_reset_n(cold),
        .init_complete_async(init_complete),.adapter_fault_mem(adapter_fault),
        .src_rst_n(r_reset),.mem_rst_n(mem_reset),.started(startup_started),.fault(startup_fault));
    wire mem_alive,mem_armed,mem_fault,mem_permit_p,mem_fault_p,mem_fault_r;
    ntrpu_axi_bridge_hbm_mem_watch memory_watch(
        .mgmt_clk(mgmt_clk),.mem_clk(mem_clk),.p_clk(p_clk),.r_clk(r_clk),
        .m_cold_rst_n(m_reset),.mem_cold_rst_n(mem_reset),.p_cold_rst_n(p_reset),.r_cold_rst_n(r_reset),
        .clock_alive(mem_alive),.armed(mem_armed),.fault(mem_fault),
        .p_permit(mem_permit_p),.p_fault(mem_fault_p),.r_fault(mem_fault_r));
    wire cdc_started,cdc_fault,cdc_stop,reader_alive,reader_armed,reader_fault;
    wire effective_permit=pcie_link&&cdc_started&&mem_permit_p&&!cdc_fault&&!mem_fault_p;
    wire effective_fault=pcie_stop||cdc_fault||mem_fault_p;
    reg [1:0] p_cmd_valid=0,r_data_valid=0;
    wire [1:0] p_cmd_ready,r_cmd_valid,r_data_ready,p_data_valid;
    wire [127:0] r_cmd_addr,r_cmd_tag;
    wire [15:0] r_cmd_len;
    wire [1023:0] p_data;
    localparam [511:0] WORD={16{32'h1234abcd}};
    ntrpu_axi_bridge_hbm_cdc transport(
        .p_clk(p_clk),.r_clk(r_clk),.mgmt_clk(mgmt_clk),
        .p_cold_rst_n(p_reset),.r_cold_rst_n(r_reset),.m_cold_rst_n(m_reset),
        .p_stop(pcie_stop||mem_fault_p),.r_started(startup_started),.r_fault(startup_fault||mem_fault_r),
        .p_started(cdc_started),.p_fault(cdc_fault),.r_stop(cdc_stop),
        .clock_alive(reader_alive),.armed(reader_armed),.management_fault(reader_fault),
        .p_cmd_valid(p_cmd_valid),.p_cmd_ready(p_cmd_ready),.p_cmd_addr(128'h200001000_00001000),
        .p_cmd_tag(128'hface_beef),.p_cmd_len(16'h0101),
        .r_cmd_valid(r_cmd_valid),.r_cmd_ready(2'b00),.r_cmd_addr(r_cmd_addr),.r_cmd_tag(r_cmd_tag),.r_cmd_len(r_cmd_len),
        .r_data_valid(r_data_valid),.r_data_ready(r_data_ready),.r_data( {WORD,WORD}),.r_data_last(2'b11),.r_data_tag(128'hface_beef),
        .r_cpl_valid(2'b00),.r_cpl_ready(),.r_cpl_tag(128'b0),.r_cpl_status(4'b0),.r_cpl_bytes(26'b0),
        .p_data_valid(p_data_valid),.p_data_ready(2'b00),.p_data(p_data),.p_data_last(),.p_data_tag(),
        .p_cpl_valid(),.p_cpl_ready(2'b00),.p_cpl_tag(),.p_cpl_status(),.p_cpl_bytes());
    integer comparisons=0,p_release_edges=0,r_release_edges=0,m_release_edges=0,mem_release_edges=0;
    task require(input bit ok,input string why);
        if(ok!==1'b1)$fatal(1,"HC CHECK case=%0d seed=%0d %s",test_id,seed,why);
    endtask
    always @(posedge p_clk or negedge cold)if(!cold)p_release_edges=0;else p_release_edges++;
    always @(posedge r_clk or negedge cold)if(!cold)r_release_edges=0;else r_release_edges++;
    always @(posedge mgmt_clk or negedge cold)if(!cold)m_release_edges=0;else m_release_edges++;
    always @(posedge mem_clk or negedge cold)if(!cold)mem_release_edges=0;else mem_release_edges++;
    always @(posedge p_reset)if($time>1)require(p_release_edges>=3,"PCIe release after three local edges");
    always @(posedge r_reset)if($time>1)require(r_release_edges>=3,"reader release after three local edges");
    always @(posedge m_reset)if($time>1)require(m_release_edges>=3,"management release after three local edges");
    always @(posedge mem_reset)if($time>1)require(mem_release_edges>=3,"memory release after three local edges");
    always @(negedge mgmt_clk)begin
        #0.001;
        require({p_reset,pcie_link,pcie_stop,guard_alive,guard_armed,guard_fault,guard_code}===
                {ref_reset,ref_link,ref_stop,ref_alive,ref_armed,ref_fault,ref_code},"exported guard matches frozen V3");
        require(m_reset===reference.cold_m[2],"management output aliases existing release");
        comparisons++;
    end
    task reset_epoch;
        cold=0;#0.01;
        require({p_reset,m_reset,r_reset,mem_reset}===4'b0000,"asynchronous cold assertion in all domains");
        #40;cold=1;
    endtask
    task fill_queues;
        @(negedge p_clk);p_cmd_valid=3;
        @(posedge p_clk);require(p_cmd_ready==3,"both command queues ready");
        @(negedge p_clk);p_cmd_valid=0;
        @(negedge r_clk);r_data_valid=3;
        @(posedge r_clk);require(r_data_ready==3,"both return queues ready");
        @(negedge r_clk);r_data_valid=0;
        wait(r_cmd_valid==3 && p_data_valid==3);#0.1;
    endtask
    task retained;
        require({p_reset,m_reset,r_reset,mem_reset}==4'b1111,"operational fault does not clear cold resets");
        require(r_cmd_valid==3 && p_data_valid==3,"queued offers retained");
        require(p_data=={WORD,WORD} && r_cmd_addr==128'h200001000_00001000 &&
                r_cmd_tag==128'hface_beef && r_cmd_len==16'h0101,"queued payload unchanged");
    endtask
    realtime stopped_at;
    initial begin
        if(!$value$plusargs("case=%d",test_id))test_id=0;
        if(!$value$plusargs("seed=%d",seed))seed=0;
        #0.2;cold=0;
        if(test_id==10)mem_run=0;
        #(41.3+seed*3.7);native_ok=1;init_complete=3;cold=1;
        if(test_id==10)begin
            #1400;require(!effective_permit&&!mem_reset&&!mem_armed,"absent memory clock cannot arm epoch");
            mem_run=1;wait(effective_permit);#100;
            $display("HC ABSENT_RECOVERED case=%0d",test_id);
        end else wait(effective_permit);
        require(!effective_fault&&!startup_fault&&!mem_fault,"healthy startup");
        fill_queues();
        #(37.1+seed*2.3);
        case(test_id)
            1,2,11:begin wait(mem_clk==(test_id==2));#0.01;mem_run=0;end
            3,4:begin wait(r_clk==(test_id==4));#0.01;r_run=0;end
            5,6:begin wait(p_clk==(test_id==6));#0.01;p_run=0;end
            7:native_ok=0;
            8:adapter_fault=1;
            9:init_complete=1;
            12:begin wait(mem_clk==0);#0.01;mem_run=0;wait(r_clk==1);#0.01;r_run=0;end
            default:begin end
        endcase
        stopped_at=$realtime;
        if(test_id!=0 && test_id!=10)begin
            if(test_id==12)wait(mem_fault&&reader_fault&&effective_fault);
            else if(test_id==5 || test_id==6)wait(guard_fault);
            else wait(effective_fault);
            require($realtime-stopped_at<900,"clock/status loss reaches guard within 900 ns");
            #1;require(!effective_permit,"new admission closed");retained();
            $display("HC FAULT case=%0d latency_ns=%0.3f mem=%b reader=%b pcie=%b",test_id,$realtime-stopped_at,mem_fault,reader_fault,guard_fault);
            if(test_id==11)begin
                reset_epoch();#200;require(!mem_reset&&!effective_permit,"cold release waits for halted memory clock");
                mem_run=1;
            end else begin
                p_run=1;r_run=1;mem_run=1;native_ok=1;adapter_fault=0;init_complete=3;
                #1000;require(!effective_permit&&effective_fault,"resume cannot recover failed epoch");retained();
                reset_epoch();
            end
        end else begin
            #1400;require(effective_permit&&!effective_fault,"continuous clocks remain healthy");retained();reset_epoch();
        end
        p_run=1;r_run=1;mem_run=1;native_ok=1;adapter_fault=0;init_complete=3;
        wait(effective_permit);#100;
        require(!effective_fault&&!mem_fault&&!startup_fault&&r_cmd_valid==0&&p_data_valid==0,"cold epoch discards old queued offers");
        fill_queues();retained();
        require(comparisons>40,"frozen guard comparison exercised");
        $display("HC PASS case=%0d seed=%0d comparisons=%0d cold_domains=4 retained_lanes=2",test_id,seed,comparisons);
        $finish;
    end
    initial begin #15000;$fatal(1,"HC TIMEOUT case=%0d",test_id);end
endmodule
`default_nettype wire
