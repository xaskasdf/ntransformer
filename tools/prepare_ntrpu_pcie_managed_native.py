"""Derive an isolated native XDMA bench from the qualified cold host fixture."""
from pathlib import Path
import hashlib,json,re


def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def bench(text):
    text=once(text,'module tb_ntrpu_pcie_managed_host_cold_r2;','module ntrpu_pcie_managed_native_harness;')
    text=once(text,'// Actual two MIGs and generic DDR4 model; MMIO/F01 host is a protocol driver.',
        '// Native XDMA BAR0/BAR2, two host CDCs and two generic MIG models.\n// External isolation and direct UI readback remain trusted test drivers.')
    # Replace only host-side driver declarations. The DUT and all memory
    # fixtures, observers, patterns and destination addresses stay unchanged.
    start=text.index('    reg [63:0] s_axi_awaddr=0;')
    end=text.index('    wire line_valid;',start)
    data_widths=dict(awaddr=64,awid=4,awlen=8,awsize=3,awburst=2,awlock=1,awvalid=1,
        awready=1,wdata=512,wstrb=64,wlast=1,wvalid=1,wready=1,bid=4,bresp=2,bvalid=1,bready=1)
    outputs={'awready','wready','bid','bresp','bvalid'}
    lines=[]
    for name,width in data_widths.items():
        lines.append('    wire '+('' if width==1 else f'[{width-1}:0] ')+f's_axi_{name}'+
            (';' if name in outputs else f'=board.EP.m_axib_{name};'))
    for name in sorted(outputs):lines.append(f'    assign board.EP.m_axib_{name}=s_axi_{name};')
    for name,value in dict(arready="1'b0",rid="4'b0",rdata="512'b0",rresp="2'b11",rlast="1'b1",rvalid="1'b0").items():
        lines.append(f'    assign board.EP.m_axib_{name}={value};')
    text=text[:start]+'\n'.join(lines)+'\n'+text[end:]
    start=text.index('    reg [31:0] s_axil_awaddr=0,')
    end=text.index('    wire irq, control_fault;',start)
    control_widths=dict(awaddr=32,wdata=32,araddr=32,wstrb=4,awvalid=1,wvalid=1,bready=1,arvalid=1,rready=1,
        awready=1,wready=1,arready=1,bvalid=1,rvalid=1,bresp=2,rresp=2,rdata=32)
    outputs={'awready','wready','arready','bvalid','rvalid','bresp','rresp','rdata'}
    lines=[]
    for name,width in control_widths.items():
        lines.append('    wire '+('' if width==1 else f'[{width-1}:0] ')+f's_axil_{name}'+
            (';' if name in outputs else f'=board.EP.m_axil_{name};'))
    for name in sorted(outputs):lines.append(f'    assign board.EP.m_axil_{name}=s_axil_{name};')
    text=text[:start]+'\n'.join(lines)+'\n'+text[end:]
    text=once(text,'reg host_clk=0,host_rst_n=0,transport_rst_n=0;',
        'wire host_clk=board.EP.user_clk;\n    reg host_rst_n=0,transport_rst_n=0;\n    reg suite_started=0;\n    integer host_data_b=0,host_control_b=0,host_control_ar=0,host_control_r=0;\n    reg [1:0] last_data_b=0,last_control_b=0,last_control_r=0;\n    reg [31:0] last_control_data=0,last_read_address=0;')
    text=once(text,'    initial begin #0.731;forever #3.65 host_clk=~host_clk;end\n','')
    text=once(text,'        repeat(5) @(negedge host_clk);host_rst_n=1;',
        "        wait(board.EP.user_resetn===1'b1);\n        repeat(5) @(negedge host_clk);host_rst_n=1;")
    start=text.index('    task automatic aw;');end=text.index('    task automatic desc(',start)
    text=text[:start]+'''    // Root-port tasks generate the packets; these witnesses only wait for
    // endpoint acceptance and never drive or force AXI outputs from XDMA.
    task automatic tx(input reg [511:0] payload,input integer resp);
        integer wanted;wanted=host_data_b+1;
        board.RP.tx_usrapp.NTRPU_FRAME_WRITE(64'h9000fffc0,payload);
        repeat(4000) begin @(negedge host_clk);if(host_data_b==wanted) break;end
        need(host_data_b==wanted && last_data_b==resp,"NATIVE_FRAME_RESPONSE");
        bcount++;if(resp!=0) rejections++;
    endtask
    task automatic wr(input [31:0] address,input [31:0] value,input [3:0] strobe=15,input [1:0] expected=0);
        integer wanted;wanted=host_control_b+1;
        need(strobe==15,"NATIVE_MMIO_STROBE_PROFILE");
        board.RP.tx_usrapp.NTRPU_CTRL_WRITE(address,value);
        repeat(4000) begin @(negedge host_clk);if(host_control_b==wanted) break;end
        need(host_control_b==wanted && last_control_b==expected,"NATIVE_CONTROL_WRITE_RESPONSE");
        mmio_writes++;
        $display("NTRD case=%0d CTL_WRITE n=%0d addr=%h data=%h strb=%h resp=%h cycle=%0d",CASE_ID,mmio_writes,address,value,strobe,expected,cycle);
    endtask
    task automatic rd(input [31:0] address,input [31:0] expected,input [1:0] resp=0,input integer hold_cycles=3);
        integer wanted;wanted=host_control_r+1;
        need(resp==0,"NATIVE_READ_COMPLETION_PROFILE");
        board.RP.tx_usrapp.NTRPU_CTRL_READ(address,expected,'1);
        repeat(4000) begin @(negedge host_clk);if(host_control_r==wanted) break;end
        need(host_control_r==wanted && last_control_r==resp && last_control_data==expected,"NATIVE_CONTROL_READ_RESPONSE");
        mmio_reads++;
        $display("NTRD case=%0d CTL_READ n=%0d addr=%h data=%h resp=%h cycle=%0d",CASE_ID,mmio_reads,address,expected,resp,cycle);
    endtask
''' +text[end:]
    start=text.index('    task automatic cb_response(');end=text.index('    task automatic setup(',start)
    text=text[:start]+text[end:]
    text=once(text,'        need(ack_ready,"RETIRE_READY");wr(\'h10,4);idle(3);',
        '''        need(ack_ready,"RETIRE_READY");
        @(negedge clk);source_stopped=0;ingress_isolated=0;
        wr('h10,4,15,2);need(result_valid && bridge_lease_held,"NATIVE_REJECTED_ACK_RETAINED");
        @(negedge clk);source_stopped=1;ingress_isolated=1;
        repeat(4) @(negedge clk);need(ack_ready,"RESTORED_RETIRE_BARRIER");
        wr('h10,4);idle(8);''')
    text=once(text,'    initial begin\n        s_axi_awsize=6;s_axi_awburst=1;',
        '''    task automatic run_suite;
        need(host_rst_n && rst_n && host_control_r==1 && mmio_reads==1,"ENUMERATION_PROBE_ACCOUNTED");
        suite_started=1;''')
    text=once(text,"        rd(0,'h4452544e);rd(4,'h10000);rd(8,'hff);", "        rd(4,'h10000);rd(8,'hff);")
    text=once(text,"        rd('hc4,0,3);wr('hc4,1,15,3);", "        wr('hc4,1,15,3);")
    text=once(text,'        passed=1;$finish;\n    end', '        passed=1;\n    endtask')
    text=once(text,'#150000;$fatal', '#700000;$fatal')
    # Source tracing and memory counters remain unchanged. Host monitoring
    # accounts for the enumeration probe before the suite is called.
    at=text.index('    always @(posedge host_clk) begin')
    text=text[:at]+'''    always @(posedge host_clk) if(host_rst_n && !passed) begin
        need(board.EP.user_resetn===1'b1,"RUNTIME_PCIE_RESET_OUTSIDE_COLD_CONTRACT");
        if(s_axi_awvalid && s_axi_awready) need(s_axi_awid==0,"NATIVE_XDMA_AWID_PROFILE");
        if(s_axi_bvalid && s_axi_bready) begin
            need(s_axi_bid==0,"NATIVE_XDMA_BID_PROFILE");host_data_b++;last_data_b=s_axi_bresp;
        end
        if(s_axil_bvalid && s_axil_bready) begin host_control_b++;last_control_b=s_axil_bresp;end
        if(s_axil_arvalid && s_axil_arready) begin host_control_ar++;last_read_address=s_axil_araddr;end
        if(s_axil_rvalid && s_axil_rready) begin
            host_control_r++;last_control_r=s_axil_rresp;last_control_data=s_axil_rdata;
            if(!suite_started) begin
                need(host_control_r==1 && last_read_address==0 && s_axil_rresp==0 && s_axil_rdata==32'h4452544e,"ENUMERATION_ID_PROBE");
                mmio_reads++;
                $display("NTRD case=%0d CTL_READ n=%0d addr=%h data=%h resp=%h cycle=%0d",CASE_ID,mmio_reads,last_read_address,s_axil_rdata,s_axil_rresp,cycle);
            end
        end
    end
''' +text[at:]
    return text


def tasks(text):
    text=text[:text.index('task NTRPU_FRAME_WRITE(')]+'''task NTRPU_FRAME_WRITE(input [63:0] destination,input [511:0] payload);
    reg [63:0] address;reg [511:0] frame;reg [3:0] last_be;
    begin
        address=destination;frame=payload;last_be=4'hf;
        DEFAULT_TAG=10'h040+ntrpu_data_writes;ntrpu_data_writes++;
        $display("NTRD_NATIVE_FRAME n=%0d job=%0d addr=%h data=%h tag=%h ps=%0.0f",ntrpu_data_writes,board.ntrpu_bypass.job,address,frame,DEFAULT_TAG,$realtime/1ps);
        TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd16,address,last_be,4'hf,1'b0);
        TSK_TX_CLK_EAT(4);
    end
endtask
task NTRPU_BYPASS_RUN;
    begin
        $display("NTRPU_LINK_ENUM bar0=%h bar1=%h bar2=%h bar3=%h type0=%b type1=%b type2=%b range2=%h dma_bar=%0d ps=%0.0f",
            BAR_INIT_P_BAR[0],BAR_INIT_P_BAR[1],BAR_INIT_P_BAR[2],BAR_INIT_P_BAR[3],
            BAR_INIT_P_BAR_ENABLED[0],BAR_INIT_P_BAR_ENABLED[1],BAR_INIT_P_BAR_ENABLED[2],BAR_INIT_P_BAR_RANGE[2],xdma_bar,$realtime/1ps);
        if({BAR_INIT_P_BAR[3][31:0],BAR_INIT_P_BAR[2][31:0]}!==64'h900000000 || xdma_bar!=1 ||
           BAR_INIT_P_BAR_ENABLED[0]!==2'b10 || BAR_INIT_P_BAR_ENABLED[1]!==2'b10 || BAR_INIT_P_BAR_ENABLED[2]!==2'b11)
            $fatal(1,"NTRD_NATIVE_ENUMERATION");
        board.ntrpu_bypass.run_suite();
        if(ntrpu_control_writes!=294 || ntrpu_control_reads!=8 || ntrpu_data_writes!=35)
            $fatal(1,"NTRD_NATIVE_STIMULUS_COUNT");
        $display("NTRD_NATIVE_PASS writes=%0d reads=%0d frames=%0d ps=%0.0f",ntrpu_control_writes,ntrpu_control_reads,ntrpu_data_writes,$realtime/1ps);
    end
endtask
'''
    return text


def tcl(text,managed):
    text=once(text,'patch_ntrpu_pcie_fragment_native_host.py','patch_ntrpu_pcie_managed_native.py')
    text=once(text,'ntrpu_pcie_fragment_native_host.sv','ntrpu_pcie_managed_native.sv')
    for name in ['ntrpu_pcie_fragment_decode','ntrpu_pcie_fragment_ingress','ntrpu_peer_rx_guard','ntrpu_pcie_fragment_peer']:
        text=once(text,'    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl '+name+'.sv]\n','')
    source_line=next(x for x in managed.splitlines() if x.startswith('    foreach name {ntrpu_buffer_pool_allocator '))
    old_line=next(x for x in text.splitlines() if x.startswith('    foreach name {ntrpu_axi_capacity_guard '))
    text=once(text,old_line,source_line)
    text=once(text,'    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_ingress_axi_cdc',
        '''    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_peer_memory_cdc
    set cdc [get_ips ntrpu_peer_memory_cdc]
    set_property -dict {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} $cdc
    generate_target all $cdc
    report_property -all $cdc -file [file join $output memory_cdc_properties.rpt]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_ingress_axi_cdc''')
    assert text.count('ntrpu_ddr_periodic_ingress.vh')==2
    text=text.replace('ntrpu_ddr_periodic_ingress.vh','ntrpu_dual_mig_periodic.vh')
    text=once(text,'    update_compile_order -fileset sim_1',
        '    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_dual_mig_fixture.sv]\n    update_compile_order -fileset sim_1')
    start=text.index('foreach suffix {stop source_stopped ');end=text.index('set objects {}',start)
    text=text[:start]+'''foreach suffix {awaddr wdata araddr wstrb awvalid wvalid bready arvalid rready awready wready arready bvalid rvalid bresp rresp rdata} {
    lappend names /board/EP/m_axil_$suffix
}
foreach suffix {host_rst_n transport_rst_n source_clk lifetime_reset_n} {lappend names /board/ntrpu_bypass/$suffix}
''' +text[end:]
    text=text.replace('NTRPU_NATIVE_MIG_SIMULATOR_RETURNED','NTRD_NATIVE_DUAL_MIG_SIMULATOR_RETURNED')
    return text


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'fpga/tb/tb_ntrpu_pcie_managed_host_cold_r2.sv'
    native=root/'tools/ntrpu_board_pcie_fragment_native_host.tcl'
    managed=root/'tools/ntrpu_board_pcie_managed_host_cold_r2.tcl'
    task_source=root/'fpga/tb/ntrpu_pcie_fragment_native_host_tasks.vh'
    products={'fpga/tb/ntrpu_pcie_managed_native.sv':bench(source.read_text()),
              'fpga/tb/ntrpu_pcie_managed_native_tasks.vh':tasks(task_source.read_text()),
              'tools/ntrpu_board_pcie_managed_native.tcl':tcl(native.read_text(),managed.read_text())}
    assert all(not (root/name).exists() for name in products)
    for name,value in products.items():
        with (root/name).open('x',newline='\n') as f:f.write(value)
    print(json.dumps(dict(inputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,native,managed,task_source]},
        outputs={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in products}),indent=2))


if __name__=='__main__':main()
