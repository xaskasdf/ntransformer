"""Create a native-control variant, preserving all MIG/model and data monitors."""
from pathlib import Path


def once(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)


def build_bench(s):
    s=once(s,'module ntrpu_pcie_ingress_memory_harness;','module ntrpu_pcie_ingress_native_host_harness;')
    s=once(s,'reg stop=0,source_stopped=0,ingress_isolated=0;','wire stop;\n    reg source_stopped=0,ingress_isolated=0;')
    s=once(s,'reg arm_valid=0;','wire arm_valid;')
    s=once(s,'reg [63:0] arm_epoch=0,arm_tag=0,arm_window=0,arm_destination=0;','wire [63:0] arm_epoch,arm_tag,arm_window,arm_destination;')
    s=once(s,'reg result_ready=0;','wire result_ready;\n    wire irq,protocol_fault;')
    s=s.replace('dut.core.','dut.data_path.core.').replace('dut.d_axi_','dut.data_path.d_axi_')
    pins=['awaddr','wdata','araddr','wstrb','awvalid','wvalid','bready','arvalid','rready','awready','wready','arready','bvalid','rvalid','bresp','rresp','rdata']
    ports=''.join('        .s_axil_'+n+'(board.EP.m_axil_'+n+'),\n' for n in pins)
    s=once(s,'ntrpu_pcie_ingress_async dut (','ntrpu_pcie_ingress_host dut (\n'+ports+'        .irq(irq),.protocol_fault(protocol_fault),')
    a=s.index('    task arm_job(');z=s.index('    task finish_job(',a)
    s=s[:a]+'''    task wait_cold_ready;
        wait(host_rst_n && rst_n);
    endtask
    task prepare_job(input integer job);
        begin
            @(negedge clk);
            current_job=job;begin_b=b_count;begin_line=line_count;begin_result=result_count;begin_dest=dest_b;
            source_stopped=0;ingress_isolated=0;quiesce_epoch=epoch-1;
        end
    endtask
    task allow_arm(input integer job);
        begin
            if(!arm_valid || epoch!=64'(job)) $fatal(1,"NTRPU_NATIVE_CONTROL_PENDING_ARM");
            @(negedge clk);source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        end
    endtask
    task after_arm(input integer job);
        begin
            @(negedge clk);
            if(epoch!==64'(job+1) || peer_epoch!==epoch || valid_bytes!==0 || failed || arm_valid)
                $fatal(1,"NTRPU_NATIVE_CONTROL_ARM_STATE");
            source_stopped=0;ingress_isolated=0;
        end
    endtask
''' +s[z:]
    s=once(s,'    task finish_job(input integer job);','    task drain_job(input integer job);')
    marker='            repeat(4) @(negedge clk);\n            if(result_valid) $fatal(1,"NTRPU_LINK_BARRIER_REQUIRED");'
    s=once(s,marker,'        end\n    endtask\n    task publish_job(input integer job);\n        begin\n'+marker)
    s=once(s,"result_status!==(job==2 ? 4'd4 : 4'd0)","result_status!==(job==2 ? 4'd1 : 4'd0)")
    s=once(s,'            result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;',
        '        end\n    endtask\n    task after_retire(input integer job);\n        begin')
    s=once(s,'            verify_periodic_complete();',
        '            verify_periodic_complete();\n            if(control_aw!=78 || control_w!=78 || control_b!=78 || control_ar!=52 || control_r!=52 || protocol_fault) $fatal(1,"NTRPU_NATIVE_CONTROL_DRAIN");')
    monitor='''    integer control_aw=0,control_w=0,control_b=0,control_ar=0,control_r=0;
    always @(posedge host_clk) if(host_rst_n) begin
        if(protocol_fault) $fatal(1,"NTRPU_NATIVE_CONTROL_FAULT");
'''
    for domain,clock,prefix,guard in [('HOST','host_clk','board.EP.m_axil_','host_rst_n'),('DEVICE','clk','dut.d_axil_','rst_n')]:
        if domain=='DEVICE':monitor+='    end\n    always @(posedge clk) if(rst_n) begin\n'
        for channel,names in [('AW',['awaddr']),('W',['wdata','wstrb']),('B',['bresp']),('AR',['araddr']),('R',['rdata','rresp'])]:
            ch=channel.lower();fmt=' '.join(n+'=%h' for n in names)
            monitor+=f'        if({prefix}{ch}valid && {prefix}{ch}ready) begin\n'
            if domain=='HOST':monitor+=f'            control_{ch}++;\n'
            monitor+=f'            $display("NTRPU_NATIVE_CONTROL_{domain}_{channel} {fmt} ps=%0d",'+','.join(prefix+n for n in names)+',$time);\n        end\n'
    monitor+='    end\n'
    s=once(s,'    initial begin #300000000;',monitor+'    initial begin #700000000;')
    return s


def build_tcl(s):
    s=s.replace('patch_ntrpu_pcie_ingress_memory.py','patch_ntrpu_pcie_native_host.py').replace('ntrpu_pcie_ingress_memory_observed.sv','ntrpu_pcie_ingress_native_host.sv')
    s=s.replace('ntrpu_axi_capacity_guard ntrpu_pcie_ingress_guarded ntrpu_pcie_ingress_async}',
        'ntrpu_axi_capacity_guard ntrpu_pcie_ingress_guarded ntrpu_pcie_ingress_async ntrpu_pcie_ingress_control ntrpu_axil_cdc_recode ntrpu_pcie_ingress_host}')
    marker='    report_ip_status -file [file join $output ip_status.rpt]'
    s=once(s,marker,'''    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_host_axil_cdc
    set control [get_ips ntrpu_host_axil_cdc]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $control]]
    set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3] $control
    generate_target all $control
    report_property -all -file [file join $output control_cdc_properties.rpt] $control
'''+marker)
    s=s.replace('/dut/d_axi_','/dut/data_path/d_axi_')
    s=once(s,'set objects {}','''foreach suffix {awaddr wdata araddr wstrb awvalid wvalid bready arvalid rready awready wready arready bvalid rvalid bresp rresp rdata} {
    lappend names /board/EP/m_axil_$suffix
    lappend names /board/ntrpu_bypass/dut/d_axil_$suffix
}
foreach suffix {irq protocol_fault} {lappend names /board/ntrpu_bypass/$suffix}
set objects {}''')
    s=s.replace('{$i<305}','{$i<705}')
    return s


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    outputs={
      'fpga/tb/ntrpu_pcie_ingress_native_host.sv':build_bench((root/'fpga/tb/ntrpu_pcie_ingress_memory_observed.sv').read_text()),
      'tools/ntrpu_board_pcie_ingress_native_host.tcl':build_tcl((root/'tools/ntrpu_board_pcie_ingress_memory_observed.tcl').read_text())}
    for path,text in outputs.items():
        with (root/path).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
