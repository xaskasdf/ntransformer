"""Create the isolated dual-clock control/data wrapper and its observed MIG bank.

Preserves the qualified data path and memory model. All outputs are new files.
The AXI-Lite bridge reuses the reversible response coding from the FFN path.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
managed=(root/'fpga/rtl/ntrpu_pcie_ingress_managed.sv').read_text()
head=managed[managed.index('module '):managed.index('\n);')+3]
head=head.replace('ntrpu_pcie_ingress_managed','ntrpu_pcie_ingress_host',1)
head=head.replace('input wire clk, rst_n,','input wire clk, rst_n, host_clk, host_rst_n,')
fields={
    'awaddr':32,'wdata':32,'araddr':32,'wstrb':4,
    'awvalid':1,'wvalid':1,'bready':1,'arvalid':1,'rready':1,
    'awready':1,'wready':1,'bvalid':1,'arready':1,'rvalid':1,
    'bresp':2,'rresp':2,'rdata':32,
}
body='\n'
for name,width in fields.items():
    body+='    wire '+(f'[{width-1}:0] ' if width>1 else '')+'d_axil_'+name+';\n'
body+='    ntrpu_pcie_ingress_control control (\n'
body+=',\n'.join('        .s_axil_'+n+'(d_axil_'+n+')' for n in fields)+',\n        .*);\n'
body+='    ntrpu_pcie_ingress_async #(.MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH)) data_path (.*);\n'
body+='    ntrpu_axil_cdc_recode control_cdc (\n'
body+='        .s_axi_aclk(host_clk), .s_axi_aresetn(host_rst_n),\n        .m_axi_aclk(clk), .m_axi_aresetn(rst_n),\n'
body+='        .s_axi_awprot(3\'b0), .s_axi_arprot(3\'b0), .m_axi_awprot(), .m_axi_arprot(),\n'
body+=',\n'.join('        .s_axi_'+n+'(s_axil_'+n+'), .m_axi_'+n+'(d_axil_'+n+')' for n in fields)+'\n    );\nendmodule\n'
with (root/'fpga/rtl/ntrpu_pcie_ingress_host.sv').open('x',newline='\n') as f:
    f.write('`timescale 1ns / 1ps\n// Host AXI/AXI-Lite cross through real AMD bridges to UI-clock ingress.\n// External trusted barrier includes both upstream producers and queued data.\n// Diagnostic/control outputs are UI-domain; reset is cold/coordinated only.\n'+head+body)

s=(root/'fpga/tb/tb_ntrpu_pcie_ingress_cdc_observed.sv').read_text()
def replace(before,after):
    global s
    assert s.count(before)==1,before
    s=s.replace(before,after)
replace('module tb_ntrpu_pcie_ingress_cdc #','module tb_ntrpu_pcie_ingress_host #')
replace('reg rst_n=0,stop=0,source_stopped=0,ingress_isolated=0;',
        'reg rst_n=0,source_stopped=0,ingress_isolated=0;\n    wire stop;')
replace('reg arm_valid=0;','wire arm_valid;')
replace('reg [63:0] arm_epoch=0,arm_tag=0,arm_window=0,arm_destination=0;',
        'wire [63:0] arm_epoch,arm_tag,arm_window,arm_destination;')
replace('reg result_ready=0;','wire result_ready;')
s=s.replace('=dut.d_axi_','=dut.data_path.d_axi_')
replace('ntrpu_pcie_ingress_async dut (','ntrpu_pcie_ingress_host dut (')
mmio=(root/'fpga/tb/tb_ntrpu_pcie_ingress_managed.sv').read_text()
mmio=mmio[mmio.index('    reg [31:0] s_axil_awaddr'):mmio.index('    function automatic [511:0] pattern')]
mmio=mmio.replace('    ntrpu_pcie_ingress_managed dut (.*);\n','')
mmio=mmio.replace('negedge clk','negedge host_clk').replace('posedge clk','posedge host_clk')
mmio=mmio.replace('idle(','host_idle(')
mmio=mmio.replace('input [1:0] resp=0);','input [1:0] resp=0,input [3:0] strb=15);')
mmio=mmio.replace('s_axil_wdata=data;s_axil_wvalid=1;','s_axil_wdata=data;s_axil_wstrb=strb;s_axil_wvalid=1;')
mmio=mmio.replace('input [31:0] value);','input [31:0] value,input [1:0] resp=0);')
mmio=mmio.replace('s_axil_rresp==0','s_axil_rresp===resp')
mmio=mmio.replace('data=%h resp=%0d cycle=%0d",control_writes,address,data,resp,cycle',
                  'data=%h strb=%h resp=%0d ps=%0d",control_writes,address,data,strb,resp,longint\'($realtime*1000)')
mmio=mmio.replace('data=%h cycle=%0d",control_reads,address,value,cycle',
                  'data=%h resp=%0d ps=%0d",control_reads,address,value,resp,longint\'($realtime*1000)')
# A read response crosses the real CDC, so wait for RVALID after the address.
mmio=mmio.replace('        need(s_axil_rvalid && s_axil_rresp===resp',
                  '        while(!s_axil_rvalid) @(negedge host_clk);\n        need(s_axil_rvalid && s_axil_rresp===resp',1)
mmio=mmio.replace('NTRPU_MANAGED_MMIO','NTRPU_HOST_CONTROL_MMIO').replace('NTRPU_MANAGED_READ','NTRPU_HOST_CONTROL_READ')
mmio=mmio.replace('cycle=%0d",control_writes,address,data,resp,cycle','ps=%0d",control_writes,address,data,resp,longint\'($realtime*1000)')
mmio=mmio.replace('cycle=%0d",control_reads,address,value,cycle','ps=%0d",control_reads,address,value,longint\'($realtime*1000)')
replace('    logic sys_clk=0,sys_rst=1;',mmio+'\n    logic sys_clk=0,sys_rst=1;')
start=s.index('    task automatic start_job(');end=s.index('    task automatic fragment(',start)
s=s[:start]+'''    task automatic start_job(input string name);
        reg [63:0] e,t,win,dest;
        source_stopped=0;ingress_isolated=0;quiesce_epoch=epoch;
        e=epoch+1;t=64'h1000+jobs+1;
        win=(jobs%2==0) ? 64'h1000fffc0 : 64'h100000000;
        if(jobs+1<=6) dest=location(jobs);
        else case(jobs+1)
            7:dest=64'h2000000000;
            8:dest=64'h1000000000000000;
            9:dest=64'hffffffffffffffc0;
            10:dest=location(1);
            11:dest=location(5);
            default:dest=0;
        endcase
        jobs=jobs+1;down_aw=0;down_w=0;down_b=0;result_in_job=0;expected_status=0;
        expected_data=pattern(jobs);expected_bitmap=0;
        $display("NTRPU_MIG_LINE_CASE job=%0d name=%s",jobs,name);
        configure(e,t,win,dest);mmio('h10,1);
        idle(5);need(arm_valid && epoch==e-1 && !arm_ready,"ARM_WITHOUT_BARRIER");
        ingress_isolated=1;idle(5);need(!arm_ready && epoch==e-1,"ARM_WITHOUT_STOPPED_SOURCE");
        ingress_isolated=0;source_stopped=1;idle(5);need(!arm_ready && epoch==e-1,"ARM_WITHOUT_ISOLATION");
        quiesce_epoch=~epoch;ingress_isolated=1;
        idle(5);need(!arm_ready && epoch==e-1,"ARM_STALE_BARRIER");
        quiesce_epoch=epoch;
        while(epoch!=e) @(negedge clk);
        source_stopped=0;ingress_isolated=0;
        need(epoch==e && peer_epoch==e && lease_held && !failed && valid_bytes==0,"ARM_STATE");
        mmio('h20,0,2);mmio('h10,4,2);
    endtask
'''+s[end:]
replace('        stop=1;',"        mmio('h10,2);")
replace('        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;', '''
        peek('h60,result_epoch[31:0]);peek('h64,result_epoch[63:32]);
        peek('h68,result_tag[31:0]);peek('h6c,result_tag[63:32]);peek('h70,status);
        mmio('h40,result_epoch[31:0]);mmio('h44,result_epoch[63:32]);
        mmio('h48,result_tag[31:0]);mmio('h4c,result_tag[63:32]^1);mmio('h10,4,2);
        need(result_valid && lease_held && !arm_ready,"FORGED_ACK_RELEASED");
        mmio('h4c,result_tag[63:32]);mmio('h10,4);
''')
start=s.index('        // Actively offer invalid rearm requests;');end=s.index('    endtask',start)
s=s[:start]+'''        configure(epoch,arm_tag,arm_window,arm_destination);mmio('h10,1,2);
        configure(epoch+1,arm_tag,arm_window+1,arm_destination);mmio('h10,1,2);
        configure(epoch+1,arm_tag,arm_window,arm_destination+1);mmio('h10,1,2);
        need(epoch==jobs && peer_epoch==jobs && !protocol_fault,"CONTROL_RECOVERY_STATE");
'''+s[end:]
replace('        need(epoch===peer_epoch,"EPOCH_DIVERGENCE");','        need(epoch===peer_epoch && !protocol_fault,"EPOCH_DIVERGENCE");')
replace('        for(integer job=1;job<=13;job=job+1) begin', '''        peek(0,32'h4952544e);peek(1,0,3);peek('h10000000,0,3);peek('h90,0,3);
        mmio('h20,32'h11223344);mmio('h20,32'haabbccdd,0,5);peek('h20,32'h11bb33dd);
        for(integer job=1;job<=13;job=job+1) begin''')
replace('        $finish;', '''        need(control_writes==588 && control_reads==70,"CONTROL_COUNTS");
        $display("NTRPU_HOST_CONTROL_PASS writes=%0d reads=%0d jobs=%0d ps=%0d",control_writes,control_reads,jobs,longint'($realtime*1000));
        $finish;''')
# Observe both EXTERNAL sides of the recoding bridge, not its internal encoding.
monitor=''
for domain,clock,reset,prefix in [('HOST','host_clk','host_rst_n','s_axil_'),('DEVICE','clk','rst_n','dut.d_axil_')]:
    monitor+=f'    always @(posedge {clock}) if({reset}) begin\n'
    for ch,values in [('AW',('awaddr',)),('W',('wdata','wstrb')),('B',('bresp',)),('AR',('araddr',)),('R',('rdata','rresp'))]:
        lower=ch.lower(); fmt=' '.join(n+'=%h' for n in values)
        args=','.join(prefix+n for n in values)
        monitor+=f'        if({prefix}{lower}valid && {prefix}{lower}ready) $display("NTRPU_HOST_CONTROL_{domain}_{ch} {fmt} ps=%0d",{args},longint\'($realtime*1000));\n'
    monitor+='    end\n'
replace('    `include "ntrpu_ddr_periodic_ingress.vh"',monitor+'    `include "ntrpu_ddr_periodic_ingress.vh"')
with (root/'fpga/tb/tb_ntrpu_pcie_ingress_host.sv').open('x',newline='\n') as f:
    f.write('// Generated by tools/prepare_ntrpu_ingress_host.py. STOP application follows MMIO latency.\n'+s)
print('Created dual-clock ingress wrapper and observed MIG fixture')
