"""Derive the existing 21-case data-path regression with AXI-Lite-only control.

The source data, destination BFM and byte comparison monitors remain unchanged.
STOP cases initiate an MMIO command at the named phase; application follows the
AXI-Lite handshake latency. They do not claim same-cycle STOP injection.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
s = (root/'fpga/tb/tb_ntrpu_pcie_ingress_guarded.sv').read_text()


def replace(before, after):
    global s
    assert s.count(before) == 1, before
    s = s.replace(before, after)


replace('module tb_ntrpu_pcie_ingress_guarded;', 'module tb_ntrpu_pcie_ingress_managed;')
replace('reg rst_n=0,stop=0,source_stopped=0,ingress_isolated=0;',
        'reg rst_n=0,source_stopped=0,ingress_isolated=0;\n    wire stop;')
replace('reg arm_valid=0;', 'wire arm_valid;')
replace('reg [63:0] arm_epoch=0,arm_tag=0,arm_window=0,arm_destination=0;',
        'wire [63:0] arm_epoch,arm_tag,arm_window,arm_destination;')
replace('reg result_ready=0;', 'wire result_ready;')
replace('ntrpu_pcie_ingress_guarded dut (.*);', '''
    reg [31:0] s_axil_awaddr=0,s_axil_wdata=0,s_axil_araddr=0;
    reg [3:0] s_axil_wstrb=15;
    reg s_axil_awvalid=0,s_axil_wvalid=0,s_axil_bready=0,s_axil_arvalid=0,s_axil_rready=0;
    wire s_axil_awready,s_axil_wready,s_axil_arready,s_axil_bvalid,s_axil_rvalid;
    wire [1:0] s_axil_bresp,s_axil_rresp;
    wire [31:0] s_axil_rdata;
    wire irq,protocol_fault;
    integer control_writes=0,control_reads=0;
    ntrpu_pcie_ingress_managed dut (.*);
    task automatic mmio(input [31:0] address,input [31:0] data,input [1:0] resp=0);
        fork
            begin
                @(negedge clk);s_axil_awaddr=address;s_axil_awvalid=1;
                do @(posedge clk);while(!s_axil_awready);
                @(negedge clk);s_axil_awvalid=0;
            end
            begin
                idle(1+seed%3);s_axil_wdata=data;s_axil_wvalid=1;
                do @(posedge clk);while(!s_axil_wready);
                @(negedge clk);s_axil_wvalid=0;
            end
        join
        while(!s_axil_bvalid) @(negedge clk);
        need(s_axil_bresp===resp,"MMIO_BRESP");
        idle(2);need(s_axil_bvalid && s_axil_bresp===resp,"MMIO_B_STABLE");
        s_axil_bready=1;@(posedge clk);@(negedge clk);s_axil_bready=0;
        control_writes=control_writes+1;
        $display("NTRPU_MANAGED_MMIO n=%0d addr=%h data=%h resp=%0d cycle=%0d",control_writes,address,data,resp,cycle);
    endtask
    task automatic peek(input [31:0] address,input [31:0] value);
        @(negedge clk);s_axil_araddr=address;s_axil_arvalid=1;
        do @(posedge clk);while(!s_axil_arready);
        @(negedge clk);s_axil_arvalid=0;
        need(s_axil_rvalid && s_axil_rresp==0 && s_axil_rdata===value,"MMIO_READ");
        idle(3);need(s_axil_rvalid && s_axil_rdata===value,"MMIO_R_STABLE");
        s_axil_rready=1;@(posedge clk);@(negedge clk);s_axil_rready=0;
        control_reads=control_reads+1;
        $display("NTRPU_MANAGED_READ n=%0d addr=%h data=%h cycle=%0d",control_reads,address,value,cycle);
    endtask
    task automatic configure(input [63:0] e,input [63:0] t,input [63:0] win,input [63:0] dest);
        mmio('h20,e[31:0]);mmio('h24,e[63:32]);mmio('h28,t[31:0]);mmio('h2c,t[63:32]);
        mmio('h30,win[31:0]);mmio('h34,win[63:32]);mmio('h38,dest[31:0]);mmio('h3c,dest[63:32]);
    endtask
''')
start=s.index('    task automatic start_job(')
end=s.index('    task automatic fragment(',start)
s=s[:start]+'''    task automatic start_job(input string name);
        reg [63:0] e,t,win,dest;
        source_stopped=0;ingress_isolated=0;quiesce_epoch=epoch;
        e=epoch+1;t=64'h1000+jobs+1;
        win=(jobs%2==0) ? 64'h1000fffc0 : 64'h100000000;
        case(jobs+1)
            13: dest=64'h2000000000;
            14: dest=64'h1000000000000000;
            18,21: dest=0;
            19: dest=64'h100000040;
            20: dest=64'hffffffffffffffc0;
            default: dest=64'h1fffffffc0;
        endcase
        jobs=jobs+1;down_aw=0;down_w=0;down_b=0;result_in_job=0;expected_status=0;
        expected_data=pattern(jobs);expected_bitmap=0;
        $display("NTRPU_GUARDED_CASE job=%0d name=%s",jobs,name);
        configure(e,t,win,dest);mmio('h10,1);
        idle(5);need(arm_valid && epoch==e-1 && !arm_ready,"ARM_WITHOUT_BARRIER");
        source_stopped=1;idle(5);need(!arm_ready && epoch==e-1,"ARM_WITHOUT_ISOLATION");
        quiesce_epoch=~epoch;ingress_isolated=1;
        idle(5);need(!arm_ready && epoch==e-1,"ARM_STALE_BARRIER");
        quiesce_epoch=epoch;
        while(epoch!=e) @(negedge clk);
        source_stopped=0;ingress_isolated=0;
        need(epoch==e && peer_epoch==e && lease_held && !failed && valid_bytes==0,"ARM_STATE");
        mmio('h20,0,2);mmio('h10,4,2); // descriptor locked; no early release
    endtask
'''+s[end:]
replace('        stop=1;','        mmio(\'h10,2);')
replace('        arm_epoch=epoch+1;arm_valid=1;', "        mmio('h10,1,2);mmio('h10,4,2);")
replace('        arm_valid=0;source_stopped=0;ingress_isolated=0;', '        source_stopped=0;ingress_isolated=0;')
replace('        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;', '''
        peek('h60,result_epoch[31:0]);peek('h64,result_epoch[63:32]);
        peek('h68,result_tag[31:0]);peek('h6c,result_tag[63:32]);peek('h70,status);
        mmio('h40,result_epoch[31:0]);mmio('h44,result_epoch[63:32]);
        mmio('h48,result_tag[31:0]);mmio('h4c,result_tag[63:32]^1);mmio('h10,4,2);
        need(result_valid && lease_held && !arm_ready,"FORGED_ACK_RELEASED");
        mmio('h4c,result_tag[63:32]);mmio('h10,4);
''')
start=s.index('        // Actively offer invalid rearm requests;')
end=s.index('    endtask',start)
s=s[:start]+'''        configure(epoch,arm_tag,arm_window,arm_destination);mmio('h10,1,2);
        configure(epoch+1,arm_tag,arm_window+1,arm_destination);mmio('h10,1,2);
        configure(epoch+1,arm_tag,arm_window,arm_destination+1);mmio('h10,1,2);
        need(epoch==jobs && peer_epoch==jobs && !protocol_fault,"CONTROL_RECOVERY_STATE");
'''+s[end:]
replace('idle(5);stop=1;finish_job(1,0);', "idle(5);mmio('h10,2);finish_job(1,0);")
replace('aw(arm_window,5,0,2);stop=1;idle(5);', "aw(arm_window,5,0,2);mmio('h10,2);idle(5);")
replace('        need(epoch===peer_epoch,"EPOCH_DIVERGENCE");', '        need(epoch===peer_epoch && !protocol_fault,"EPOCH_DIVERGENCE");')
replace('        $finish;', '''        $display("NTRPU_MANAGED_PASS writes=%0d reads=%0d jobs=%0d seed=%0d",control_writes,control_reads,jobs,seed);
        $finish;''')
out=root/'fpga/tb/tb_ntrpu_pcie_ingress_managed.sv'
with out.open('x') as f:
    f.write('// Generated by tools/prepare_ntrpu_ingress_managed_tb.py. See STOP latency scope.\n'+s)
print(out)
