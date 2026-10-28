"""Derive a new native NTRF bench; retain qualified cold MIG and observers."""
from pathlib import Path


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def build_bench(text):
    text = once(text, 'module ntrpu_pcie_ingress_native_host_harness;',
                'module ntrpu_pcie_fragment_native_host_harness;')
    text = once(text, 'ntrpu_pcie_ingress_host dut (', 'ntrpu_pcie_fragment_host dut (')
    text = once(text, '    wire [3:0] error_code;', '''    wire [3:0] error_code;
    wire [31:0] rejected_frames,rejected_identity;
    wire [3:0] last_reject_code;
    wire [6:0] last_reject_faults;''')
    text = once(text, '        .transport_idle(transport_idle),.valid_bytes(valid_bytes),',
                '''        .transport_idle(transport_idle),.valid_bytes(valid_bytes),
        .rejected_frames(rejected_frames),.rejected_identity(rejected_identity),
        .last_reject_code(last_reject_code),.last_reject_faults(last_reject_faults),''')
    changes = {
        'before the 52 fixture reads.': 'before the 72 fixture reads.',
        "epoch!=64'(job)": "epoch!=(job==0 ? 64'd0 : 64'h100000000+job)",
        "epoch!==64'(job+1)": "epoch!==64'h100000001+job",
        'case(job) 0:count=16;1:count=33;2:count=2;3:count=32;endcase':
            'case(job) 0:count=3;1:count=4;2:count=2;3:count=3;endcase',
        "!failed || error_code!=4 || line_valid || line_count!=begin_line || valid_bytes!==64'hf":
            "failed || error_code!=0 || line_valid || line_count!=begin_line || valid_bytes!==64'hffffffff",
        'NTRPU_LINK_CONFLICT_STATE': 'NTRPU_LINK_PARTIAL_BEFORE_STOP',
        'control_aw!=78 || control_w!=78 || control_b!=78 || control_ar!=53 || control_r!=53':
            'control_aw!=82 || control_w!=82 || control_b!=82 || control_ar!=73 || control_r!=73',
        'aw_count!=83 || w_count!=83 || b_count!=83': 'aw_count!=12 || w_count!=12 || b_count!=12',
        'device_aw!=83 || device_w!=83 || device_b!=83': 'device_aw!=12 || device_w!=12 || device_b!=12',
        'source=83 ps=': 'source=12 ps=',
        "((current_job==2 && b_count==begin_b+1) ? 2'd2 : 2'd0)":
            "(b_count==begin_b ? (current_job==3 ? 2'd3 : 2'd2) : 2'd0)",
    }
    for old,new in changes.items(): text = once(text,old,new)
    text = once(text, '            verify_periodic_complete();', '''            verify_periodic_complete();
            if(rejected_frames!=4 || rejected_identity!=2 || last_reject_code!=3 || last_reject_faults!=4)
                $fatal(1,"NTRPU_NATIVE_FRAGMENT_TELEMETRY");''')
    return text


def build_tcl(text):
    for old,new in {
        'patch_ntrpu_pcie_native_host.py':'patch_ntrpu_pcie_fragment_native_host.py',
        'ntrpu_pcie_ingress_native_host_r2.sv':'ntrpu_pcie_fragment_native_host.sv',
        'ntrpu_pcie_dword_decode.sv':'ntrpu_pcie_fragment_decode.sv',
        'ntrpu_pcie_line_ingress.sv':'ntrpu_pcie_fragment_ingress.sv',
        'ntrpu_pcie_ingress_peer.sv':'ntrpu_pcie_fragment_peer.sv',
        'ntrpu_axi_capacity_guard ntrpu_pcie_ingress_guarded ntrpu_pcie_ingress_async ntrpu_pcie_ingress_control ntrpu_axil_cdc_recode ntrpu_pcie_ingress_host':
        'ntrpu_axi_capacity_guard ntrpu_pcie_fragment_guarded ntrpu_pcie_fragment_async ntrpu_pcie_fragment_control ntrpu_axil_cdc_recode ntrpu_pcie_fragment_host',
        'foreach suffix {irq protocol_fault}':'foreach suffix {irq protocol_fault rejected_frames rejected_identity last_reject_code last_reject_faults}',
    }.items(): text = once(text,old,new)
    return text


def build_tasks(text):
    start=text.index('task NTRPU_LINK_WRITE(')
    end=text.index('task NTRPU_BYPASS_RUN;',start)
    text=text[:start]+'''task NTRPU_FRAME_WRITE(input integer job,half_offset,kind,input [31:0] offset);
    reg [511:0] frame;
    reg [63:0] address,frame_epoch,frame_tag,magic;
    reg [3:0] last_be;
    integer payload_job;
    begin
        magic=64'h313046555052544e;
        frame_epoch=64'h100000001+job;frame_tag=64'h8000000100002000+job;
        address=64'h900000000+offset;last_be=4'hf;payload_job=job;
        case(kind)
            1:magic=magic^1;
            2:frame_tag=frame_tag^64'h10000000000;
            3:begin frame_epoch=64'h100000002;frame_tag=64'h8000000100002001;payload_job=1;end
            4:address=64'h900000000;
        endcase
        frame[63:0]=magic;frame[127:64]=frame_epoch;frame[191:128]=frame_tag;
        frame[255:192]=64'(half_offset);
        for(integer lane=0;lane<32;lane++) frame[256+lane*8+:8]=8'(payload_job*53+(half_offset+lane)*29+23);
        DEFAULT_TAG=10'h040+ntrpu_data_writes;ntrpu_data_writes++;
        $display("NTRPU_NATIVE_FRAME_WRITE n=%0d job=%0d addr=%h data=%h kind=%0d tag=%h ps=%0.0f",ntrpu_data_writes,job,address,frame,kind,DEFAULT_TAG,$realtime/1ps);
        // Replaced by the qualified two-beat producer in the fresh public copy.
        TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd16,address,last_be,4'hf,1'b0);
        TSK_TX_CLK_EAT(4);
    end
endtask
''' +text[end:]
    text=once(text,"NTRPU_CTRL_READ('h00,'h4952544e,'1);","NTRPU_CTRL_READ('h00,'h4652544e,'1);")
    text=once(text,"NTRPU_CTRL_READ('h08,1,'1);", """NTRPU_CTRL_READ('h08,'h1f,'1);
        NTRPU_CTRL_READ('h9c,'h5052544e,'1);NTRPU_CTRL_READ('ha0,'h31304655,'1);
        NTRPU_CTRL_READ('ha4,64,'1);NTRPU_CTRL_READ('ha8,32,'1);""")
    for old,new in {
        "NTRPU_CTRL_WRITE('h24,0)":"NTRPU_CTRL_WRITE('h24,1)",
        "NTRPU_CTRL_WRITE('h2c,0)":"NTRPU_CTRL_WRITE('h2c,'h80000001)",
        "NTRPU_CTRL_READ('h64,0,'1)":"NTRPU_CTRL_READ('h64,1,'1)",
        "NTRPU_CTRL_READ('h6c,0,'1)":"NTRPU_CTRL_READ('h6c,'h80000001,'1)",
        "NTRPU_CTRL_WRITE('h44,0)":"NTRPU_CTRL_WRITE('h44,1)",
        "NTRPU_CTRL_WRITE('h4c,1)":"NTRPU_CTRL_WRITE('h4c,'h80000000)",
        "NTRPU_CTRL_WRITE('h4c,0)":"NTRPU_CTRL_WRITE('h4c,'h80000001)",
        'writes=83 lines=3':'writes=12 lines=3',
    }.items():text=once(text,old,new)
    start=text.index('            if(job==2) begin\n                NTRPU_LINK_WRITE')
    end=text.index('            board.ntrpu_bypass.drain_job(job);',start)
    text=text[:start]+'''            NTRPU_FRAME_WRITE(job,0,job+1,offset);
            NTRPU_FRAME_WRITE(job,0,0,offset);
            if(job==1) NTRPU_FRAME_WRITE(job,0,0,offset);
            if(job!=2) NTRPU_FRAME_WRITE(job,32,0,offset);
'''+text[end:]
    text=once(text,'            board.ntrpu_bypass.publish_job(job);', '''            NTRPU_CTRL_WRITE('h10,8);
            NTRPU_CTRL_READ('h90,job+1,'1);
            NTRPU_CTRL_READ('h94,job==0 ? 0 : job==1 ? 1 : 2,'1);
            NTRPU_CTRL_READ('h98,job==0 ? 'h82 : job==1 ? 'h207 : job==2 ? 'h307 : 'h43,'1);
            NTRPU_CTRL_READ('hac,1,'1);
            board.ntrpu_bypass.publish_job(job);''')
    return text


def main():
    root=Path(__file__).resolve().parents[1]
    sources=[('fpga/tb/ntrpu_pcie_ingress_native_host_r2.sv','fpga/tb/ntrpu_pcie_fragment_native_host.sv',build_bench),
             ('tools/ntrpu_board_pcie_ingress_native_host_r2.tcl','tools/ntrpu_board_pcie_fragment_native_host.tcl',build_tcl),
             ('fpga/tb/ntrpu_pcie_native_host_tasks.vh','fpga/tb/ntrpu_pcie_fragment_native_host_tasks.vh',build_tasks)]
    for old,new,transform in sources:
        with (root/new).open('x',newline='\n') as f:f.write(transform((root/old).read_text()))
        print(new)


if __name__=='__main__':main()
