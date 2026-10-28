"""Prepare an auditable private variant; never edit the qualified source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from run_ntrpu_read_return import SOURCES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def prepare(root,out,depth,binding,stub):
    assert depth in (1,2,4,8)
    out.mkdir(parents=True,exist_ok=False)
    extra=['tools/run_ntrpu_read_return_native.py','tools/ntrpu_read_return_native_run.tcl',
           'tools/check_ntrpu_read_return_native.py','tools/check_ntrpu_read_return.py',
           'tools/prepare_ntrpu_read_window.py','tools/check_ntrpu_read_window.py',
           'fpga/experimental/ntrpu_hbm_read_tracker.sv']
    originals={}
    for name in dict.fromkeys(SOURCES+extra):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target);originals[name]=sha(root/name)
    name='fpga/experimental/ntrpu_hbm_read_adapter.sv'
    original=(root/name).read_text()
    start=original.index('    reg busy,ar_sent;')
    end=original.index('    hbm_read_width width_converter (')
    prefix=original[:start].replace('output reg ','output wire ')
    control='''    wire [255:0] validated_data;
    wire [5:0] validated_rid;
    wire [1:0] validated_resp;
    wire validated_last;
    ntrpu_hbm_read_tracker #(.MAX_PENDING(DEPTH)) tracker(.*);
'''.replace('DEPTH',str(depth))
    (out/name).write_text(prefix+control+original[end:]+'\n'+
                         (root/'fpga/experimental/ntrpu_hbm_read_tracker.sv').read_text())
    name='fpga/tb/tb_ntrpu_read_return.sv'
    text=(root/name).read_text()
    text=replace_once(text,'revision=3",COMMON,NATIVE','revision=3 window='+str(depth)+'",COMMON,NATIVE')
    text=replace_once(text,'    integer injection=0,inject_index=0;',
                     '    integer injection=0,inject_index=0;\n    integer injection_cursor=0;\n    always @(negedge mem_clk)injection_cursor=native_reads[0];')
    text=replace_once(text,'(native_reads[0]==inject_index)','(injection_cursor==inject_index)')
    monitor='''
    for(genvar w=0;w<2;w++)begin: window_monitor
        always @(posedge mem_clk)if(mem_rst_n)begin
            $display("RW STATE %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",stamp(),epoch,w,
                dut.channels[w].adapter.tracker.reserved_count,dut.channels[w].adapter.tracker.accepted_count,
                dut.channels[w].adapter.tracker.capture_remaining,dut.channels[w].adapter.tracker.offer_valid,
                dut.channels[w].adapter.tracker.response_cut.count,dut.channels[w].adapter.tracker.validated_cut.count,
                dut.channels[w].adapter.tracker.fault);
            if(dut.channels[w].adapter.tracker.reserve_fire)
                $display("RW RESERVE %0d %0d %0d %h %0d",stamp(),epoch,w,
                    dut.channels[w].adapter.p_araddr,int'(dut.channels[w].adapter.p_arlen)+1);
            if(dut.channels[w].adapter.tracker.validate_fire)
                $display("RW VALIDATE %0d %0d %0d %0d %0d %0d %h",stamp(),epoch,w,
                    dut.channels[w].adapter.tracker.q_rid,dut.channels[w].adapter.tracker.q_resp,
                    dut.channels[w].adapter.tracker.q_last,dut.channels[w].adapter.tracker.q_data);
            if(dut.channels[w].adapter.p_rvalid&&dut.channels[w].adapter.p_rready)
                $display("RW RETIRE %0d %0d %0d %0d %0d %0d %h",stamp(),epoch,w,
                    dut.channels[w].adapter.validated_rid,dut.channels[w].adapter.validated_resp,
                    dut.channels[w].adapter.validated_last,dut.channels[w].adapter.validated_data);
        end
    end
'''
    text=replace_once(text,'    initial begin #200000;',monitor+'    initial begin #200000;')
    (out/name).write_text(text)
    # Separate experiment/review identifiers prevent accidental baseline credit.
    names=['tools/run_ntrpu_read_return.py','tools/run_ntrpu_read_return_native.py',
           'tools/check_ntrpu_read_return_native.py']
    for name in names:
        text=(out/name).read_text()
        text=text.replace("experiment='read_return'","experiment='read_window'")
        text=text.replace("experiment='read_return_native'","experiment='read_window_native'")
        text=text.replace("r['experiment']=='read_return_native'","r['experiment']=='read_window_native'")
        text=text.replace('READ_RETURN_PREFLIGHT_REVIEWED','READ_WINDOW_PREFLIGHT_REVIEWED')
        text=text.replace('READ_RETURN_NATIVE_SNAPSHOT_REVIEWED','READ_WINDOW_NATIVE_SNAPSHOT_REVIEWED')
        text=text.replace('READ_RETURN_NATIVE_TRAFFIC_REVIEWED','READ_WINDOW_NATIVE_TRAFFIC_REVIEWED')
        text=text.replace("('DUAL_HBM_NATIVE_SNAPSHOT_REVIEWED','READ_WINDOW_NATIVE_SNAPSHOT_REVIEWED')",
                          "('DUAL_HBM_NATIVE_SNAPSHOT_REVIEWED','READ_RETURN_NATIVE_SNAPSHOT_REVIEWED','READ_WINDOW_NATIVE_SNAPSHOT_REVIEWED')")
        (out/name).write_text(text)
    shutil.copyfile(root/'tools/check_ntrpu_read_window.py',out/'tools/check_ntrpu_read_return.py')
    fixtures=out/'fixtures';fixtures.mkdir()
    shutil.copyfile(binding,fixtures/'native.sv')
    text=stub.read_text()
    text=replace_once(text,'    reg [1:0] active=0,reading=0;', '''    reg [1:0] active=0,reading=0;
    reg [33:0] requests[0:1][0:127];
    integer request_lengths[0:1][0:127],request_due[0:1][0:127];
    reg [5:0] request_ids[0:1][0:127];
    integer request_write[0:1],request_read[0:1],request_beat[0:1],memory_cycle=0;''')
    text=replace_once(text,'            active=0;reading=0;', '''            memory_cycle=0;
            for(integer lane=0;lane<2;lane++)begin request_write[lane]=0;request_read[lane]=0;request_beat[lane]=0;end
            active=0;reading=0;''')
    text=replace_once(text,'        end else for(integer lane=0;lane<2;lane++)begin',
                     '        end else begin\n        memory_cycle++;\n        for(integer lane=0;lane<2;lane++)begin')
    start=text.index('            if(s_arvalid[lane]&&s_arready[lane])begin')
    end=text.index('\n        end\n    end\n    always @(negedge',start)
    text=text[:start]+'''            if(s_arvalid[lane]&&s_arready[lane])begin
                if(request_write[lane]>=128)$fatal(1,"STUB_REQUEST_OVERFLOW");
                requests[lane][request_write[lane]]=s_araddr[lane*34+:34];
                request_lengths[lane][request_write[lane]]=int'(s_arlen[lane*4+:4])+1;
                request_ids[lane][request_write[lane]]=s_arid[lane*6+:6];
                request_due[lane][request_write[lane]]=memory_cycle+48;
                request_write[lane]++;
            end
            if(s_rvalid[lane]&&s_rready[lane])begin
                request_beat[lane]++;
                if(request_beat[lane]==request_lengths[lane][request_read[lane]])begin
                    request_read[lane]++;request_beat[lane]=0;
                end
            end
            if(!s_rvalid[lane]||s_rready[lane])begin
                reading[lane]=request_read[lane]<request_write[lane]&&memory_cycle>=request_due[lane][request_read[lane]];
                s_rvalid[lane]<=reading[lane];
                if(reading[lane])begin
                    raddress[lane]=requests[lane][request_read[lane]]+32*request_beat[lane];
                    if(!memory.exists(raddress[lane]))$fatal(1,"STUB_READ_UNWRITTEN");
                    s_rdata[lane*256+:256]<=memory[raddress[lane]];
                    s_rid[lane*6+:6]<=request_ids[lane][request_read[lane]];s_rresp[lane*2+:2]<=0;
                    s_rlast[lane]<=request_beat[lane]+1==request_lengths[lane][request_read[lane]];
                end
            end
        end'''+text[end:]
    text=replace_once(text,'axi_cold_reset_n&&!reading[lane]&&!s_rvalid[lane]',
                     'axi_cold_reset_n&&(request_write[lane]-request_read[lane]<32)')
    (fixtures/'stub.sv').write_text(text)
    result=dict(variant='read_window',depth=depth,originals=originals,
                prepared={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},
                qualified_tree_unchanged=all(sha(root/n)==h for n,h in originals.items()),
                cold_reset_only=True,extra_payload_frames=0)
    assert result['qualified_tree_unchanged']
    (out/'VARIANT.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);ap.add_argument('--depth',type=int,choices=(1,2,4,8),required=True)
    ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--stub',type=Path,required=True)
    a=ap.parse_args();r=prepare(Path(__file__).resolve().parents[1],a.output,a.depth,a.binding,a.stub)
    print(json.dumps(dict(depth=r['depth'],files=len(r['prepared']),qualified_tree_unchanged=r['qualified_tree_unchanged'])))
