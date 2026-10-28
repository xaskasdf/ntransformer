"""Compare public RTL and routed outputs without changing the routed netlist."""
import json,re
from pathlib import Path
from check_ntrpu_read_return import sha
SOURCE=('rst_n','started','cmd_ready','data_valid','cpl_valid','fault','drained','stopped','frame_idle')
MEMORY=('mem_rst_n','h_arvalid','h_rready','adapter_pending','adapter_fault')
S_DATA={'data':('data_valid',512),'data_tag':('data_valid',64),'data_last':('data_valid',1),'cpl_tag':('cpl_valid',64),'cpl_status':('cpl_valid',2),'cpl_bytes':('cpl_valid',13)}
M_DATA={'h_araddr':('h_arvalid',34),'h_arlen':('h_arvalid',4),'h_arid':('h_arvalid',6),'h_arsize':('h_arvalid',3),'h_arburst':('h_arvalid',2)}

def prepare(root,out,netlist):
    header=(root/'fpga/experimental/ntrpu_hbm_read_return.sv').read_text().split('module ntrpu_hbm_read_return (',1)[1].split(');',1)[0]
    ports=[]
    for direction,width,names in re.findall(r'\b(input|output)\s+(?:wire|reg)\s*(\[[^\]]+\])?\s*(.*?)(?=\b(?:input|output)\b|\Z)',header,re.S):
        for name in names.strip().rstrip(',').split(','):
            name=name.strip();assert re.fullmatch(r'\w+',name);ports.append((direction,width,name))
    assert {n for d,w,n in ports if d=='output'}==set(SOURCE+MEMORY)|set(S_DATA)|set(M_DATA)
    raw=netlist.read_bytes();text=raw.decode();top=text.split('module ntrpu_read_return_physical',1)[1].split('\n  wire',1)[0]
    assert set(re.findall(r'^  (input|output) (\[[^\]]+\])?\s*(\w+);',top,re.M))==set(ports)
    lines=['integer dm_source=0,dm_memory=0,dm_resets=0;',
           'wire dm_bad_payload=$test$plusargs("DM_BAD_PAYLOAD");',
           'wire dm_bad_ready=$test$plusargs("DM_BAD_READY");']
    lines += [f'wire {w} route_{n};' for d,w,n in ports if d=='output']
    lines += ['wire [1023:0] raw_data;wire [1:0] raw_cmd_ready;',
              "assign route_data=raw_data ^ ((dm_bad_payload&&scenario==1&&route_data_valid[0])?1024'b1:1024'b0);",
              "assign route_cmd_ready=raw_cmd_ready ^ ((dm_bad_ready&&scenario==1&&started)?2'b1:2'b0);",
              'ntrpu_read_return_physical routed_dut(']
    pins=[]
    for d,w,n in ports:
        prefix=('raw_' if n in ('data','cmd_ready') else 'route_') if d=='output' else ''
        pins.append(f'.{n}({prefix}{n})')
    lines += [',\n'.join(pins)+');',
              'task automatic dm_reset_check;',
              'if({rst_n,route_rst_n,mem_rst_n,route_mem_rst_n}!==4\'b0)$fatal(1,"DM_RESET_MISMATCH");',
              'dm_resets++;$display("DM RESET %0d %0d",stamp(),epoch);endtask',
              'initial begin #150.2;dm_reset_check;end',
              'always @(negedge cold_reset_n) if($realtime>150)begin #0.2;dm_reset_check;end']
    for clock,reset,names,data,domain in [('clk','rst_n',SOURCE,S_DATA,'source'),('mem_clk','mem_rst_n',MEMORY,M_DATA,'memory')]:
        lines += [f'always @(posedge {clock})begin',
                  'if($realtime>150&&cold_reset_n)begin',
                  f'if(route_{reset}!=={reset})$fatal(1,"DM_MISMATCH {reset}");',f'if({reset})begin']
        for n in names:
            if n!=reset:lines.append(f'if(route_{n}!=={n})$fatal(1,"DM_MISMATCH {n} case=%0d t=%0d ref=%h route=%h",scenario,stamp(),{n},route_{n});')
        lines+=['for(integer c=0;c<2;c++)begin']
        for n,(valid,width) in data.items():
            part=f'[c*{width}+:{width}]';lines.append(f'if({valid}[c]&&route_{n}{part}!=={n}{part})$fatal(1,"DM_MISMATCH {n} case=%0d t=%0d",scenario,stamp());')
        lines+=['end',f'dm_{domain}++;$display("DM {domain.upper()} %0d %0d",stamp(),dm_{domain});','end end end']
    bench=(root/'fpga/tb/tb_ntrpu_read_return.sv').read_text()
    # The standalone monitor increments native_reads with a blocking assignment
    # on the active memory edge. Deriving injected inputs directly from it races
    # a routed BUFG's delta-cycle propagation. Freeze the injection cursor on
    # the falling edge so both DUTs receive the same stable response metadata.
    cursor='(native_reads[0]==inject_index)'
    assert bench.count(cursor)==1
    bench=bench.replace(cursor,'(injection_cursor==inject_index)')
    bench=bench.replace('    integer injection=0,inject_index=0;',
        '    integer injection=0,inject_index=0;\n'
        '    integer injection_cursor=0;\n'
        '    always @(negedge mem_clk) injection_cursor=native_reads[0];')
    marker='    initial begin\n        $display("RR CONFIG'
    assert bench.count(marker)==1;bench=bench.replace(marker,'    initial begin\n        #150;\n        $display("RR CONFIG')
    assert bench.count('$finish;')==2
    bench=bench.replace('$finish;','$display("DM ALL_PASS source=%0d memory=%0d resets=%0d",dm_source,dm_memory,dm_resets);$finish;')
    assert bench.count('endmodule')==1;bench=bench.replace('endmodule','\n'.join(lines)+'\nendmodule')
    (out/'tb_ntrpu_read_return.sv').write_text(bench);(out/'route_isolated.v').write_bytes(raw)
    (out/'route_library.v').write_text('`uselib lib=read_return_routed lib=unisims_ver lib=secureip\n`include "route_isolated.v"\n`uselib\n')
    flow=(root/'tools/ntrpu_read_return.tcl').read_text()
    flow=flow[:flow.index('if {[llength $argv]!=3}')]
    flow=flow.replace('[file join $root fpga tb tb_ntrpu_read_return.sv]','[file join $output tb_ntrpu_read_return.sv]')
    flow=flow.replace('    launch_simulation;run all;close_sim','''    add_files -fileset sim_1 [file join $output route_library.v]
    set_property library read_return_routed [get_files [file join $output route_library.v]]
    launch_simulation;run all;close_sim''')
    flow=flow.replace('    close_project','''    if {[string first {DM ALL_PASS source=} $text]<0} {error "Routed comparison incomplete"}
    set here [pwd];set simdir [file join $output project hbm_adapter.sim sim_1 behav xsim]
    set f [open [file join $output negative.tcl] w];puts $f {run all;quit};close $f
    cd $simdir
    foreach kind {PAYLOAD READY} {
        set log [file join $output negative_${kind}.log]
        set failed [catch {exec xsim tb_ntrpu_read_return_behav -testplusarg DM_BAD_$kind -tclbatch [file join $output negative.tcl] -log $log 2>@1} message]
        set f [open [file join $output negative_${kind}_console.log] w];puts $f $message;close $f
        set ntext [read_text $log]
        set needle [expr {$kind eq "PAYLOAD" ? "DM_MISMATCH data" : "DM_MISMATCH cmd_ready"}]
        if {[string first $needle $ntext]<0 || [string first {DM ALL_PASS} $ntext]>=0} {error "Undetected comparator injection $kind"}
        puts "DM_NEGATIVE_REJECTED $kind"
    }
    cd $here
    close_project''')
    flow+='\nif {[catch {simulate_read_return {'+str(root)+'} {'+str(out)+'} 0 0} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n'
    (out/'run.tcl').write_text(flow)
    (out/'COMPARISON.json').write_text(json.dumps(dict(netlist_sha256=sha(netlist),isolated_sha256=sha(out/'route_isolated.v'),
        renamed_modules=0,library='read_return_routed',source_ports=SOURCE,memory_ports=MEMORY,ports=ports,
        reference_bench_sha256=sha(root/'fpga/tb/tb_ntrpu_read_return.sv'),sdf=False,force=False,
        latency_edges=0,global_reset_wait_ns=150,cold_reset_settle_ps=200,idle_data_masked=True,
        injection_cursor_sampling='falling_memory_edge'),indent=2)+'\n')
