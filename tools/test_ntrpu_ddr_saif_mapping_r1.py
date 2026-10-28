"""Exercise Tcl sequencing and review rejection with synthetic tools, never Vivado."""
import argparse,copy,json,tempfile,tkinter
from pathlib import Path
import review_ntrpu_ddr_saif_mapping_r1 as check

def simulate(script,root,bad_top=False):
    root=Path(root);root.mkdir()
    for name in ('design.dcp','0.saif','1.saif'):(root/name).write_text('synthetic\n')
    t=tkinter.Tcl();t.call('set','argv',tuple(str(root/n) for n in ('design.dcp','0.saif','1.saif','output')))
    t.eval(r'''
set trace {}; set active 0; set phase inherited
proc get_projects {args} {return {}}
proc set_param {args} {}
proc open_checkpoint {path} {
 global trace active phase
 if {$active} {error "Design reused"};set active 1;set phase inherited;lappend trace open
}
proc close_project {} {global trace active;lappend trace close;set active 0}
proc current_design {} {return design_1}
proc version {args} {return 2026.1}
proc get_ports {} {set p {};for {set i 0} {$i<154} {incr i} {lappend p port_$i};return $p}
proc get_nets {args} {return {net_0 {bus[0]}}}
proc get_cells {args} {return {cell_0}}
proc get_property {key object} {
 if {$key eq "NAME"} {return $object}
 if {$key eq "TOP"} {return $::test_top}
 if {$key eq "PART"} {return xcvu5p-flva2104-2-i}
 if {$key eq "TYPE"} {return SIGNAL}
 if {$key eq "DIRECTION"} {return INOUT}
 if {$key eq "REF_NAME"} {return IOBUF}
 if {$key eq "IS_PRIMITIVE"} {return 1}
 error "Unexpected property $key"
}
proc report_file {arguments text} {
 set i [lsearch -exact $arguments -file]
 if {$i<0} {error "Missing report file"}
 set f [open [lindex $arguments [expr {$i+1}]] w];puts $f $text;close $f
}
proc report_clocks {args} {report_file $args {synthetic clocks}}
proc report_route_status {args} {report_file $args {synthetic routed}}
proc report_utilization {args} {report_file $args {synthetic utilization}}
proc report_switching_activity {args} {
 global phase trace
 lappend trace report_$phase
 set tag [dict get {inherited A reset D annotated S} $phase]
 report_file $args "net_0: static probability = 0.5 ($tag) signal rate = 100 ($tag) toggle rate = 50 ($tag)"
}
proc reset_switching_activity {args} {
 global phase trace
 if {$args ne "-all"} {error "Expected full reset"};set phase reset;lappend trace reset
}
proc read_saif {args} {
 global phase trace
 if {$phase ne "reset"} {error "No reset before annotation"}
 if {[lindex $args 0] ne "-strip_path" || [lindex $args 1] ne "ntrpu_ddr_example_activity_tb/dut" || [lindex $args 2] ne "-out_file" || [llength $args]!=5} {error "Invalid SAIF arguments"}
 set f [open [lindex $args 3] w];puts $f {synthetic unmatched};close $f
 set phase annotated;lappend trace annotate
}
proc redirect {args} {report_file $args {synthetic annotation};uplevel 1 [lindex $args end]}
rename exit original_exit
proc exit {code} {set ::exit_code $code}
''')
    t.call('set','test_top','wrong' if bad_top else 'example_top')
    t.call('source',str(script))
    return int(t.getvar('exit_code')),list(t.getvar('trace'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--script',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='ntrpu-mapping-controls-') as tmp:
        root=Path(tmp)/'good';code,trace=simulate(a.script,root)
        expected=['open','report_inherited','reset','report_reset','annotate','report_annotated','report_annotated','close']*2
        assert code==0 and trace==expected,(code,trace)
        bad,short=simulate(a.script,Path(tmp)/'bad-top',True);assert bad==1 and short==['open']
        source='a'*64;inputs={n:dict(remote='/synthetic/'+n,sha256=c*64) for n,c in (('dcp','b'),('prbs_0','c'),('prbs_1','d'))}
        inv=dict(exit_code=0,finished_utc='synthetic',source_sha256=source,inputs=inputs,
            frozen_source_unchanged=True,external_inputs_unchanged=True,power_estimate_executed=False)
        (root/'invocation.json').write_text(json.dumps(inv))
        console='\n'.join(['NTRPU_DDR_SAIF_MAPPING_WINDOW_EXPORTED window=0','NTRPU_DDR_SAIF_MAPPING_WINDOW_EXPORTED window=1','NTRPU_DDR_SAIF_MAPPING_R1_EXPORTED'])+'\n'
        (root/'console.log').write_text(console)
        result=check.review(root,source,inputs);assert not result['errors'],result
        controls=[]
        changes=[('exit','invocation.json',lambda s:s.replace('"exit_code": 0','"exit_code": 1')),
            ('source','invocation.json',lambda s:s.replace(source,'e'*64)),
            ('modified_source','invocation.json',lambda s:s.replace('"frozen_source_unchanged": true','"frozen_source_unchanged": false')),
            ('modified_external','invocation.json',lambda s:s.replace('"external_inputs_unchanged": true','"external_inputs_unchanged": false')),
            ('marker','console.log',lambda s:s.replace('NTRPU_DDR_SAIF_MAPPING_R1_EXPORTED','missing')),
            ('duplicate_marker','console.log',lambda s:s+'NTRPU_DDR_SAIF_MAPPING_R1_EXPORTED\n'),
            ('native_error','console.log',lambda s:s+'ERROR: synthetic\n'),
            ('prefix','output/prbs_0/identity.tsv',lambda s:s.replace('ntrpu_ddr_example_activity_tb/dut','dut')),
            ('top','output/prbs_0/identity.tsv',lambda s:s.replace('example_top','wrong')),
            ('window','output/prbs_1/identity.tsv',lambda s:s.replace('prbs_1','prbs_0')),
            ('different_nets','output/prbs_1/nets.tsv',lambda s:s.replace('net_0','other')),
            ('duplicate_net','output/prbs_0/nets.tsv',lambda s:s+s.splitlines()[1]+'\n'),
            ('empty_annotation','output/prbs_0/annotation.rpt',lambda s:''),
            ('scope','output/prbs_0/scope.txt',lambda s:s.replace('NOT_POWER_QUALIFIED','QUALIFIED'))]
        for label,name,mutation in changes:
            path=root/name;original=path.read_bytes();bad=mutation(original.decode());assert bad.encode()!=original
            path.write_text(bad)
            try:assert check.review(root,source,inputs)['errors'],label
            finally:path.write_bytes(original)
            controls.append(label)
        assert check.review(root,source,inputs)['status']=='DDR_SAIF_MAPPING_R1_EXPORTED_WITH_OPEN_ITEMS'
    report=dict(status='PASS_DDR_SAIF_MAPPING_R1_SYNTHETIC_CONTROLS',tcl_flow=trace,
        rejected=controls,wrong_design_rejected=True,native_tool_executed=False,power_estimate_executed=False)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=report['status'],controls=len(controls))))

if __name__=='__main__':main()
