"""Complete read-only synthesis audits on the exact checkpoint of a failed audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(original,checkpoint,output):
    if output.exists():raise ValueError('Fresh output required')
    output.mkdir(parents=True)
    for name in ('audit_hbm_capture_physical.tcl','audit_hbm_ready_physical.tcl',
                 'audit_hbm_reference_ownership.tcl','run_hbm_native_join.py'):
        shutil.copy2(original/name,output/name)
    here=Path(__file__).resolve().parent
    for name in ('audit_hbm_controls_physical.tcl','prepare_hbm_controls_recovery.py'):
        shutil.copy2(here/name,output/name)
    entry='''set_param general.maxThreads 2
if {[catch {
 lassign $argv checkpoint out
 if {[llength $argv]!=2 || [file exists $out]} {error "CHECKPOINT NEW_OUTPUT required"}
 set source [file dirname [info script]]
 file mkdir $out
 open_checkpoint $checkpoint
 if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" || [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong complete-board checkpoint"}
 if {[llength [get_cells -hier -quiet -filter {IS_BLACKBOX == 1}]] || [llength [get_cells -hier -quiet -filter {REF_NAME =~ LD*}]]} {error "Unresolved black box or latch"}
 source $source/audit_hbm_capture_physical.tcl
 source $source/audit_hbm_ready_physical.tcl
 source $source/audit_hbm_controls_physical.tcl
 cp_probe $out/response_capture 0
 rp_probe $out/registered_ready 0
 pp_probe $out/control_predictors 0 1
 check_timing -verbose -file $out/check_timing.rpt
 report_cdc -details -file $out/cdc.rpt
 report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
 report_drc -file $out/drc.rpt
 report_utilization -hierarchical -file $out/utilization.rpt
 report_clocks -file $out/clocks.rpt
 write_xdc -constraints all $out/effective.xdc
 source $source/audit_hbm_reference_ownership.tcl
 ro_audit $out
 set f [open $out/native_instances.tsv {WRONLY CREAT EXCL}];puts $f "cell\\treference"
 foreach name {ntrpu_bridge_i hbm_binding/memory clock_generator core} {puts $f "$name\\t[get_property REF_NAME [get_cells $name]]"};close $f
 set f [open $out/ports.tsv {WRONLY CREAT EXCL}];puts $f "port\\tdirection"
 foreach p [lsort [get_ports]] {puts $f "$p\\t[get_property DIRECTION [get_ports $p]]"};close $f
 set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f CONTROLS_SYNTHESIS_RECOVERY_EXPORTED;close $f
 close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
'''
    (output/'inspect_hbm_native_join.tcl').write_text(entry,newline='\n')
    record=dict(checkpoint_sha256=sha(checkpoint),read_only=True,
                original_source_files={p.name:sha(p) for p in original.iterdir() if p.is_file()},
                reason='Return RID registers were pruned; prove constant or disconnected interfaces without changing the netlist',
                prepared_files={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'CONTROLS_SYNTH_RECOVERY.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(checkpoint_sha256=record['checkpoint_sha256'],read_only=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('original','checkpoint','output'):p.add_argument(key,type=Path)
    a=p.parse_args();prepare(a.original,a.checkpoint,a.output)
