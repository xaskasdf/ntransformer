"""Reuse the frozen mapped physical flow with a newly tested tracker alias."""
from pathlib import Path
from prepare_ntrpu_peer_commit import once
from run_ntrpu_peer_regions_physical import driver as regions_driver
from prepare_ntrpu_tracker_head import verify_profile


def driver():
    text=regions_driver().replace('peer_regions_physical','tracker_head_physical')
    text=text.replace('REGIONS_PHYSICAL','TRACKER_HEAD_PHYSICAL').replace(
        'REGIONS_SYNTHESIS_REVIEWED','TRACKER_HEAD_SYNTHESIS_REVIEWED')
    text=once(text,"review['status'] == 'REGIONS_NATIVE_TRAFFIC_REVIEWED' and review['native_hbm_controller']",
              "review['status'] == 'TRACKER_HEAD_PUBLIC_REVIEWED' and not review['native_hbm_controller']")
    text=once(text,"review['ownership']['counters']['successful_retire'] == 16",
              "review['counters']['successful_results'] == 27")
    anchor='    report_utilization -file [file join $output synthesis_utilization.rpt]'
    # Checked before any physical result is exported: this flow must use the
    # embedded candidate, not merely hash an unused standalone source file.
    proof='''    set heads [get_cells -hier -regexp {(^|.*/)tracker/head_length_reg\[[0-3]\]}]
    if {[llength $heads]!=8} {error "Candidate head register inventory is not eight"}
    set f [open [file join $output head_registers.tsv] w]
    foreach c $heads {puts $f "$c\\t[get_property REF_NAME $c]"}
    close $f
'''
    insertion="    if a.stage == 'synth':\n        baseline = once(baseline, "+repr(anchor)+", "+repr(proof+anchor)+")\n"
    return once(text,'    # Generated Tcl lives in output;',insertion+'    # Generated Tcl lives in output;')


if __name__=='__main__':
    verify_profile(Path(__file__).resolve().parents[1])
    exec(compile(driver(),__file__,'exec'),dict(__name__='__main__',__file__=__file__))
