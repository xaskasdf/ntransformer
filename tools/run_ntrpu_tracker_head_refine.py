"""One bounded post-route optimization of the unchanged cached-head checkpoint."""
from prepare_ntrpu_peer_commit import once
from run_ntrpu_tracker_head_physical import driver


def refined_driver():
    text=driver()
    text=once(text,'    assert 0 < a.timeout <= 3600','    assert a.stage == "route" and 0 < a.timeout <= 420')
    text=once(text,"donor['stage'] == 'synth'","donor['stage'] == 'route'")
    text=once(text,"review['status'] == 'TRACKER_HEAD_SYNTHESIS_REVIEWED'",
              "review['status'] == 'TRACKER_HEAD_ROUTED_INTERNAL_TIMING_OPEN'")
    text=text.replace('suite/synthesized.dcp','suite/routed.dcp')
    text=once(text,"if p.is_file() and p.suffix != '.dcp':",
              "if p.is_file() and (p.name.endswith('_config.tsv') or "
              "p.name in ('clocks.xdc','ntrpu_tracker_head_physical.sv','head_registers.tsv')):")
    text=once(text,"baseline = header + baseline[baseline.index('    opt_design\\n'):]",
              "baseline = header + '    phys_opt_design -directive Explore\\n' + "
              "baseline[baseline.index('    write_checkpoint [file join $output routed.dcp]\\n'):]")
    return once(text,'stage=a.stage, sources=sources,',
                "stage=a.stage, refinement='post_route_explore', sources=sources,")


if __name__=='__main__':
    exec(compile(refined_driver(),__file__,'exec'),dict(__name__='__main__',__file__=__file__))
