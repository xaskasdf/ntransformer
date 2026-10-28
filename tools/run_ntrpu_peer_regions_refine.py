"""One bounded post-route physical optimization; clocks and RTL remain unchanged."""
from run_ntrpu_peer_regions_physical import driver
from prepare_ntrpu_peer_commit import once


def refined_driver():
    text=driver()
    text=once(text,'    assert 0 < a.timeout <= 3600','    assert a.stage == "route" and 0 < a.timeout <= 420')
    text=once(text,"donor['stage'] == 'synth'","donor['stage'] == 'route'")
    text=once(text,"review['status'] == 'REGIONS_SYNTHESIS_REVIEWED'",
              "review['status'] == 'REGIONS_ROUTED_INTERNAL_TIMING_OPEN'")
    text=text.replace('suite/synthesized.dcp','suite/routed.dcp')
    text=once(text,"if p.is_file() and p.suffix != '.dcp':",
              "if p.is_file() and (p.name.endswith('_config.tsv') or "
              "p.name in ('clocks.xdc', 'ntrpu_peer_regions_physical.sv')):")
    text=once(text,"baseline = header + baseline[baseline.index('    opt_design\\n'):]",
              "baseline = header + '    phys_opt_design -directive Explore\\n' + "
              "baseline[baseline.index('    write_checkpoint [file join $output routed.dcp]\\n'):]")
    text=once(text,"stage=a.stage, sources=sources,",
              "stage=a.stage, refinement='post_route_explore', sources=sources,")
    return text


if __name__=='__main__':
    exec(compile(refined_driver(),__file__,'exec'),dict(__name__='__main__',__file__=__file__))
