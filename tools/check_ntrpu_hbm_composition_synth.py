"""Audit composition synthesis without claiming placement or timing closure."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--simulation',type=Path,required=True);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();out=a.output;sim=a.simulation
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((out/'result.json').read_text());files=json.loads((out/'FILES.json').read_text())
    old=json.loads((sim/'result.json').read_text());accepted=json.loads((sim.parent/'REVIEW.json').read_text())
    assert all(sha(out/n)==h for n,h in files.items())
    assert r['stage']=='synth' and r['experiment']=='hbm_transport' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['sources']==r['sources_after']==old['sources']==old['sources_after']
    assert r['generated']==r['generated_after'] and old['generated']==old['generated_after']
    # JSON key ordering follows filesystem enumeration; all values must match.
    hdl=lambda values:{n:h for n,h in values.items() if n!='COMPOSITION.json'}
    assert hdl(r['generated'])==hdl(old['generated'])
    assert json.loads((out/'suite/generated/COMPOSITION.json').read_text())==json.loads((sim/'suite/generated/COMPOSITION.json').read_text())
    assert r['prior']==dict(result=sha(sim/'result.json'),review=sha(sim.parent/'REVIEW.json'))
    assert accepted['status']=='HBM_TRANSPORT_FUNCTIONAL_PASS'
    props=lambda p:dict(l.split('\t',1) for l in p.read_text().splitlines() if not l.startswith('IP_FILE\t'))
    for name in ('ip_config.tsv','hbm_width_config.tsv','hbm_protocol_config.tsv'):
        assert props(out/'suite'/name)==props(sim/'suite'/name)
    metrics=dict(l.split(None,1) for l in (out/'suite/metrics.txt').read_text().splitlines())
    assert metrics['LATCHES']==metrics['BLACKBOXES']=='0'
    assert metrics['CLOCKS']=='1' and float(metrics['CLOCK_PERIOD'])==5
    log=(out/'console.log').read_text();assert 'NTRPU_HBM_TRANSPORT_COMPLETE synth' in log
    assert not re.search(r'^\s*ERROR:',log,re.M)
    drc=(out/'suite/drc.rpt').read_text();assert re.search(r'Checks found:\s+0',drc)
    check=(out/'suite/check_timing.rpt').read_text()
    checks=dict((n,int(v)) for n,v in re.findall(r'checking (\w+) \((\d+)\)',check))
    assert all(checks[n]==0 for n in ('no_clock','constant_clock','unconstrained_internal_endpoints','multiple_clock','loops','latch_loops'))
    utilization=(out/'suite/utilization.rpt').read_text()
    line=next(l for l in utilization.splitlines() if re.match(r'\| ntrpu_hbm_transport\s+\|',l))
    fields=[x.strip() for x in line.split('|')[1:-1]]
    counts=dict(zip(('total_lut','logic_lut','lutram','srl','ff','ramb36','ramb18','uram','dsp'),map(int,fields[2:])))
    counts['bram36_equivalent']=counts['ramb36']+counts['ramb18']/2
    paths=(out/'suite/internal_setup.rpt').read_text()
    source=re.search(r'Source:\s+(.+)',paths)[1];destination=re.search(r'Destination:\s+(.+)',paths)[1]
    result=dict(status='SYNTHESIS_AUDITED_NOT_TIMING_QUALIFIED',seconds=r['seconds'],resources=counts,
        setup_ns=float(metrics['INTERNAL_SETUP']),hold_ns=float(metrics['INTERNAL_HOLD']),
        worst_path=dict(source=source.strip(),destination=destination.strip()),
        internal_timing_coverage_complete=True,external_input_delays_missing=checks['no_input_delay'],
        external_output_delays_missing=checks['no_output_delay'],drc_violations=0,
        checkpoint_sha256=sha(out/'suite/composition_synth.dcp'),checker_sha256=sha(Path(__file__)),
        placed=False,routed=False,qualified_timing=False,hard_hbm=False,
        scope='Pre-placement OOC estimates on VU5P; no physical clock buffer or external I/O timing qualification')
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
