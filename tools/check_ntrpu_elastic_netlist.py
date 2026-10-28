"""Review routed elastic netlist provenance and independent public replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from check_ntrpu_multibuffer_route import digest, require
from check_ntrpu_elastic import replay_external


def review(root,routed_profile,rtl_log,reserve):
    output=root/'output'
    record=json.loads((output/'result.json').read_text())
    require(record['status']=='EXPORTED_PENDING_REVIEW' and record['variant']=='elastic','incomplete/wrong variant')
    require(record['sources']==record['sources_after'],'sources changed')
    for n,h in record['sources'].items(): require(digest(root/'inputs'/n)==h,'input hash '+n)
    for n,h in json.loads((output/'FILES.json').read_text()).items(): require(digest(output/n)==h,'output hash '+n)
    selected=[r for r in record['runs'] if r['reserve']==reserve]
    require(len(selected)==1 and selected[0]['returncode']==0 and not selected[0]['timeout'],'failed simulation')
    folder=output/f'reserve_{reserve}'
    sha=digest(routed_profile/'multibuffer_funcsim.v')
    require(digest(folder/'input_netlist.v')==sha,'wrong routed netlist')
    require([v for n,v in record['netlists'].items() if n.endswith(f'/reserve_{reserve}/multibuffer_funcsim.v')]==[sha],
            'netlist provenance')
    log=(folder/'simulation.log').read_text()
    console=(output/f'reserve_{reserve}.console.log').read_text()
    require(not re.search(r'^\s*(ERROR:|Fatal:|FAIL |CRITICAL WARNING:)',log+'\n'+console,re.M|re.I),'simulation diagnostics')
    require(f'MULTIBUFFER_NETLIST_EXPORTED RESERVE={reserve}' in console,'missing export')
    actual=replay_external(log,(2,reserve,64,16,4),1)
    require(actual==selected[0]['pin_replay'],'downloaded replay differs')
    reference=replay_external(rtl_log.read_text(),(2,reserve,64,16,4),1)
    for key in ('jobs','bytes','completed','foreign_b','error_b'):
        require(actual[key]==reference[key],'RTL/netlist transaction count '+key)
    return dict(status='ELASTIC_ROUTED_NETLIST_PUBLIC_REPLAY_PASS',reserve=reserve,
                external=actual,rtl_external=reference,seconds=selected[0]['seconds'],
                routed_netlist_sha256=sha,
                warnings=dict(Counter(re.findall(r'^WARNING: \[([^]]+)\]',console,re.M))),
                limits=['Functional simulation at 10 ns, no SDF','No internal lease checks in netlist',
                        'Transaction equivalence; cycles may differ due to GSR startup',
                        'No physical PCIe/HBM transport or CDC integration'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('routed_profile',type=Path)
    ap.add_argument('rtl_log',type=Path)
    ap.add_argument('--reserve',type=int,choices=(4,8),default=4)
    args=ap.parse_args()
    result=review(args.run,args.routed_profile,args.rtl_log,args.reserve)
    with (args.run/'REVIEW.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))
