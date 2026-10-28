"""Review a restored, self-contained canonical-clock composition evidence ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
from review_hbm_clocked import review,review_case,review_source

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def review_bundle(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(manifest)|{'MANIFEST.json'}:raise ValueError('Bundle file set changed')
    for name,digest in manifest.items():
        if sha(root/name)!=digest:raise ValueError('Bundle checksum mismatch: '+name)
    selection=json.loads((root/'SELECTION.json').read_text())
    reviews={}
    for name in ('local','xsim','synth'):
        reviews[name]=review(root/selection[name],root/'PACKAGING_AUDIT.json' if name!='local' else None)
        if reviews[name]!=json.loads((root/(name.upper()+'_REVIEW.json')).read_text()):raise ValueError('Review receipt changed')
    negative=root/selection['negative'];receipt=json.loads((negative/'REVIEW.json').read_text())
    if receipt['count']!=12 or len(receipt['checks'])!=12 or not all(c['rejected'] for c in receipt['checks']):raise ValueError('Missing negative checks')
    for p in negative.glob('*.log'):
        try:review_case(p.read_text(),1,0)
        except ValueError:pass
        else:raise ValueError('Negative simulation evidence accepted')
    try:review_source(negative/'modified-source')
    except ValueError:pass
    else:raise ValueError('Early management reset release accepted')
    for name in ('real_rtl_changed','snapshot_changed','unclassified_input'):
        try:review(root/selection['xsim'],negative/(name+'.json'))
        except ValueError:pass
        else:raise ValueError('Invalid packaging recovery accepted')
    process=json.loads((root/'PROCESS_COMPLETION.json').read_text())
    if len(process['jobs'])!=2 or any(j['remaining_group'] for j in process['jobs']):raise ValueError('Private VM processes remain')
    for row in json.loads((root/'PACKAGING_AUDIT.json').read_text()):
        archive=root/Path(row['archive']).name
        if sha(archive)!=row['sha256'] or archive.stat().st_size!=row['bytes']:raise ValueError('VM archive mismatch')
    donor=root/'provenance/20261011-hbm-cdc-physical.zip'
    if sha(donor)!='2b6c9587dbe8acf98dd94c5317169c354e8153e890a3783d4f4e038f2a864c59':raise ValueError('Prior physical evidence changed')
    return dict(status='HBM_CLOCKED_BUNDLE_REVIEWED',files=len(manifest),functional_cases=52,
                canonical_reset_connections=17,negative_checks=12,packaging_recovered=True,
                native_hbm_pcie_ip=False,physical_timing_qualified=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);a=ap.parse_args()
    print(json.dumps(review_bundle(a.root),indent=2))
