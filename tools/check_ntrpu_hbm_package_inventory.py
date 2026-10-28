"""Review the installed package database and a deliberately restricted bank budget."""
import argparse,csv,json,re
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
from check_ntrpu_read_return import sha

def properties(path):
    result=defaultdict(dict)
    for row in csv.DictReader(path.open(),delimiter='\t'):
        obj,prop,value=(row[k] for k in ('object','property','value'))
        assert prop not in result[obj];result[obj][prop]=value
    return result

def bank_budget(banks,regions,width):
    hp=sorted(int(n) for n,p in banks.items() if p['BANK_TYPE']=='BT_HIGH_PERFORMANCE')
    candidates=[]
    for group in combinations(hp,width):
        rows=[regions[str(n)] for n in group]
        if list(group)!=list(range(group[0],group[0]+width)):continue
        if len({row['slrs'] for row in rows})!=1:continue
        if len({re.fullmatch(r'X(\d+)Y\d+',row['regions'])[1] for row in rows})!=1:continue
        candidates.append(group)
    feasible=[c for count in range(len(candidates)+1) for c in combinations(candidates,count)
              if len({n for group in c for n in group})==width*len(c)]
    largest=max(map(len,feasible))
    return dict(banks_per_channel=width,candidates=candidates,maximum_disjoint_channels=largest,
                maximal_choices=[c for c in feasible if len(c)==largest])

def review(out):
    r=json.loads((out/'result.json').read_text());assert r['experiment']=='hbm_package_inventory'
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and r['source_sha256']==r['source_after']
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    assert 'NTRPU_HBM_PACKAGE_INVENTORY_COMPLETE' in (out/'console.log').read_text()
    part=properties(out/'part.tsv');assert list(part)==['xcvu47p-fsvh2892-2-e'];p=next(iter(part.values()))
    banks=properties(out/'banks.tsv');pins=properties(out/'package.tsv');sites=properties(out/'sites.tsv')
    regions={r['bank']:r for r in csv.DictReader((out/'bank_regions.tsv').open(),delimiter='\t')}
    assert set(banks)==set(regions) and len(pins)==int(p['IO_PIN_COUNT'])==2892
    hp={n:v for n,v in banks.items() if v['BANK_TYPE']=='BT_HIGH_PERFORMANCE'};assert set(map(int,hp))==set(range(64,76))
    counts=Counter(v['BANK'] for v in pins.values() if v['IS_GENERAL_PURPOSE']=='1')
    assert set(counts)==set(hp) and all(n==52 for n in counts.values()) and sum(counts.values())==int(p['AVAILABLE_IOBS'])==624
    for bank in hp:
        assert regions[bank]['slrs']=='SLR'+hp[bank]['SLR_INDEX']
        assert regions[bank]['regions']=='X4Y'+str(int(bank)-64)
    kinds=Counter(v['SITE_TYPE'] for v in sites.values());assert kinds['GTYE4_CHANNEL']==int(p['GB_TRANSCEIVERS'])==96
    pcie={n:v['SITE_TYPE'] for n,v in sites.items() if v['SITE_TYPE'].startswith('PCIE')}
    assert len(pcie)==int(p['PCI_BUSES'])==6 and kinds['HBM_REF_CLK']==2
    budgets=[bank_budget(banks,regions,width) for width in (3,4)]
    assert [v['maximum_disjoint_channels'] for v in budgets]==[3,3]
    return dict(status='HBM_PACKAGE_INVENTORY_REVIEWED',result_sha256=sha(out/'result.json'),part=next(iter(part)),
                package_pins=len(pins),hp_banks={n:dict(pins=counts[n],**regions[n]) for n in hp},
                site_types=dict(kinds),pcie_sites=pcie,restricted_bank_budgets=budgets,
                assumptions=['exclusive banks per channel','contiguous banks','single column and SLR per channel'],
                four_channel_128_gib_candidate_fits_under_these_assumptions=False,
                memory_ip_generated=False,joint_pinout_qualified=False,board_qualified=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
