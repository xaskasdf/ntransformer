"""Review public mapped addresses, payload and ownership on the routed netlist."""
import argparse
import json
from pathlib import Path
from prepare_ntrpu_peer_commit import once
from check_ntrpu_peer_regions_unit import ledger_module, physical, region_map


def auditor():
    text=Path(__file__).with_name('check_ntrpu_peer_shared_routed.py').read_text(encoding='utf-8')
    text=text.replace('SHARED_ROUTED_INTERNAL_TIMING_PASS','REGIONS_ROUTED_INTERNAL_TIMING_PASS')
    text=text.replace('SHARED_ROUTED_PUBLIC_REVIEWED','REGIONS_ROUTED_PUBLIC_REVIEWED')
    text=once(text,"'AR','DATA','DMA','ACK'}","'AR','DATA','DMA','ACK','MAP'}")
    text=once(text,"        for p in rows['CFG']:",
              "        require(len(rows['MAP'])==len(rows['CFG']),'map/config pairing')\n        for p in rows['CFG']:")
    text=once(text,"new=lease(p[0]);require(not bad_config(new,c)",
              "new=lease(p[0]);new['map']=region_map(rows['MAP'][0][0]);require(not bad_config(new,c)")
    text=once(text,"item['cfg']['src']+item['r']['src']+item['native']",
              "physical(item['cfg'],item['r']['src'],item['r']['size'])+item['native']")
    text=once(text,"item['cfg']['src']+r['src']+item['beats']*64",
              "physical(item['cfg'],r['src'],r['size'])+item['beats']*64")
    text=once(text,'(a>>16)^0x5b','(a>>16)^(a>>24)^(a>>32)^0x5b')
    text=once(text,"stats['native_ar']+=1","stats['native_ar']+=1;stats['pc16_ar' if address&(1<<33) else 'pc0_ar']+=1")
    text=once(text,"stats['rejections']==5","stats['rejections']==6 and stats['pc0_ar']>0 and stats['pc16_ar']>0")
    text=once(text,"'SP REQ ','SP REJECT '","'SP MAP ','SP REQ ','SP REJECT '")
    text=once(text,'    for name,changed in variants:', '''    index=next(i for i,s in enumerate(lines) if s.startswith('SF AR ') and int(s.split()[5])&(1<<33))
    p=lines[index].split();p[5]=str(int(p[5])&~(1<<33))
    variants.append(('truncate_stack_bit',lines[:index]+[' '.join(p)]+lines[index+1:]))
    for name,changed in variants:''')
    namespace=dict(__name__='mapped_public_audit',__file__=__file__,physical=physical,region_map=region_map)
    exec(compile(text,__file__,'exec'),namespace)
    ledger=ledger_module(Path(__file__).resolve().parents[1])
    namespace.update(bad_config=ledger['bad_config'],rejection=ledger['rejection'])
    return namespace


def review(out):
    api=auditor(); checked=api['review'](out); need=api['require']; sha=api['sha']
    meta=json.loads((out/'suite/COMPARISON.json').read_text(encoding='utf-8'))
    need(meta['sole_dut']=='ntrpu_peer_regions_physical' and meta['mapped_regions']==2 and
         meta['remap_generations']==[1,2] and not meta['native_hbm_controller'],'mapped public scope')
    need(meta['generated_bench_sha256']==sha(out/'suite/tb_ntrpu_peer_shared.sv'),'bench binding')
    from prepare_ntrpu_peer_regions_routed import public_bench
    need((out/'suite/tb_ntrpu_peer_shared.sv').read_text(encoding='utf-8')==public_bench(out.parent/'repo'),
         'unexpected public bench changes')
    checked.update(native_hbm_controller=False,public_ports_only=True,mapped_regions=2,remap_generations=[1,2])
    return checked


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
