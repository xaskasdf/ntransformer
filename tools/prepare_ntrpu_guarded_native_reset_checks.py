"""Explicitly adapt the qualified loss ledger to held native PERST recovery."""
from pathlib import Path
from prepare_ntrpu_guarded_native import once


def loss(text):
    start=text.index('def review_loss(text):')
    end=text.index('\n\ndef review(text):',start)
    text=text[start:end].replace('def review_loss(text):','def review(text):')
    for a,z in [('GUARDED_MIG_PASS','NTRD_NATIVE_RESET_RETENTION_PASS'),
                ('GUARDED_MIG_HEALTHY_END','NTRD_NATIVE_RESET_HEALTHY_END'),
                ('GUARDED_MIG_EVENT','NTRD_NATIVE_RESET_EVENT'),
                ('GUARDED_MIG_LOSS','NTRD_NATIVE_RESET_LOSS'),
                ('GUARDED_MIG_MGMT','NTRD_GUARD_NATIVE')]:text=text.replace(a,z)
    text=once(text,"['ready','link_loss','release_memory_b']",
        "['ready','perst_assert','endpoint_down','release_memory_b','memory_verified','perst_release','link_return']")
    text=once(text,"    ready,loss,release=map(lambda x:int(x['ps']),events_)",
        "    ready,loss,endpoint_down,release,verified,perst_release,link_return=map(lambda x:int(x['ps']),events_)\n    require(ready<loss<=endpoint_down<release<verified<=perst_release<link_return<int(end['ps']),'actual native reset stages')\n    require(end['coordination']=='blocked','missing coordinator is explicit')")
    text=once(text,"    faulted=[x for x in mgmt if x['state']=='8']", "    faulted=[x for x in mgmt if x['state']=='6']")
    text=once(text,"x['hrst']==x['srst']=='1' and x['loss']==x['ack']=='1' for x in faulted),'faulted transport restored with reuse closed'",
        "x['hrst']==x['srst']==x['held']==x['ready']=='0' and x['req']==x['loss']==x['ack']=='1' for x in faulted),'transport remains held awaiting an actual coordinator'")
    text=once(text,"groups==[1,0,1],'one guarded runtime reset '","groups==[1,0],'one guarded runtime reset kept asserted '")
    text=once(text,"status='PASS_NTRD_NATIVE_RESET_LOSS_RETENTION'","status='PASS_NATIVE_RESET_RETENTION_PEER_COORDINATION_BLOCKED'")
    return '''"""Native reset variant of the independently qualified ownership/DDR ledger."""
import re
from collections import Counter
from check_ntrpu_pcie_fragment_pool import fields,require


'''+text+'\n'


def periodic(text):
    start=text.index('def periodic(text):')
    text=text[start:].replace('def periodic(text):','def review(text):',1)
    start=text.index("    finish=re.findall(")
    end=text.index('    result=[]',start)
    text=text[:start]+'''    # This prefix did not terminate the simulator. Its endpoint is the actual
    # healthy NTRD/native boundary emitted before the next job starts.
    endings=[fields(x[len('NTRD_NATIVE_PASS '):]) for x in text.splitlines() if x.startswith('NTRD_NATIVE_PASS ')]
    need(len(endings)==1,'one actual healthy native boundary')
    finish_ps=int(endings[0]['ps'])
'''+text[end:]
    return '''"""Frozen periodic lifecycle checks scoped to an observed healthy boundary."""
import re
from check_ntrpu_multibuffer_periodic import review_periodic
from check_ntrpu_pcie_ddr_periodic import WARNING,WARNING_DELAY,BEAT_PS,records
from check_ntrpu_pcie_fragment_pool import fields,require as need


'''+text


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    products={
        'check_ntrpu_guarded_native_reset_loss.py':loss((root/'check_ntrpu_guarded_mig_r3.py').read_text()),
        'check_ntrpu_guarded_native_prefix_periodic.py':periodic((root/'check_ntrpu_guarded_native_core.py').read_text())}
    assert all(not (root/n).exists() for n in products)
    for name,value in products.items():
        with (root/name).open('x',newline='\n') as f:f.write(value)
