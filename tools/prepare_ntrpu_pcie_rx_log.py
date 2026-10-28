"""Extract actual AMD CQ/RC builders and RX reader for the interleaving unit."""
import argparse,hashlib,json
from pathlib import Path
from patch_ntrpu_pcie_logger import patch_text
from patch_ntrpu_pcie_concurrent_log import patch_concurrent
from patch_ntrpu_pcie_rx_log import patch_rx,RX_TASKS
from prepare_ntrpu_pcie_logger_unit import task


def prepare(example,output):
    output.mkdir(exist_ok=False)
    names=['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v']
    original=[(example/'imports'/n).read_text() for n in names]
    tx,rx,com=patch_text(*original)
    tx,com,_=patch_concurrent(tx,com)
    fixed,common,changes=patch_rx(rx,com)
    assert sum(v['calls'] for v in changes if v['operation']=='DATA')==22
    assert sum(v['calls'] for v in changes if v['operation']=='PARSE')==12
    for name in ['TSK_READ_DATA','TSK_PARSE_FRAME','TSK_DECIPHER_FRAME','TSK_3DW']:
        assert task(common,name)==task(com,name)
    (output/'rx_common.svh').write_text(RX_TASKS+'\n'+task(common,'TSK_READ_DATA')+'\n'+task(common,'TSK_3DW'))
    builders=[]
    for channel in ['CQ','RC']:
        name='TSK_BUILD_'+channel+'_TO_PCIE_PKT'
        builders.extend([task(fixed,name),task(rx,name).replace(name,'TSK_LEGACY_'+channel)])
    (output/'rx_builders.svh').write_text('\n'.join(builders))
    for name,text in zip(names,[tx,fixed,common]):(output/name).write_text(text)
    manifest=dict(rx_calls=changes,reader_parser_unchanged=True,callback_rx_only=True,
        files=[dict(name=n,before_sha256=hashlib.sha256(a.encode()).hexdigest(),after_sha256=hashlib.sha256(z.encode()).hexdigest())
               for n,a,z in zip(names,original,[tx,fixed,common])])
    (output/'extraction.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',type=Path);parser.add_argument('output',type=Path)
    a=parser.parse_args();prepare(a.example,a.output)
