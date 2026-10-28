"""Extract real patched root-port tasks for a bounded concurrent logging test."""
import argparse, hashlib, json
from pathlib import Path
from patch_ntrpu_pcie_logger import patch_text
from patch_ntrpu_pcie_concurrent_log import patch_concurrent, CHANNEL_STATE, CHANNEL_TASK
from prepare_ntrpu_pcie_logger_unit import task


def prepare(example, output):
    output.mkdir(exist_ok=False)
    names=['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v']
    original=[(example/'imports'/n).read_text() for n in names]
    tx, rx, com=patch_text(*original)
    legacy=task(tx,'TSK_TX_SYNCHRONIZE').replace('TSK_TX_SYNCHRONIZE','TSK_SHARED_TX_SYNCHRONIZE')
    fixed, common, changes=patch_concurrent(tx,com)
    assert task(common,'TSK_PARSE_FRAME')==task(original[2],'TSK_PARSE_FRAME')
    assert task(common,'TSK_DECIPHER_FRAME')==task(com,'TSK_DECIPHER_FRAME')
    (output/'concurrent_sync.svh').write_text(task(fixed,'TSK_TX_SYNCHRONIZE')+'\n'+legacy+'\n')
    (output/'concurrent_logger.svh').write_text(CHANNEL_STATE+CHANNEL_TASK+'\n'+
        '\n'.join(task(common,n) for n in ['TSK_NTRPU_LOG_TX','TSK_READ_DATA_512','TSK_3DW'])+'\n'+
        task(com,'TSK_3DW').replace('TSK_3DW','TSK_UNGUARDED_3DW'))
    for name,text in zip(names,[fixed,rx,common]):(output/name).write_text(text)
    guarded=task(common,'TSK_3DW').replace(
        "if (!txrx) `TX_TASKS.TSK_SET_READ_DATA(4'hf,dword_data); // RX completion only",
        "`TX_TASKS.TSK_SET_READ_DATA(4'hf,dword_data); // send the data to the tx_app")
    assert guarded==task(com,'TSK_3DW')
    manifest=dict(producer_changes=changes,parser_dispatch_unchanged=True,completion_callback_rx_only=True,
        files=[dict(name=n,before_sha256=hashlib.sha256(a.encode()).hexdigest(),
                    after_sha256=hashlib.sha256(z.encode()).hexdigest())
               for n,a,z in zip(names,original,[fixed,rx,common])])
    (output/'extraction.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',type=Path);parser.add_argument('output',type=Path)
    a=parser.parse_args();prepare(a.example,a.output)
