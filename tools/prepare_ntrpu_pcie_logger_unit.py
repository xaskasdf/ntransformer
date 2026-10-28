"""Extract actual public-example logger tasks into a small XSIM reproduction."""
import argparse
import hashlib
import json
import pathlib
import re

from patch_ntrpu_pcie_logger import patch_text


def task(text, name):
    matches = re.findall(r'task (?:automatic )?' + name + r';.*?endtask', text, re.S)
    if len(matches) != 1:
        raise ValueError('Expected exactly one task: ' + name)
    return matches[0]


def prepare(example, output):
    output.mkdir(exist_ok=False)
    names = ['usp_pci_exp_usrapp_tx.v', 'usp_pci_exp_usrapp_rx.v', 'usp_pci_exp_usrapp_com.v']
    original = [(example / 'imports' / n).read_text() for n in names]
    patched = patch_text(*original)
    # Invalid-frame checks and parse logic must remain byte-for-byte intact.
    assert task(original[2], 'TSK_PARSE_FRAME') == task(patched[2], 'TSK_PARSE_FRAME')
    manifest = []
    for name, before, after in zip(names, original, patched):
        (output / name).write_text(after)
        manifest.append({'file': name, 'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                         'after_sha256': hashlib.sha256(after.encode()).hexdigest()})
    tasks = [task(patched[2], n) for n in ['TSK_READ_DATA', 'TSK_READ_DATA_512',
             'TSK_DECIPHER_FRAME', 'TSK_NTRPU_LOG_TX']]
    tasks.append(task(original[2], 'TSK_DECIPHER_FRAME').replace('TSK_DECIPHER_FRAME', 'TSK_ORIGINAL_DECIPHER'))
    tasks.append(task(patched[1], 'TSK_BUILD_CQ_TO_PCIE_PKT'))
    (output / 'logger_tasks.svh').write_text('\n\n'.join(tasks) + '\n')
    sync = task(patched[0], 'TSK_TX_SYNCHRONIZE')
    original_sync = sync.replace('task automatic TSK_TX_SYNCHRONIZE;', 'task TSK_ORIGINAL_TX_SYNCHRONIZE;')
    (output / 'logger_sync_tasks.svh').write_text(sync + '\n' + original_sync + '\n')
    (output / 'extraction.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()
    prepare(args.example, args.output)
