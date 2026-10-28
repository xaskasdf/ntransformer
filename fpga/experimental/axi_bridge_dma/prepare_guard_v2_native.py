"""Freeze a V2 binding for the existing private AMD native clock-loss bench."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from prepare_clock_loss_native import BASE_SHA, fixture

HERE=Path(__file__).resolve().parent


def prepare(output):
    base=(HERE/'ntrpu_axi_bridge_native.sv').read_bytes()
    if hashlib.sha256(base).hexdigest()!=BASE_SHA:raise ValueError('Native base changed')
    output.mkdir(parents=True,exist_ok=False)
    names=['ntrpu_axi_bridge_native.sv','ntrpu_axi_bridge_guard_v2.sv','ntrpu_axi_bridge_writer.sv',
           'ntrpu_axi_bridge_mapper.sv','ntrpu_axi_bridge_endpoint_client.sv','clock_loss_scenario.svh',
           'prepare_clock_loss_native.py','prepare_guard_v2_native.py','run_clock_loss_native.py',
           'run_clock_loss_trace.tcl','review_clock_loss_native.py','review_native.py','check_clock_loss_reviewer.py']
    for name in names:shutil.copy2(HERE/name,output/name)
    shutil.copy2(HERE.parents[1]/'rtl/ntrpu_xdma_local_master.sv',output/'ntrpu_xdma_local_master.sv')
    # The unchanged runner uses this filename; its module and fixture binding
    # explicitly remain V2. This alias exists only inside the frozen experiment.
    shutil.copy2(HERE/'ntrpu_axi_bridge_guard_v2.sv',output/'ntrpu_axi_bridge_guard.sv')
    bench=fixture(base)
    if bench.count('ntrpu_axi_bridge_guard guard(')!=1:raise ValueError('Unexpected binding')
    (output/'ntrpu_axi_bridge_clock_loss.sv').write_text(
        bench.replace('ntrpu_axi_bridge_guard guard(','ntrpu_axi_bridge_guard_v2 guard('),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    info=dict(status='PREPARED_NOT_NATIVE_VALIDATED',variant='ntrpu_axi_bridge_guard_v2',files=hashes,
              native_base_sha256=BASE_SHA,guard_sha256=hashes['ntrpu_axi_bridge_guard_v2.sv'],
              runner_filename_alias={'ntrpu_axi_bridge_guard.sv':'ntrpu_axi_bridge_guard_v2.sv'})
    (output/'PREPARED.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    return info


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(prepare(ap.parse_args().output)))
