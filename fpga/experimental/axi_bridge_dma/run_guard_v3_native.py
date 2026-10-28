"""Run a bounded, isolated native-source synthesis on the Vivado VM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from prepare_guard_v3 import prepare


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def run(a):
    allowed=Path('/home/ubuntu/vivado/ntrpu-side').resolve();root=a.output.resolve();example=a.example.resolve()
    if os.name!='posix' or not root.is_relative_to(allowed) or not example.is_relative_to(allowed):raise ValueError('Private VM namespace required')
    reference=a.reference_checkpoint.resolve()
    if not reference.is_relative_to(allowed):raise ValueError("Private reference required")
    reference_digest=sha(reference)
    archive=Path(str(root)+'.tar.gz')
    if root.exists() or archive.exists() or not a.settings.is_file():raise ValueError('New output and installed Vivado required')
    seed_names=['imports/xilinx_axi_pcie4_ep.v','imports/xilinx_axi_pcie_ep_app.v','imports/xilinx_xdma_pcie_x0y0.xdc']
    seed_names += [f'ntrpu_bridge_ex.srcs/sources_1/ip/{n}/{n}.xci' for n in ('ntrpu_bridge','axi_bram_ctrl_0')]
    seed_hashes={n:sha(example/n) for n in seed_names}
    here=Path(__file__).resolve().parent
    names=['ntrpu_axi_bridge_guard_v3.sv','ntrpu_axi_bridge_guarded_client_v3.sv','ntrpu_axi_bridge_endpoint_client.sv',
           'ntrpu_axi_bridge_writer.sv','ntrpu_axi_bridge_mapper.sv','ntrpu_xdma_local_master.sv',
           'prepare_guard_integration.py','prepare_guard_v3.py','audit_guard_v3.tcl','run_guard_v3_native.py','synth_guard_v3_native.tcl']
    hashes={n:sha(here/n) for n in names}
    root.mkdir(parents=True);source=root/'source';source.mkdir();work=root/'work';work.mkdir()
    for name in names:shutil.copy2(here/name,source/name)
    shutil.copy2(example/'imports/xilinx_axi_pcie_ep_app.v',source/'xilinx_axi_pcie_ep_app.v')
    seed=root/'seed';seed.mkdir()
    for name in seed_names:
        target=seed/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(example/name,target)
    save(root/'SEED_FILES.json',seed_hashes)
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    started=time.monotonic();error=None;code=None;expired=False
    try:
        prepare(example,source/'ntrpu_axi_bridge_guarded_client_v3.sv',root/'prepared')
        shutil.copy2(root/'prepared/ntrpu_axi_bridge_guard_native.sv',source/'ntrpu_axi_bridge_guard_native.sv')
        args=['vivado','-mode','batch','-source',str(source/'synth_guard_v3_native.tcl'),'-journal',str(root/'vivado.jou'),
              '-log',str(root/'vivado.log'),'-tclargs',str(source),str(example),str(root/'output'),str(reference)]
        command='source '+shlex.quote(str(a.settings))+' && exec '+shlex.join(args)
        with (root/'synthesis.log').open('w') as log:
            p=subprocess.Popen(['bash','-c',command],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'own_process.json',dict(pid=p.pid,pgid=p.pid,timeout=1800,command=args))
            try:code=p.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
        if code or expired or not (root/'output/STATUS.txt').exists():raise ValueError('Native synthesis failed; inspect preserved output')
    except (OSError,ValueError,RuntimeError) as exc:error=str(exc)
    unchanged=sha(reference)==reference_digest and seed_hashes=={n:sha(example/n) for n in seed_names} and hashes=={n:sha(here/n) for n in names}
    if not unchanged:error='Input or seed changed'
    result=dict(status='V3_SHARED_NATIVE_SYNTHESIS_EXPORTED' if error is None else 'FAILED',error=error,returncode=code,
                timed_out=expired,seconds=time.monotonic()-started,inputs_unchanged=unchanged,sources=hashes,seed_sources=seed_hashes,reference_sha256=reference_digest,
                cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),board_timing_closed=False,physical_route_executed=False)
    save(root/'RESULT.json',result)
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(work) and '.Xil' not in p.parts]
    save(root/'FILES.json',{p.relative_to(root).as_posix():sha(p) for p in paths})
    with tarfile.open(archive,'x:gz') as tar:
        for path in paths+[root/'FILES.json']:tar.add(path,arcname=path.relative_to(root).as_posix())
    result.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    print(json.dumps(result),flush=True)
    return error is None


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--reference-checkpoint',type=Path,required=True);ap.add_argument('--example',type=Path,required=True);ap.add_argument('--settings',type=Path,required=True)
    ap.add_argument('--execute-native',action='store_true');a=ap.parse_args()
    if not a.execute_native:ap.error('--execute-native required')
    raise SystemExit(0 if run(a) else 1)
