"""Bounded local RTL simulation/replay. Output directory must be new."""
import argparse
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES=("fpga/rtl/ntrpu_buffer_pool_allocator.sv","fpga/rtl/ntrpu_peer_multibuffer.sv",
         "fpga/tb/tb_ntrpu_peer_multibuffer.sv","tools/check_ntrpu_peer_multibuffer.py",
         "tools/run_ntrpu_peer_multibuffer.py")
PROFILES={"small4":(2,4,4,16,4),"full4":(2,4,64,16,4),"full8":(2,8,64,16,4),
          "one_pending":(2,4,4,16,1),"odd_slots":(3,3,4,5,4),
          "single":(1,0,4,3,2),"one_beat":(2,0,1,3,2),"no_reserve":(2,0,64,5,4)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output",type=Path)
    ap.add_argument("--iverilog",default="iverilog");ap.add_argument("--vvp",default="vvp")
    ap.add_argument("--ivl-base")
    args=ap.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    snap=out/"inputs"; hashes={}
    for name in SOURCES:
        dest=snap/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    sys.path.insert(0,str(snap/"tools"))
    from check_ntrpu_peer_multibuffer import replay
    def command(argv,name):
        start=time.perf_counter(); r=subprocess.run(argv,capture_output=True,text=True,cwd=out,timeout=45)
        (out/(name+".stdout.log")).write_text(r.stdout,encoding="utf-8")
        (out/(name+".stderr.log")).write_text(r.stderr,encoding="utf-8")
        (out/(name+".command.json")).write_text(json.dumps({"argv":argv,"exit_code":r.returncode,
            "seconds":time.perf_counter()-start},indent=2)+"\n",encoding="utf-8")
        return r
    def compile_rtl(name,profile,rtl):
        exe=out/(name+".vvp");argv=[args.iverilog]
        if args.ivl_base:argv += ["-B",args.ivl_base]
        argv += ["-g2012","-Wall","-s","tb_ntrpu_peer_multibuffer"]
        for key,value in zip(("CHANNELS","RESERVE_BANKS","BANK_BEATS","DESCRIPTORS","MAX_OUTSTANDING"),profile):
            argv += [f"-Ptb_ntrpu_peer_multibuffer.{key}={value}"]
        argv += ["-o",str(exe),str(snap/SOURCES[0]),str(rtl),str(snap/SOURCES[2])]
        if command(argv,"compile_"+name).returncode:raise RuntimeError("compile failed: "+name)
        return exe
    started=time.perf_counter();results=[]
    for name,profile in PROFILES.items():
        exe=compile_rtl(name,profile,snap/SOURCES[1])
        for seed in (1,7):
            r=command([args.vvp,str(exe),f"+SEED={seed}"],f"{name}_seed{seed}")
            if r.returncode:raise RuntimeError(f"simulation failed: {name} seed {seed}")
            results.append({"name":name,"profile":profile,"seed":seed,**replay(r.stdout,profile,seed)})
    original=(snap/SOURCES[1]).read_text()
    mutants={
      "corrupt_payload":("payload[fill_index[RAM_BITS-1:0]] <= data;","payload[fill_index[RAM_BITS-1:0]] <= data ^ 512'd1;","payload mismatch"),
      "early_release":("assign finish_w[g] = pop_w[g] && last0[g];","assign finish_w[g] = m_wvalid[g] && last0[g];","W changed under backpressure"),
      "ignore_pending_limit":("outstanding[g] < MAX_OUTSTANDING;","outstanding[g] < DESCRIPTORS;","outstanding overflow"),
      "early_completion":("state[g*DESCRIPTORS+result_head[g]] == DONE;","state[g*DESCRIPTORS+result_head[g]] >= WAIT_B;",None),
      "foreign_discharges":(" && m_bid[g*4 +:4] == 0;",";",None),
    }
    negatives=[]
    for name,(old,new,expected_fatal) in mutants.items():
        if original.count(old)!=1:raise RuntimeError("mutation anchor: "+name)
        rtl=out/(name+".sv");rtl.write_text(original.replace(old,new))
        profile=PROFILES["small4"];exe=compile_rtl(name,profile,rtl)
        r=command([args.vvp,str(exe),"+SEED=1"],name)
        reason=None
        try:replay(r.stdout,profile,1,partial=True)
        except RuntimeError as e:reason=str(e)
        if reason is None and expected_fatal and expected_fatal in r.stdout and "FATAL:" in r.stdout:
            reason="bench assertion: "+expected_fatal
        if reason is None:raise RuntimeError("mutation escaped semantic checks: "+name)
        negatives.append({"name":name,"status":"DETECTED","reason":reason,"exit_code":r.returncode})
    report={"status":"ISOLATED_MULTIBUFFER_RTL_PASS_NOT_INTEGRATED","created_utc":datetime.now(timezone.utc).isoformat(),
            "wall_seconds":time.perf_counter()-started,"input_sha256":hashes,"runs":results,"mutations":negatives}
    (out/"REVIEW.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"runs":len(results),"jobs":sum(x['jobs'] for x in results),
        "bytes_checked":sum(x['bytes_checked'] for x in results),"mutants":len(negatives),"seconds":report['wall_seconds']},indent=2))

if __name__=="__main__":main()
