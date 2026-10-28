import json
from pathlib import Path
import shutil
import subprocess
import threading
from run import ROOT, PREFIX, MODEL, command, status, save, sha, admitted, capture

assert json.loads((ROOT/'status.json').read_text())['state']=='complete'
initial=ROOT/'initial-debug';initial.mkdir(exist_ok=True)
for path in ROOT.iterdir():
    if path.is_dir() and (path.name.startswith('p32-') or path.name.startswith('p128-') or path.name.startswith('p512-') or path.name.startswith('pilot-')):
        shutil.copytree(path,initial/path.name,dirs_exist_ok=True)
    elif path.is_file() and (path.name.startswith(('p32-','p128-','p512-','pilot-','compile-harness')) or
                            path.name in ['telemetry.csv','compute-processes.jsonl','environment.json','admission.json','status.json','after.json','runner.log','benchmark_llama_ds41']):
        shutil.copy2(path,initial/path.name)
shutil.copy2(ROOT/'benchmark_llama_ds41-initial.cpp',initial/'benchmark_llama_ds41.cpp')

try:
    source=ROOT/'source'
    command('compile-final',[PREFIX/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
        '-I'+str(source/'include'),'-I'+str(source/'ggml/include'),'-I'+str(source/'vendor'),
        ROOT/'benchmark_llama_ds41.cpp','-L'+str(ROOT/'build/bin'),'-lllama','-lggml','-lggml-base','-lggml-cpu',
        '-L'+str(PREFIX/'targets/x86_64-linux/lib/stubs'),'-lcuda','-lpthread','-o',ROOT/'benchmark_llama_ds41'])
    environment=json.loads((ROOT/'environment.json').read_text())
    environment.update(binary_sha256=sha(ROOT/'benchmark_llama_ds41'),source_sha256=sha(ROOT/'benchmark_llama_ds41.cpp'),
                       initial_debug_binary_sha256=sha(initial/'benchmark_llama_ds41'),log_level='info')
    save('environment.json',environment)
    chosen=json.loads((ROOT/'selection.json').read_text())['threads']
    with admitted():
        stop=threading.Event()
        with (ROOT/'telemetry.csv').open('w') as log:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem',
                '--format=csv','--loop-ms=1000'],stdout=log,stderr=subprocess.STDOUT)
            def observe():
                with (ROOT/'compute-processes.jsonl').open('w') as output:
                    while not stop.is_set():
                        apps=capture(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader'])
                        output.write(json.dumps(dict(apps=apps))+'\n');output.flush();stop.wait(3)
            thread=threading.Thread(target=observe,daemon=True);thread.start()
            try:
                def run_case(label,n,repeats,ubatch):
                    command(label,['/usr/bin/time','-v','-o',ROOT/(label+'-resources.txt'),ROOT/'benchmark_llama_ds41',
                        MODEL,ROOT/('workload-p'+str(n)+'.json'),ROOT/label,repeats,chosen,31,ubatch])
                run_case('serial-p128-d64',128,1,1)
                for n in [32,128,512]:run_case(f'p{n}-d64',n,3,512)
            finally:
                stop.set();thread.join(timeout=10);monitor.terminate();monitor.wait(timeout=10)
    # Logging must not change any primary-case output.
    for n in [32,128,512]:
        label=f'p{n}-d64'
        for r in range(4):
            name=f'final-logits-round{r}.f32'
            assert (ROOT/label/name).read_bytes()==(initial/label/name).read_bytes(),label
    save('after.json',dict(compute_apps=capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']),
                          log_filter_final_logits_bitwise_equal_to_initial=True))
    status('complete',validation='info logging, serial-prefill control, unchanged logits')
except BaseException as error:
    status('failed',error=repr(error));raise
