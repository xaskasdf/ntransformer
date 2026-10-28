import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def quality(a,b,targets):
    kl=[];nll_a=[];nll_b=[];agreement=[]
    for x,y,target in zip(a,b,targets):
        x=x.astype(np.float64);y=y.astype(np.float64)
        x-=x.max();x-=np.log(np.exp(x).sum())
        y-=y.max();y-=np.log(np.exp(y).sum())
        kl.append(max(0,float((np.exp(x)*(x-y)).sum())))
        nll_a.append(float(-x[target]));nll_b.append(float(-y[target]))
        agreement.append(int(x.argmax())==int(y.argmax()))
    return dict(nt_nll=statistics.mean(nll_a),llama_nll=statistics.mean(nll_b),
        nll_delta_llama_minus_nt=statistics.mean(nll_b)-statistics.mean(nll_a),
        mean_kl_nt_to_llama=statistics.mean(kl),p99_kl_nt_to_llama=float(np.quantile(kl,.99)),
        argmax_agreement=statistics.mean(agreement),positions=len(targets))

def load_case(label, workload, engine, rounds, selection):
    data=[json.loads(line) for line in (ROOT/label/'measurements.jsonl').read_text().splitlines()]
    setup=next(r for r in data if r['kind']=='setup')
    command=read(ROOT/(label+'-command.json'))
    assert command['returncode']==0 and command['environment']['CUDA_VISIBLE_DEVICES']=='1,0'
    assert setup['teacher_forcing'] and setup['prompt_tokens']==len(workload['prompt_ids'])
    assert setup['decode_steps']==64 and len(workload['target_ids'])==65
    assert read(ROOT/label/'prompt-ids.json')==workload['prompt_ids']
    if engine=='llama':
        assert read(ROOT/label/'continuation-ids.json')==workload['continuation_ids']
        assert setup['gpu_count']==2 and setup['devices']==['CUDA0','CUDA1'] and setup['split_mode']=='layer'
        assert setup['threads']==selection['threads'] and setup['cpu_moe_layers']==selection['cpu_moe']
        assert command['environment']['GGML_OP_OFFLOAD_MIN_BATCH']==('32' if selection['op_offload'] else '1000000')
    else:
        assert read(ROOT/label/'target-ids.json')==workload['target_ids']
        assert sha(ROOT/label/'all-logits.f32')==workload['nt_reference_logits_sha256']
        for key,value in [('DS41_PEER_MODE','execute'),('DS41_VRAM_GB','62'),('DS41_PEER_VRAM_GB','62'),
            ('DS41_MXFP4_COMPUTE','fp32'),('DS41_Q8_COOPERATIVE','1'),('DS41_CACHE_POLICY',engine.removeprefix('nt-'))]:
            assert command['environment'][key]==value
    width=setup['vocab']
    logits=np.fromfile(ROOT/label/'all-logits.f32',dtype='<f4')
    assert logits.size==65*width and np.isfinite(logits).all()
    phases={(r['round'],r['phase']):r for r in data if r['kind']=='phase'}
    assert len(phases)==10 and set(phases)=={(i,p) for i in range(5) for p in ['prefill','decode']}
    completion=[r for r in data if r['kind']=='completion']
    assert len(completion)==5 and {r['round'] for r in completion}==set(range(5))
    assert all(r['finite_logit_values']==logits.size for r in completion)
    for (round,phase),r in phases.items():
        assert r['tokens']==(64 if phase=='decode' else len(workload['prompt_ids']))
        assert np.isfinite(r['seconds']) and r['seconds']>0
        assert all(r[k]>=0 for k in ['major_page_faults','minor_page_faults','input_blocks','output_blocks'])
        if engine!='llama':
            assert r['peer_fallback_calls']==r['routed_cpu_calls']==0
            assert r['peer_execute_calls']==r['routed_gpu_calls']>0
    start=datetime.datetime.fromisoformat(command['utc_start']).replace(tzinfo=None)
    end=start+datetime.timedelta(seconds=command['wall_seconds'])
    peaks={}
    with (ROOT/'telemetry.csv').open() as stream:
        for r in csv.DictReader(stream,skipinitialspace=True):
            stamp=datetime.datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f')
            if start<=stamp<=end:
                peaks[r['index']]=max(peaks.get(r['index'],0),float(r['memory.used [MiB]'].split()[0]))
    assert set(peaks)=={'0','1'} and min(peaks.values())>60000, (label,'missing dual-GPU allocation samples',peaks)
    resources=(ROOT/(label+'-resources.txt')).read_text()
    return dict(label=label,decode_seconds=[phases[i,'decode']['seconds'] for i in rounds],
        prefill_seconds=[phases[i,'prefill']['seconds'] for i in rounds],
        request_seconds=[phases[i,'decode']['seconds']+phases[i,'prefill']['seconds'] for i in rounds],
        warm_major_page_faults=sum(phases[i,p]['major_page_faults'] for i in rounds for p in ['prefill','decode']),
        warm_input_blocks=sum(phases[i,p]['input_blocks'] for i in rounds for p in ['prefill','decode']),
        cold_request_seconds=phases[0,'prefill']['seconds']+phases[0,'decode']['seconds'],
        cold_major_page_faults=phases[0,'prefill']['major_page_faults']+phases[0,'decode']['major_page_faults'],
        repeat_bitwise_equal=all(r['repeat_bitwise_equal'] for r in completion),
        logits_sha256=sha(ROOT/label/'all-logits.f32'),sampled_peak_vram_mib=peaks,
        peak_rss_mib=float(re.search(r'Maximum resident set size \(kbytes\): (\d+)',resources)[1])/1024,
        setup=setup),logits.reshape(65,width)

def main():
    plan=read(ROOT/'plan.json');selection=read(ROOT/'selection.json')['selected']['config']
    observations=[json.loads(line) for line in (ROOT/'compute-processes.jsonl').read_text().splitlines()]
    assert observations and not any(r.get('foreign') or r.get('error') for r in observations)
    result=dict(protocol=plan,selected_llama=selection,workloads={},quality={},comparisons={})
    for w in plan['workloads']:
        workload=read(ROOT/('workload-'+w+'.json'))
        assert workload['continuation_ids']==workload['target_ids'][:-1]
        assert len(workload['prompt_ids'])==int(w.split('p')[-1])
        engines=plan['main_engines']+(['nt-lru'] if w=='p512' else [])
        cases={};arrays={}
        for engine in engines:
            launches=[]
            for launch in range(plan['independent_launches']):
                case,logits=load_case(f'{engine}-{w}-r{launch}',workload,engine,plan['measured_rounds'],selection)
                launches.append(case);arrays[engine,launch]=logits
            d=[v for c in launches for v in c['decode_seconds']]
            p=[v for c in launches for v in c['prefill_seconds']]
            r=[v for c in launches for v in c['request_seconds']]
            cases[engine]=dict(decode_tokens_per_second=64/statistics.median(d),
                prefill_tokens_per_second=len(workload['prompt_ids'])/statistics.median(p),
                median_request_seconds=statistics.median(r),decode_range_seconds=[min(d),max(d)],
                warm_major_page_faults=sum(c['warm_major_page_faults'] for c in launches),
                warm_input_blocks=sum(c['warm_input_blocks'] for c in launches),
                cross_launch_logits_equal=launches[0]['logits_sha256']==launches[1]['logits_sha256'],launches=launches)
        result['workloads'][w]=cases
        result['quality'][w]=[quality(arrays['nt-slru',i],arrays['llama',i],workload['target_ids']) for i in range(2)]
        nt,ll=cases['nt-slru'],cases['llama']
        result['comparisons'][w]=dict(llama_over_nt_decode=ll['decode_tokens_per_second']/nt['decode_tokens_per_second'],
            llama_over_nt_prefill=ll['prefill_tokens_per_second']/nt['prefill_tokens_per_second'],
            llama_over_nt_request=nt['median_request_seconds']/ll['median_request_seconds'],
            paired_decode_ratios=[statistics.median(nt['launches'][i]['decode_seconds'])/statistics.median(ll['launches'][i]['decode_seconds']) for i in range(2)],
            warm_without_observed_major_faults=nt['warm_major_page_faults']==ll['warm_major_page_faults']==0)
    lines=['# Matched two-H100 DeepSeek comparison','',
        'Same ten GGUF shards, prompt IDs and 64 consumed continuation IDs, same two H100 80 GB SXM5 GPUs and host. Two independent launches per case, counterbalanced engine order; two warmup requests and three measured requests per launch. Each request resets sequence/KV state and retains the engine\'s weight cache. Rates below use the median of six warm phase times.', '',
        'NTransformer uses owner-GPU expert execution, cooperative Q8 backbone, FP32 routed arithmetic and SLRU80 with 62 GiB expert cache per GPU. LRU is a separate context-512 control. llama.cpp uses its pinned native layer split and balanced routed-weight placement, chosen in a bounded eight-candidate pilot before main measurements. Available hardware is equal; allocated VRAM and arithmetic are measured/reported rather than forced equal.', '',
        'Selected llama configuration: `'+json.dumps(selection,sort_keys=True)+'`; ubatch 512, 1:1 layer split, F32 KV requested. Pilot score is warm prefill plus projected 64-step decode at context 128. The separate narrative is excluded from this selection. This is not an exhaustive search.', '',
        '| Workload | NT decode tok/s | llama decode tok/s | llama / NT | NT prefill tok/s | llama prefill tok/s | Warm major faults NT / llama |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for w,cases in result['workloads'].items():
        nt,ll=cases['nt-slru'],cases['llama'];m=result['comparisons'][w]
        lines.append(f"| {w} | {nt['decode_tokens_per_second']:.4f} | {ll['decode_tokens_per_second']:.4f} | {m['llama_over_nt_decode']:.3f}x | {nt['prefill_tokens_per_second']:.4f} | {ll['prefill_tokens_per_second']:.4f} | {nt['warm_major_page_faults']} / {ll['warm_major_page_faults']} |")
    lines+=['','The summary retains the paired launch ratios, all six times, whole warm-request latency and default-LRU control. Two launches do not establish a population confidence interval. Any nonzero major-fault count flags residual paging in the measured warm envelope; input-block counters are also retained.','',
        '## Numerical evidence','',
        'Each engine consumes the same fixed tokens regardless of its own predictions. All 65 vocabulary vectors are checked for finiteness in every request. Cross-engine bitwise equality is not required. The following values are the mean across the two independent launches; KL is NT to llama. This small likelihood probe does not certify equivalent downstream quality or instruction following.','',
        '| Workload | NT target NLL | llama target NLL | Mean KL | Argmax agreement |','|---|---:|---:|---:|---:|']
    for w,values in result['quality'].items():
        means={k:statistics.mean(r[k] for r in values) for k in ['nt_nll','llama_nll','mean_kl_nt_to_llama','argmax_agreement']}
        lines.append(f"| {w} | {means['nt_nll']:.6f} | {means['llama_nll']:.6f} | {means['mean_kl_nt_to_llama']:.6f} | {means['argmax_agreement']*100:.2f}% |")
    lines+=['','## Memory and first-request diagnostics','','| Case | Peak GPU 0 / 1 MiB | Peak RSS MiB | First request s | First-request major faults | Repeat bitwise |','|---|---:|---:|---:|---:|---|']
    for cases in result['workloads'].values():
        for case in cases.values():
            for c in case['launches']:
                peaks=c['sampled_peak_vram_mib']
                lines.append(f"| {c['label']} | {peaks['0']:.0f} / {peaks['1']:.0f} | {c['peak_rss_mib']:.0f} | {c['cold_request_seconds']:.3f} | {c['cold_major_page_faults']} | {c['repeat_bitwise_equal']} |")
    lines+=['','The first request is not a controlled cold-storage comparison: the global OS file cache is never flushed. Page faults/I/O counters cover each phase across process threads; decode timing excludes validation copies while its resource envelope includes them. VRAM peaks are one-second samples; RSS is the process high-water mark. Setup, initialization, file output and validation are excluded from phase throughput and recorded separately.','',
        '## Provenance and reproduction','',
        'The engine library/source identities, harnesses, predeclared plan, input-token JSONs, selected pilot and rejected/OOM pilots, commands, logits, telemetry, per-phase counters and analysis are preserved. The NT engine library is exactly the version that passed the preceding 52 CTests; only its benchmark harness gains resource counters. The llama harness gains balanced two-GPU placement, full-logit output and the same counters. GPU admission holds shared training locks and requires both GPUs idle; a foreign GPU process appearing later aborts only this benchmark. Model identity records shard sizes/mtime rather than newly hashing the entire 473 GiB payload.','']
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(result['comparisons']),flush=True)

if __name__=='__main__':main()
