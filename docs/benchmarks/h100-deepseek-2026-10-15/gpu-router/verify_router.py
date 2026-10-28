import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    summary=json.loads((ROOT/'summary.json').read_text())
    assert len(summary['cases'])==6
    for name,c in summary['cases'].items():
        assert c['comparison']['all_logits_bitwise_equal'],name
        assert c['comparison']['predicted_ids_equal'],name
        assert c['all_repeats_bitwise_equal'] and c['logit_values']==129280*65,name
        assert c['prefill']['warm_requests']==c['decode']['warm_requests']==3,name
        assert json.loads((ROOT/(name+'-command.json')).read_text())['returncode']==0,name
        env=c['environment']
        for phase in ['prefill','decode']:
            p=c[phase]['per_token']
            assert p['router_gpu_calls']==(0 if env.get('DS41_ROUTER_GPU','1')=='0' else 40),name
            assert p['prologue_readback_waits']==(120 if env.get('DS41_QKV_READBACK_BATCH','1')=='0' else 40),name
            assert p['routed_cpu_calls']==p['shared_cpu_calls']==0,name
            assert p['routed_gpu_calls']==p['shared_gpu_calls']==40,name
            baseline=summary['cases']['control-p'+str(c['prompt_tokens'])][phase]['per_token']
            assert p['routed_upload_bytes']==baseline['routed_upload_bytes'],name
    historical={}
    for n in [32,128]:
        a=ROOT/f'control-p{n}/final-logits.f32'
        b=ROOT.parent/f'ds41-bench-20261015/p{n}-d64/final-logits.f32'
        historical[str(n)]=dict(control_sha256=sha(a),original_baseline_sha256=sha(b),equal=sha(a)==sha(b))
        assert historical[str(n)]['equal'],n
    meta=json.loads((ROOT/'router-weight-metadata.json').read_text())
    assert meta['total_bytes']==157286400 and len(meta['tensors'])==40
    assert all(t['dtype']=='BF16' for t in meta['tensors'])
    proof=dict(cases_verified=6,full_logit_values_per_case=129280*65,
        final_defaults_bitwise_equal_at_prompts=[32,128],historical_baseline=historical,
        router_weights_bytes=meta['total_bytes'],gpu_coverage_complete=True,expert_upload_bytes_unchanged=True)
    (ROOT/'validation.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
