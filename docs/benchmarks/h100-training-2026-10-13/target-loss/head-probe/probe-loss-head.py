import os,sys,json,fcntl,time,subprocess
from pathlib import Path
from contextlib import ExitStack
root=Path('/home/ubuntu/ntransformer-h100')
out=root/'training/loss-head-probe-20261014'
out.mkdir(exist_ok=False)
source=root/'training/target-loss-20261014/source-snapshot'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['NTRANSFORMER_TRAINING_LIB']=str(root/'training/target-loss-20261014/libntransformer_training.so')
sys.path[:0]=[str(source/'python'),str(source/'scripts')]
import torch
import torch.nn.functional as F
from train_brandon_experiment import import_brandon,install
from ntransformer_training.loss import linear_cross_entropy

def diff(a,b):
 a,b=a.double(),b.double()
 return dict(exact=torch.equal(a,b),max_abs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item())

deadline=time.monotonic()+7200
while True:
 stack=ExitStack()
 try:
  for name in ('stage1-v1','behavior-v1'):
   path=Path('/lambda/nfs/btiny/brandon-mini/stages')/name
   assert json.loads((path/'status.json').read_text())['state']=='complete'
   lock=stack.enter_context((path/'pipeline.lock').open('a'))
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  break
 except BlockingIOError:
  stack.close()
  if time.monotonic()>deadline: raise
  time.sleep(10)
with stack:
 torch.set_num_threads(2)
 torch.set_float32_matmul_precision('high')
 torch.use_deterministic_algorithms(True)
 torch.utils.deterministic.fill_uninitialized_memory=False
 stage=Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
 TinyLlama,ModelConfig,PackedCorpus,_,_=import_brandon(stage.parent.parent/'sources/stage1-v1')
 cp=torch.load(stage/'pretrain/last.pt',map_location='cpu',weights_only=False)
 model=TinyLlama(ModelConfig(**cp['config'])).cuda();model.load_state_dict(cp['model_state_dict'])
 install(model,shared_projection_casts=True,direct_gqa=True,fused_rope=True)
 settings=cp['settings'];corpus=PackedCorpus(stage/'data','train',model.config.max_seq_len,settings['seed'])
 x,y=corpus.batch(settings['batch_size'],cp['step']+1,0,0);x,y=x.cuda(),y.cuda()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  h=model.dropout(model.tok_embeddings(x))
  h,_=model._run_layers(h,model.freqs_cis[:x.shape[1]],v_first_in=None)
  h=model.norm(h).detach()
 w=model.output.weight.detach().clone()
 del model,cp
 result={}
 expected=None
 for mode,block in [('reference',8192),('fused',8192),('vocab-stream',8192),('vocab-stream',4096),('vocab-stream',2048),('vocab-stream',1024)]:
  hi=h.detach().requires_grad_();wi=w.detach().requires_grad_()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   if mode=='reference':
    z=F.linear(hi,wi);loss=F.cross_entropy(z.view(-1,8192),y.view(-1))
   else:loss=linear_cross_entropy(hi,wi,y,mode=mode,block_size=block)
  grads=torch.autograd.grad(loss,(hi,wi))
  if expected is None:expected=[g.clone() for g in grads]
  record=dict(loss=loss.item(),dh=diff(grads[0],expected[0]),dw=diff(grads[1],expected[1]))
  result[f'{mode}-{block}']=record;print(json.dumps({f'{mode}-{block}':record}),flush=True)
  del loss,grads
 # Hold dz fixed to separate loss changes from gradient GEMM accumulation.
 hb=h.bfloat16().view(-1,h.shape[-1]);wb=w.bfloat16()
 z=torch.mm(hb,wb.t()).requires_grad_()
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(z,y.view(-1))
 dz,=torch.autograd.grad(loss,z)
 full=torch.mm(dz,wb)
 fp32=torch.mm(dz,wb,out_dtype=torch.float32).bfloat16()
 result['fixed_dz_fp32_full']=diff(fp32,full)
 for block in (4096,2048,1024):
  partial=None
  for start in range(0,8192,block):
   term=torch.mm(dz[:,start:start+block].contiguous(),wb[start:start+block],out_dtype=torch.float32)
   if partial is None:partial=term
   else:partial.add_(term)
  result[f'fixed_dz_fp32_blocks-{block}']=diff(partial.bfloat16(),full)
 old=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 result['fixed_dz_no_reduced_reduction']=diff(torch.mm(dz,wb),full)
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
