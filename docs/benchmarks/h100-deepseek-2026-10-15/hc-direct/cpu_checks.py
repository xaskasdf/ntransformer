from common import *
import hashlib, re
tokens=','.join(str(1 if i==0 else 3+(i*37)%317) for i in range(18))
for label,binary in [('cpu-before',old/'build/ntransformer'),('cpu-after',root/'build/ntransformer')]:
    run(label,[binary,'-m','/home/ubuntu/tiny-deepseek41.gguf','--token-ids',tokens,'-n','4',
        '--temperature','0','--dump-logits',root/(label+'.bin')],{'DS41_GPU':'0','DS41_DUMP_STAGES':'1'})
logs=[(root/(label+'.log')).read_text(errors='replace') for label in ['cpu-before','cpu-after']]
stages=[re.findall(r'^\[stage\].*$',s,re.M) for s in logs]
assert stages[0] and stages[0]==stages[1]
assert (root/'cpu-before.bin').read_bytes()==(root/'cpu-after.bin').read_bytes()
(root/'cpu-exact.json').write_text(json.dumps(dict(bitwise_logits=True,stage_sums_equal=True,
    stages=len(stages[0]),sha256=hashlib.sha256((root/'cpu-after.bin').read_bytes()).hexdigest()),indent=2))
print('PASS CPU cross-binary logits and stage sums',flush=True)
