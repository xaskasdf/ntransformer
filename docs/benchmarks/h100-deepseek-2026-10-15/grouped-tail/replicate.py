from common import *
import hashlib, shutil
try:
    shutil.copy2(root/'analysis.json',root/'initial-analysis.json')
    with admitted('replication',wait_seconds=300):
        for label,mode in [('grouped3','1'),('serial3','0'),('serial4','0'),('grouped4','1')]:
            run(label,[root/'profile-candidate',model,'16','3',root/(label+'.bin')],{'DS41_GROUPED_TAIL':mode})
        exact=json.loads((root/'exact.json').read_text())
        exact['runs']+=['grouped3','serial3','serial4','grouped4']
        data=[(root/(name+'.bin')).read_bytes() for name in exact['runs']]
        assert all(d==data[0] for d in data),'replication logits differ'
        assert hashlib.sha256(data[0]).hexdigest()==exact['sha256']
        (root/'exact.json').write_text(json.dumps(exact,indent=2))
    status('complete')
except Exception as e:status('failed',error=repr(e));raise
