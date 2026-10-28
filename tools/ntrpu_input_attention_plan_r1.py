"""Full front-owner protocol plan; peers model replies, not HLS arithmetic."""
from check_ntrpu_qkv_rope_handoff_r1 import regions as lease_regions


def regions(mode):
    return lease_regions(20 if mode==17 else mode)


def status(mode):
    if mode>=100:
        if 300<=mode<340 and [2,6,7,8][(mode-300)//10]==(mode-300)%10:return 0
        return 1
    return {1:3,2:5,3:3,4:5,5:7,6:7,7:7,8:8,9:8,13:11,14:11,
            18:2,19:2,20:1,21:1,22:1,23:1,24:2,26:11,27:11,28:2}.get(mode,0)


def plan():
    modes=list(range(29))+list(range(100,170))+list(range(200,210))+list(range(300,340))
    result=[]
    for mode in modes:
        for m in (mode,0):
            producer=(m<100 and m not in (5,18,19,20,21,22,23,24,28)) or (m>=300 and status(m)==0)
            consumer=producer and m not in (1,2,6,8,13,26)
            result.append(dict(mode=m,status=status(m),producer=producer,consumer=consumer,late=m in (10,11,12,25)))
    return result
