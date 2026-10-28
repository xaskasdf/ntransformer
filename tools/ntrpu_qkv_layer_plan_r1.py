"""Shared-cache layer/token traversal, retries and full-width identity rebinding."""

def plan():
    jobs=[]
    def add(layer,position,mode=0,rebind=False,group=1):
        jobs.append(dict(layer=layer,position=position,mode=mode,rebind=rebind,group=group,
            want=3 if mode==34 else 2 if mode==11 else 1 if mode in (90,91,92) else 0,
            capture_status=2 if mode==11 else 1 if mode in (91,92) else 0))
    for layer in range(16):add(layer,0,rebind=layer==0)
    add(0,0,90)                       # Duplicate token must not append again.
    add(8,2,90)                       # A gap cannot borrow another layer's prefix.
    add(7,1,34)                       # Last B error leaves an unpublished slot.
    add(12,1)                        # Intervening work on another layer.
    add(7,1)                         # Retry the same position after the failed append.
    for i in range(16):
        layer=(i*5+7)%16
        if layer not in (7,12):add(layer,1)
    for layer in reversed(range(16)):add(layer,2,40 if layer==5 else 0)
    add(15,3,90)                      # Full allocation cannot accept a fourth token.
    add(16,0,91);add(0x10000000,0,92) # Never alias out-of-range layers.
    add(15,0,rebind=True,group=2)
    add(0,0,group=2)
    add(1,0,11,group=2)              # Stale source identity after rebinding.
    add(1,0,group=2)
    assert len(jobs)==58 and sum(r['want']==0 for r in jobs)==51
    return jobs

def identity(group):
    high=0 if group==1 else 1<<40
    return 0x1234567800000001^high,0xfedcba9800000001^high
