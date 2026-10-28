"""Protocol-only lease cases, each followed by a clean ownership recovery."""
def status(mode):
    if mode>=100:
        if 300<=mode<340 and (2 if (mode-300)//10==0 else (mode-300)//10+5)==(mode-300)%10:return 0
        return 1
    return {1:3,2:5,3:6,4:5,5:5,6:5,7:2,8:2,9:2,10:1,11:1,12:1,13:1,14:1,
        15:1,16:1,17:1,18:1,19:1,21:11,22:11,23:11,24:2,25:7,26:7,27:8,28:8,
        29:11,41:11,42:11,43:11,44:11,45:11,46:5,47:5,48:5,49:5,50:2,51:2,52:1,53:1}.get(mode,0)
def child(mode):
    return mode in {0,1,2,20,22,23,24,25,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46} or (mode>=300 and status(mode)==0)
def plan():
    modes=list(range(54))+list(range(100,170))+list(range(200,210))+list(range(300,340))
    return [dict(mode=m,status=status(m),child=child(m),reset=m in (39,40),late=m in (30,31,32,33,34)) for mode in modes for m in (mode,0)]
