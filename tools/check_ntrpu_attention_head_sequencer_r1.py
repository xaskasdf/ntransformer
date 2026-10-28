"""Review controller/peer transcripts; this does not qualify attention arithmetic."""
import re

SUCCESS = {0,1,19,20,22,24,26,28,30}
RESET = {21,23,25,27}
CHILDREN = {11:8,12:6,13:4,15:5,16:3,17:7,18:9,23:4,25:6,27:32}
STATUS = {2:4,3:2,4:4,5:4,6:4,7:8,8:1,9:2,10:1,11:3,12:4,13:8,
          14:7,15:7,16:2,17:7,18:2,29:8,31:2,32:4,33:1,34:1}


def review(text, seed, watchdog):
    assert not re.search(r'(?mi)^\s*(?:HEADSEQ_FAIL|ERROR|FATAL|FAIL|TIMEOUT)', text)
    lines = text.splitlines()
    assert lines.count(f'HEADSEQ_CONFIG seed={seed} watchdog={watchdog} arithmetic_peer=0') == 1
    heads = {}; releases = {}; words = {}; done = {}; current = None; begins = []; reset_jobs = []
    for line in lines:
        if line.startswith('HEADSEQ_BEGIN '):
            m = re.fullmatch(r'HEADSEQ_BEGIN job=(\d+) mode=(\d+)', line); assert m
            j, mode = map(int,m.groups()); assert j == mode+1 and j == len(begins)+1
            if current is not None: assert current in done or current in reset_jobs
            current=j;begins.append(j);heads[j]=[];releases[j]=[];words[j]=[]
        elif line.startswith('HEADSEQ_CHILD '):
            m=re.fullmatch(r'HEADSEQ_CHILD job=(\d+) head=(\d+) lease=([0-9a-f]{16})',line);assert m
            j,head=int(m[1]),int(m[2]);assert j==current and j not in done
            assert head==len(heads[j]) and head<32
            assert len(releases[j])==head
            assert int(m[3],16)==0x80000000+j*64+head
            heads[j].append(head)
        elif line.startswith('HEADSEQ_RELEASE '):
            m=re.fullmatch(r'HEADSEQ_RELEASE job=(\d+) head=(\d+) status=(\d+)',line);assert m
            j,head,status=map(int,m.groups());assert j==current and j not in done
            assert head==len(releases[j]) and head<len(heads[j])
            expected=3 if j-1==11 and head==7 else 7 if j-1 in (15,16) and head==CHILDREN[j-1]-1 else 0
            assert status==expected
            releases[j].append(head)
        elif line.startswith('HEADSEQ_WORD '):
            m=re.fullmatch(r'HEADSEQ_WORD job=(\d+) index=(\d+) data=([0-9a-f]{8})',line);assert m
            j,index=int(m[1]),int(m[2]);assert j==current and j not in done and j-1 in SUCCESS
            assert len(heads[j])==len(releases[j])==32 and index==len(words[j]) and index<2048
            assert int(m[3],16)==0x3f000000 | (j<<12) | index
            words[j].append(int(m[3],16))
        elif line.startswith('HEADSEQ_DONE '):
            m=re.fullmatch(r'HEADSEQ_DONE job=(\d+) mode=(\d+) status=(\d+) children=(\d+)',line);assert m
            j,mode,status,count=map(int,m.groups());assert j==current and mode==j-1 and j not in done and mode not in RESET
            expected=32 if mode in SUCCESS else CHILDREN.get(mode,0)
            assert count==len(heads[j])==len(releases[j])==expected
            assert status==STATUS.get(mode,0) and len(words[j])==(2048 if mode in SUCCESS else 0)
            done[j]=status
        elif line.startswith('HEADSEQ_RESET '):
            m=re.fullmatch(r'HEADSEQ_RESET count=(\d+) job=(\d+)',line);assert m
            number,j=map(int,m.groups());assert number==len(reset_jobs)+1
            if number==1:assert j==0 and current is None
            else:
                assert j==current and j-1 in RESET and j not in done and not words[j]
                assert len(heads[j])==CHILDREN.get(j-1,0)
                assert len(releases[j])==len(heads[j])-(1 if j-1 in (23,25) else 0)
            reset_jobs.append(j)
        elif line.startswith('HEADSEQ_') and not line.startswith(('HEADSEQ_CONFIG ','HEADSEQ_PASS ')):
            raise AssertionError('Unknown trace record')
    summaries=re.findall(r'^HEADSEQ_PASS jobs=(\d+) completions=(\d+) resets=(\d+) output_words=(\d+) children=(\d+) cycles=(\d+)$',text,re.M)
    assert len(summaries)==1
    jobs,completions,resets,count,children,cycles=map(int,summaries[0])
    assert jobs==len(begins)==35 and completions==len(done)==31 and resets==len(reset_jobs)==5
    assert count==sum(map(len,words.values()))==18432 and children==sum(map(len,heads.values()))==372
    assert 0<cycles<3000000
    return dict(status='PASS_HEAD_SEQUENCER_R1_CONTROLLER_TRANSCRIPT',seed=seed,watchdog=watchdog,
                jobs=jobs,completions=completions,resets=resets,output_words=count,children=children,cycles=cycles,
                scope='Standalone controller RTL with deterministic protocol peer. No actual attention, KV/AXI wrapper, or physical timing.')


def controls(text, seed, watchdog):
    changes={
        'head_order':lambda s:s.replace('head=1 lease=', 'head=0 lease=',1),
        'lease':lambda s:s.replace('lease=0000000080000040','lease=0000000080000041',1),
        'release':lambda s:re.sub(r'^HEADSEQ_RELEASE[^\n]*\n','',s,count=1,flags=re.M),
        'word':lambda s:s.replace('data=3f001000','data=3f001001',1),
        'word_order':lambda s:s.replace('index=1 data=', 'index=0 data=',1),
        'missing_word':lambda s:re.sub(r'^HEADSEQ_WORD[^\n]*\n','',s,count=1,flags=re.M),
        'status':lambda s:s.replace('mode=11 status=3','mode=11 status=0',1),
        'reset':lambda s:s.replace('HEADSEQ_RESET count=2','HEADSEQ_RESET count=3',1),
        'summary':lambda s:s.replace('HEADSEQ_PASS jobs=35','HEADSEQ_PASS jobs=34',1),
        'diagnostic':lambda s:s+'\nHEADSEQ_FAIL injected\n',
    }
    rejected=[]
    for name,change in changes.items():
        altered=change(text);assert altered!=text,name
        try:review(altered,seed,watchdog)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
