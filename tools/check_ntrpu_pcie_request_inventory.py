"""Review every frozen RTL vector and printed DUT state independently of PASS."""
import re


def require(condition,message):
    if not condition:raise ValueError(message)


def review(log,vectors,manifest):
    require(not re.search(r'^\s*(Fatal:|Failure:|ERROR:|CRITICAL WARNING:)',log,re.M),'simulator failure')
    expected=[line.split() for line in vectors.splitlines() if line.strip()]
    rows=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in log.splitlines() if line.startswith('PCIE_INV_ROW ')]
    require(len(rows)==len(expected)==manifest['rows'],'all vector outputs')
    cases={};peaks=dict(tags=0,reads=0,bytes=0)
    for index,(row,e) in enumerate(zip(rows,expected)):
        require(len(e)==28 and int(row['index'])==index and int(row['scenario'])==int(e[0]) and int(row['native_ps'])==int(e[1]),'vector identity/order')
        fault=int(row['fault'],16)
        require(int(row['observing'])==int(e[16]) and bool(fault)==bool(int(e[17])),'observation/fault result')
        case=cases.setdefault(int(e[0]),dict(rows=0,fault_sources=0,native_rows=0))
        case['rows']+=1;case['fault_sources']|=fault;case['native_rows']+=int(int(e[1])!=0)
        if int(e[18]):
            for n,column in [('tags',19),('reads',20),('bytes',21),('rq_partial',22),('rc_partial',23),('requests',24),('completed',25),('writes',26)]:
                require(int(row[n])==int(e[column]),'RTL/reference '+n+' at vector '+str(index))
            require(int(row['bitmap'],16)==int(e[27],16),'all 256 tag identities')
        empty=(int(e[16]) and not int(e[17]) and not any(int(e[c]) for c in [19,20,21,22,23,8,12]) and int(e[3]) and not int(e[4]) and int(e[5]))
        require(int(row['empty'])==int(bool(empty)) if int(e[18]) else int(row['empty'])==0,'empty cannot omit an obligation/error')
        if not fault:
            for n in peaks:peaks[n]=max(peaks[n],int(row[n]))
    require(sorted(cases)==list(range(len(manifest['scenarios']))),'all scenarios')
    expected_fault={'rc_count':8,'rc_address':8,'rc_requester':7,'rc_tc':7,'rc_attribute':7,'rc_terminal':9,
        'rc_enable':10,'rc_sop_mask':2,'rc_orphan_tag':6,'rc_unknown_payload':4,'duplicate_tag':5,
        'request_4KiB_crossing':3,'rq_ready_not_replicated':1,'stalled_valid_withdrawal':1,
        'link_preserves_pending':0,'user_reset_preserves_pending':0,'profile_preserves_pending':0,'bdf_preserves_pending':0,
        'completion_error_2':11,'completion_error_9':11,'late_rc_discontinue':12}
    for scenario in manifest['scenarios']:
        actual=cases[scenario['id']]
        require(actual['rows']==scenario['last_vector']-scenario['first_vector']+1,'scenario row extent')
        require(bool(actual['fault_sources'])==scenario['reference_fault'],'scenario fault classification')
        if scenario['name'].startswith('fault_'):
            bit=expected_fault[scenario['name'][6:]]
            require(actual['fault_sources'] & (1<<bit),'specific sticky fault source '+scenario['name'])
    markers=[line for line in log.splitlines() if line.startswith('PCIE_INV_SUITE_PASS ')]
    require(markers==['PCIE_INV_SUITE_PASS rows='+str(len(rows))+' scenarios='+str(len(cases))+' native_endpoint=0 native_trace_replayed=1 global_drain=0'],'exact suite scope')
    require('$finish called at time' in log,'simulator finish')
    return dict(status='PASS_PCIE_REQUEST_INVENTORY_RTL_REFERENCE_REPLAY',rows=len(rows),scenarios=len(cases),
        negative_scenarios=sum(bool(c['fault_sources']) for c in cases.values()),peaks=peaks,cases=cases,
        native_replay_rows=sum(c['native_rows'] for c in cases.values()),qualified_native_endpoint=False,
        qualified_synthesis=False,qualified_physical_design=False,qualified_global_drain=False)
