"""Compare HLS scheduling variants with the qualified 27-transaction baseline."""
import collections
import re


def review(candidate, baseline):
    errors=[]
    def need(ok, message):
        if not ok:errors.append(message)
    marker='PASS SwiGLU Q8 block cases=27 successful=14'
    traces=[]
    latencies=[]
    for label,data in [('baseline',baseline),('candidate',candidate)]:
        log=data['log']
        lines=[line for line in log.splitlines() if line.startswith('NTRPU_SWIGLU_BLOCK_CASE ')]
        need(log.count(marker)>=2,label+' C/RTL pass markers')
        need('*** C/RTL co-simulation finished: PASS ***' in log,label+' cosim completion')
        need(not re.search(r'^(ERROR:|CRITICAL WARNING:|FAIL|FATAL_ERROR)',log,re.M),label+' no errors')
        need(len(lines)>=54 and len(lines)%27==0,label+' complete C/RTL traces')
        first=lines[:27]
        status=[0]*10+[1]*2+[2]*6+[3]*2+[4]+[2]*2+[0]*4
        for offset in range(0,len(lines),27):
            need(lines[offset:offset+27]==first,label+' repeated C/RTL trace equality')
        need([int(re.search(r'id=(\d+)',line)[1]) for line in first]==list(range(1,28)),label+' ordered case ids')
        need([int(re.search(r'status=(\d+)',line)[1]) for line in first]==status,label+' expected status sequence')
        traces.append(first)
        need(bool(re.search(r'\|\s*Verilog\|\s*Pass\|',data['report'])),label+' Verilog Pass report')
        tx=re.findall(r'^transaction\s+(\d+):\s+(\d+)\s+(\d+|x)\s*$',data['transactions'],re.M)
        need([int(t[0]) for t in tx]==list(range(27)),label+' 27 RTL transactions')
        cycles=[int(t[1]) for t in tx]
        need(all(t>0 for t in cycles),label+' positive observed latency')
        latencies.append(cycles)
    need(traces[0]==traces[1],'candidate matches all baseline status/payload/scale bits')
    need(candidate['ports']==baseline['ports'] and len(candidate['ports'])==18,'same local memory/control interface')
    valid=[i for i in range(min(len(traces[0]),27)) if ' status=0 ' in traces[0][i]]
    ratios=[]
    if len(latencies[0])==len(latencies[1])==27:
        ratios=[dict(case=i+1,baseline=latencies[0][i],candidate=latencies[1][i],
                     speedup=latencies[0][i]/latencies[1][i]) for i in valid if latencies[1][i]>0]
    return dict(status='FAIL' if errors else 'PASS_C_RTL_BIT_EXACT',errors=errors,
                transactions=27,successful=14,valid_case_cycles=ratios,
                baseline_latency_histogram=collections.Counter(latencies[0]),
                candidate_latency_histogram=collections.Counter(latencies[1]))
