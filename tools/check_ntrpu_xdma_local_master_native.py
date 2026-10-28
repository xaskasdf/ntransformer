"""Correlate direct RTL master ownership/reports with independent native XDMA traffic."""
import re
from check_ntrpu_xdma_local_management_direct_discovery import review as review_traffic
from check_ntrpu_xdma_stop_inventory_r4 import require, markers, vcd_rows


def review(sim,vcd):
    result=review_traffic(sim,vcd)
    p='/board/ntrpu_stop_probe/'
    require(all(re.search(r'\$var \w+ \d+ \S+ '+n+r'(?: |\[)',vcd.split('$enddefinitions $end')[0])
                for n in ['master_req_valid','master_req_identity','master_rsp_valid','master_rsp_identity','master_busy']),
            'synthesizable master public trace required')
    require(markers(sim,'XDMA_MASTER_NATIVE_PASS')==[dict(t=str(result['finish_ps']),requests='44',reports='44',native_xdma='1',reset_tested='0')], 'direct native-master scope')
    transactions=sorted([dict(r,write=0) for r in result['local_reads']]+[dict(w,write=1) for w in result['local_writes']],
                        key=lambda r:r.get('request_ps',r.get('aw_ps')))
    require(len(transactions)==44,'native master transaction inventory')
    pending=None;requests=[];reports=[];measurements=[]
    for t,before,after in vcd_rows(vcd):
        if t<result['start_ps']:continue
        require(after[p+'master_timeout_latched']==after[p+'master_rsp_timeout']==0,'no unreported master timeout')
        edge=before['/board/EP/user_clk']==0 and after['/board/EP/user_clk']==1
        if edge:
            if before[p+'master_req_valid'] and before[p+'master_req_ready']:
                require(pending is None and len(requests)<44,'one master owner at a time')
                row=transactions[len(requests)];serial=len(requests)+1
                identity=(0x4e545250554d4754<<64)|(row['phase']<<32)|serial
                request=dict(t=t,phase=before[p+'phase'],write=before[p+'master_req_write'],
                             address=before[p+'master_req_address'],data=before[p+'master_req_data'],
                             strobe=before[p+'master_req_strobe'],identity=before[p+'master_req_identity'])
                require([request[k] for k in ['phase','write','address','data','strobe','identity']]==
                        [row['phase'],row['write'],row['address'],row['value'] if row['write'] else 0,15 if row['write'] else 0,identity],
                        'request/native register provenance')
                require(t<row.get('request_ps',min(row.get('aw_ps',0),row.get('w_ps',0))),'RTL request precedes native AXI request')
                requests.append(request)
                pending=dict(row=row,request=request,bus_seen=False,stalls=0,closed=False)
            if before[p+'master_rsp_valid'] and not before[p+'master_rsp_ready']:
                require(pending is not None and pending['bus_seen'],'no invented report under backpressure')
                pending['stalls']+=1
            if before[p+'master_rsp_valid'] and before[p+'master_rsp_ready']:
                require(pending is not None and pending['bus_seen'] and pending['closed'],'report after real response and admission close')
                require(pending['stalls']>=3,'actual retained master report')
                row=pending['row'];req=pending['request']
                report=dict(t=t,phase=before[p+'phase'],data=before[p+'master_rsp_data'],code=before[p+'master_rsp_code'],
                            identity=before[p+'master_rsp_identity'],timeout=before[p+'master_rsp_timeout'])
                require([report[k] for k in ['phase','data','code','identity','timeout']]==
                        [req['phase'],0 if row['write'] else row['value'],0,req['identity'],0],'native report bytes/error/identity')
                reports.append(report)
                measurements.append(dict(serial=len(reports),phase=req['phase'],write=req['write'],address=req['address'],
                    request_ps=req['t'],native_response_ps=row['response_ps'],report_ps=t,
                    request_to_native_ps=row['response_ps']-req['t'],report_stall_cycles=pending['stalls']))
                pending=None
            for channel in ('b','r'):
                if before[p+'local_'+channel+'valid'] and before[p+'local_'+channel+'ready']:
                    require(pending is not None and not pending['bus_seen'],'one real bus response per RTL request')
                    require(pending['row']['response_ps']==t and pending['row']['write']==int(channel=='b'),'native/RTL response binding')
                    pending['bus_seen']=True
        require(after[p+'master_requests']==len(requests) and after[p+'master_reports']==len(reports),'master counters/VCD')
        require(after[p+'master_busy']==int(pending is not None),'master ownership until report consumed')
        require(after[p+'master_req_ready']==int(after[p+'master_admit'] and pending is None),'admission closes without dropping owner')
        require(after[p+'master_rsp_valid']==int(pending is not None and pending['bus_seen']),'report follows actual B/R only')
        if pending:
            pending['closed'] |= after[p+'master_admit']==0
            if pending['bus_seen']:
                row=pending['row'];req=pending['request']
                require([after[p+n] for n in ['master_rsp_data','master_rsp_code','master_rsp_identity','master_rsp_timeout']]==
                        [0 if row['write'] else row['value'],0,req['identity'],0],'continuous report retention')
    require(pending is None and len(requests)==len(reports)==44,'all native-master obligations retired')
    for name,expected in [('REQ',requests),('REPORT',reports)]:
        actual=[{k:int(v,10 if k in ('t','phase','write','timeout') else 16) for k,v in r.items()}
                for r in markers(sim,'XDMA_MASTER_'+name)]
        require(actual==expected,'master log/VCD correspondence '+name)
    result.update(status='PASS_SYNTHESIZABLE_LOCAL_MASTER_NATIVE_XDMA',master_transactions=measurements,
        qualified_native_master_bounded=True,
        scope='Direct synthesizable master drives 44 native register transactions while BAR/H2C/C2H responses are held; retained tagged reports checked. No driver certificate, protected global inventory, operational reset, physical timing or inference qualification.')
    return result
