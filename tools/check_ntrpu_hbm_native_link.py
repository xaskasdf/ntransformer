"""Audit native port bindings and RTL elaboration, not memory functionality."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

WIDTHS={'awaddr':34,'awburst':2,'awid':6,'awlen':4,'awsize':3,'awvalid':1,'awready':1,
        'wdata':256,'wstrb':32,'wlast':1,'wvalid':1,'wready':1,'bid':6,'bresp':2,'bvalid':1,'bready':1}
RETURNS={'awready','wready','bid','bresp','bvalid'}

def binding(template,source):
    ports={}
    for match in re.finditer(r'\.([A-Za-z0-9_]+)\([^)]*\),?\s*// (input|output) wire(?: \[(\d+)\s*:\s*0\])? (\w+)',template):
        n,d,hi,label=match.groups();assert n==label;ports[n]=(d,int(hi)+1 if hi else 1)
    connections=dict(re.findall(r'\.([A-Za-z0-9_]+)\(([^)]*)\)',source))
    assert ports.keys()==connections.keys()
    for port,(direction,width) in ports.items():
        match=re.fullmatch(r'AXI_(\d+)_(\w+)',port)
        if not match:assert connections[port]==port;continue
        index=int(match[1]);name=match[2].lower()
        if name=='aclk':expected='axi_clk'
        elif name=='areset_n':expected='axi_cold_reset_n'
        elif index in (0,16) and name in WIDTHS:
            assert width==WIDTHS[name] and direction==('output' if name in RETURNS else 'input')
            lane=int(index==16);expected='s_'+name+(f'[{lane*width}+:{width}]' if width>1 else f'[{lane}]')
        elif index in (0,16) and name=='wdata_parity':expected=f'parity[{int(index==16)*32}+:32]'
        elif direction=='output':expected=''
        elif name in ('arburst','awburst'):expected=f"{width}'d1"
        elif name in ('arsize','awsize'):expected=f"{width}'d5"
        elif name in ('rready','bready'):expected="1'b1"
        else:expected=f"{width}'b0"
        assert connections[port]==expected,port
    assert set(range(32))=={int(n[4:6]) for n in ports if n.startswith('AXI_')}
    assert 'assign parity[k]=^s_wdata[k*8+:8];' in source
    return connections

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();out=a.output;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((out/'result.json').read_text());files=json.loads((out/'FILES.json').read_text())
    assert all(sha(out/n)==h for n,h in files.items())
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['stage']=='link' and r['sources']==r['sources_after']
    props=dict(l.split('\t',1) for l in (out/'suite/config.tsv').read_text().splitlines())
    expected=dict(USER_HBM_DENSITY='16GB',USER_HBM_STACK='2',USER_STACK_8HI='true',USER_MEMORY_DISPLAY='16384',
                  USER_SWITCH_ENABLE_00='TRUE',USER_SWITCH_ENABLE_01='TRUE',USER_AXI_CLK_FREQ='450',USER_AXI_CLK1_FREQ='450')
    assert all(props['CONFIG.'+n]==v for n,v in expected.items())
    for i in range(16):assert props[f'CONFIG.USER_MC_ENABLE_{i:02}']=='TRUE'
    template=(out/'suite/native_hbm.veo').read_text();source=(out/'suite/binding/ntrpu_hbm_native_link.sv').read_text()
    connections=binding(template,source);manifest=json.loads((out/'suite/binding/BINDING.json').read_text())
    assert connections==manifest['connections'] and manifest['binding_sha256']==sha(out/'suite/binding/ntrpu_hbm_native_link.sv')
    assert manifest['native_template_sha256']==sha(out/'suite/native_hbm.veo')
    assert (out/'suite/elaboration.txt').read_text().splitlines()==['TOP ntrpu_hbm_native_link','PART xcvu47p-fsvh2892-2-e','HBM_INSTANCE memory']
    log=(out/'console.log').read_text();assert 'RTL Elaboration Complete' in log and 'NTRPU_NATIVE_HBM_PROBE_COMPLETE link' in log
    assert not re.search(r'^(ERROR:|CRITICAL WARNING:)',log,re.M)
    warnings=[l for l in log.splitlines() if l.startswith('WARNING:')]
    assert all('ntrpu_hbm_native_link' not in l for l in warnings)
    negatives=[]
    for old,new,label in [('.AXI_16_WDATA(s_wdata[256+:256])','.AXI_16_WDATA(s_wdata[0+:256])','crossed_lane'),
                          ('.AXI_16_BID(s_bid[6+:6])','.AXI_16_BID()','lost_bid')]:
        assert source.count(old)==1
        try:binding(template,source.replace(old,new))
        except AssertionError:negatives.append(label)
        else:raise AssertionError('Undetected native binding error')
    result=dict(status='NATIVE_HBM_BINDING_ELABORATED',seconds=r['seconds'],part='xcvu47p-fsvh2892-2-e',
        enabled_memory_mib=16384,active_write_ports=[0,16],pins=len(connections),axi_clock_profile_mhz=450,
        warning_groups=dict(Counter(re.search(r'\[([^]]+)\]',l)[1] for l in warnings)),
        negative_controls=negatives,checker_sha256=sha(Path(__file__)),
        hard_controller_instantiated=True,full_synthesis=False,initialized=False,traffic_simulated=False,
        transport_composed=False,cdc=False,qualified_timing=False,
        scope='Native IP and two-port wiring elaborate; vendor internal undriven/unused warnings retained for full synthesis review')
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
