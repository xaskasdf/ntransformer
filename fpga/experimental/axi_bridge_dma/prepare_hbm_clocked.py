"""Freeze the selected reader and expose existing canonical cold reset outputs.

Only new files are written. The guard/client variants differ exclusively by
module names and an output alias; the selected reader is copied byte for byte.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

PINNED = {
    'ntrpu_axi_bridge_guard_v3.sv': '0f495c618cbf4bc37356445af9b7bd0600039e7b75b3ba0fd95f06d050472e24',
    'ntrpu_axi_bridge_hbm_client.sv': 'c3ef3b58e798187f647b7faf0d7008cda47833768c4daad256e229d297bd77a5',
    'ntrpu_axi_bridge_hbm_cdc.sv': '87a2deca7c6d3c7bedeb4bab640784d77b052b6e3d08958207657917ea0d9f9f',
    'ntrpu_axi_bridge_hbm_source.sv': '0454df4e72a9007b59274a0eb814b756e8b8bc90d581f4df39fee900d8d6a9f6',
    'ntrpu_axi_bridge_endpoint_client.sv': '0a3c338a09046b21a71bc6eff951a98990a445310d8e39f3ec24249a6899321b',
    'ntrpu_axi_bridge_writer.sv': '885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6',
    'ntrpu_axi_bridge_mapper.sv': 'f832d7e777cffc617499ee46e114a9aa9b6a6c77d3f059971b6836b2466f95c9',
}
READER_NAMES = ['ntrpu_hbm_read_return','ntrpu_hbm_read_frames','ntrpu_hbm_read_adapter','ntrpu_hbm_read_return_cut']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Changed source anchor: '+old)
    return text.replace(old,new)

def variants(guard,client):
    guard=replace_once(guard,'module ntrpu_axi_bridge_guard_v3','module ntrpu_axi_bridge_guard_v3_reset_export')
    guard=replace_once(guard,'output wire client_cold_rst_n,','output wire management_cold_rst_n, client_cold_rst_n,')
    guard=replace_once(guard,'assign client_cold_rst_n=cold_a[2];','assign client_cold_rst_n=cold_a[2];\n    assign management_cold_rst_n=cold_m[2];')
    client=replace_once(client,'module ntrpu_axi_bridge_hbm_client','module ntrpu_axi_bridge_hbm_client_reset_export')
    client=replace_once(client,'output wire  client_cold_rst_n,','output wire  client_cold_rst_n,\n    output wire  management_cold_rst_n,')
    client=replace_once(client,'ntrpu_axi_bridge_guard_v3 #','ntrpu_axi_bridge_guard_v3_reset_export #')
    client=replace_once(client,'.client_cold_rst_n(client_cold_rst_n),','.management_cold_rst_n(management_cold_rst_n),.client_cold_rst_n(client_cold_rst_n),')
    return guard,client

def composition(client,reader):
    header=client[:client.index(');')]
    header=replace_once(header,'module ntrpu_axi_bridge_hbm_client','module ntrpu_axi_bridge_hbm_clocked')
    cut=header.index('    input wire  reader_started,')
    # Requester/control/ACK ports are preserved, reader ports become internal.
    header=header[:cut]+'''    input wire reader_clk,mem_clk,
    input wire [1:0] init_complete_async,
    output wire management_cold_rst_n,reader_cold_rst_n,mem_cold_rst_n,
    output wire source_owned,source_fault,
    output wire reader_started_raw,reader_fault_raw,reader_drained,reader_stopped,
    output wire reader_clock_alive,reader_management_fault,
    output wire mem_clock_alive,mem_armed,mem_management_fault,
    output wire [1:0] frame_idle,adapter_pending,adapter_fault,
'''
    native=reader[reader.index('    output wire [1:0] h_arvalid,'):reader.index(');')]
    header+=native+');\n'
    ports={}
    original=client[:client.index(');')]
    for direction,width,name in re.findall(r'^\s*(input|output) wire\s+(\[[^]]+\]\s*)?(\w+),?\s*$',original,re.M):
        if name.startswith('r_'):ports[name]=(width or '').strip()
    if len(ports)!=15:raise ValueError('Reader interface count changed')
    body='    wire reader_started,reader_fault,reader_stop,cdc_started,cdc_fault,cdc_stop;\n'
    body+='    wire mem_permit_p,mem_fault_p,mem_fault_r;\n'
    for name,width in ports.items():
        body+=f'    wire {width} {name},{name.replace("r_","d_",1)};\n'
    body+='''    // All four releases originate from one physical cold epoch. No extra
    // release chain is added for CDC, watchdogs, or native memory interfaces.
    (* KEEP_HIERARCHY="YES" *) ntrpu_axi_bridge_hbm_client_reset_export #(
        .AXI_ADDR_BITS(AXI_ADDR_BITS),.WAIT_LIMIT(WAIT_LIMIT),
        .SHARED_NATIVE_RESET(SHARED_NATIVE_RESET),.APERTURE_BASE(APERTURE_BASE)) endpoint(.*);
    assign reader_started=cdc_started&&mem_permit_p;
    assign reader_fault=cdc_fault||mem_fault_p;
    (* KEEP_HIERARCHY="YES" *) ntrpu_axi_bridge_hbm_mem_watch memory_watch(
        .mgmt_clk(mgmt_clk),.mem_clk(mem_clk),.p_clk(clk),.r_clk(reader_clk),
        .m_cold_rst_n(management_cold_rst_n),.mem_cold_rst_n(mem_cold_rst_n),
        .p_cold_rst_n(client_cold_rst_n),.r_cold_rst_n(reader_cold_rst_n),
        .clock_alive(mem_clock_alive),.armed(mem_armed),.fault(mem_management_fault),
        .p_permit(mem_permit_p),.p_fault(mem_fault_p),.r_fault(mem_fault_r));
    (* KEEP_HIERARCHY="YES" *) ntrpu_axi_bridge_hbm_cdc transport(
        .p_clk(clk),.r_clk(reader_clk),.mgmt_clk(mgmt_clk),
        .p_cold_rst_n(client_cold_rst_n),.r_cold_rst_n(reader_cold_rst_n),
        .m_cold_rst_n(management_cold_rst_n),.p_stop(reader_stop),
        .r_started(reader_started_raw),.r_fault(reader_fault_raw),
        .p_started(cdc_started),.p_fault(cdc_fault),.r_stop(cdc_stop),
        .clock_alive(reader_clock_alive),.armed(),.management_fault(reader_management_fault),
'''
    body+=',\n'.join(f'        .p_{name[2:]}({name}),.r_{name[2:]}({name.replace("r_","d_",1)})' for name in ports)+');\n'
    body+='''    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_read_return reader(
        .clk(reader_clk),.mem_clk(mem_clk),.cold_reset_n(configuration_rst_n),
        .init_complete_async(init_complete_async),.stop(cdc_stop),.fault_in(mem_fault_r),
        .rst_n(reader_cold_rst_n),.mem_rst_n(mem_cold_rst_n),
        .started(reader_started_raw),.fault(reader_fault_raw),
        .drained(reader_drained),.stopped(reader_stopped),
'''
    body+=',\n'.join(f'        .{name[2:]}({name.replace("r_","d_",1)})' for name in ports)+',.*);\n'
    return header+body+'endmodule\n`default_nettype wire\n'

def prepare(here,selected,root,repo):
    if root.exists():raise ValueError('New prepared directory required')
    profile=json.loads((selected/'HEAD_NATIVE_PROFILE.json').read_text())
    inputs={n:here/n for n in PINNED}
    for n in READER_NAMES:inputs[n+'.sv']=selected/f'fpga/experimental/{n}.sv'
    inputs['ntrpu_hbm_startup.sv']=selected/'fpga/rtl/ntrpu_hbm_startup.sv'
    inputs['ntrpu_bridge_async_fifo.sv']=repo/'fpga/experimental/hbm_dma_rev2/ntrpu_bridge_async_fifo.sv'
    inputs['ntrpu_xdma_local_master.sv']=repo/'fpga/rtl/ntrpu_xdma_local_master.sv'
    for n in PINNED:
        if sha(inputs[n])!=PINNED[n]:raise ValueError('Frozen input changed: '+n)
    for n in READER_NAMES+['ntrpu_hbm_startup']:
        p=inputs[n+'.sv'];key=p.relative_to(selected).as_posix()
        if sha(p)!=profile['selected_rtl'][key]:raise ValueError('Selected reader changed: '+key)
    if sha(inputs['ntrpu_bridge_async_fifo.sv'])!='be809fe998385e69a52eb2fa8687b5b50e52433a5b1a42694331efc07f1f0634':raise ValueError('FIFO changed')
    if sha(inputs['ntrpu_xdma_local_master.sv'])!='5318eb1381e87224f0dafe7d31aec4067df800b3229a1fad6f6b4614d685d3fc':raise ValueError('Master changed')
    root.mkdir(parents=True)
    for n,p in inputs.items():shutil.copy2(p,root/n)
    shutil.copy2(selected/'HEAD_NATIVE_PROFILE.json',root/'HEAD_NATIVE_PROFILE.json')
    guard,client=variants(inputs['ntrpu_axi_bridge_guard_v3.sv'].read_text(),inputs['ntrpu_axi_bridge_hbm_client.sv'].read_text())
    (root/'ntrpu_axi_bridge_guard_v3_reset_export.sv').write_text(guard)
    (root/'ntrpu_axi_bridge_hbm_client_reset_export.sv').write_text(client)
    (root/'ntrpu_axi_bridge_hbm_clocked.sv').write_text(composition(inputs['ntrpu_axi_bridge_hbm_client.sv'].read_text(),inputs['ntrpu_hbm_read_return.sv'].read_text()))
    save(root/'PREPARED.json',dict(status='CLOCKED_COMPOSITION_PREPARED',inputs={n:sha(p) for n,p in inputs.items()},
        generated={p.name:sha(p) for p in root.glob('*.sv') if p.name not in inputs},
        scope='Native AXI edges; real AMD reader converters generated by synthesis runner; HBM hard IP and PCIe endpoint not included',
        cold_owners={'pcie':'endpoint/guard/cold_a[2]','management':'endpoint/guard/cold_m[2]',
                     'reader':'reader/startup/source_release[2]','memory':'reader/startup/memory_release[2]'},
        physical_clocks_qualified=False,board_qualified=False))
    return root

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);ap.add_argument('--selected',type=Path,required=True)
    ap.add_argument('--repo',type=Path,required=True);a=ap.parse_args()
    print(prepare(Path(__file__).resolve().parent,a.selected,a.output,a.repo))
