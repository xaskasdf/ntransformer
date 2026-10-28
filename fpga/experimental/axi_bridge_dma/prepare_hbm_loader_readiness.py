"""Prepare a registered-readiness candidate without changing the sealed donors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:raise ValueError('Changed anchor: '+old[:80])
    return text.replace(old,new,1)


def prepare(donor, out, mode):
    if out.exists():raise ValueError('Fresh output required')
    here=Path(__file__).parent
    loader=here/'ntrpu_hbm_weight_loader.sv'
    assert sha(loader)=='14aca9b0b01fbb575671e4399a76a35613010b7a2e16b1e0637b9dbb3bda1103'
    out.mkdir()
    if mode=='physical':
        top=donor/'ntrpu_axi_bridge_hbm_controlled.sv'
        assert sha(top)=='02d3c8b39e75c6b77b3263ac72edd65a300c06f2caa6cdd434e3ec218fb714bd'
        for p in donor.iterdir():
            if p.is_file():shutil.copy2(p,out/p.name)
        original=top.read_text()
        text=once(original,'    wire weights_sealed;','''    wire weights_sealed;
    wire loader_ready_mgmt;
    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_loader_readiness loader_readiness(
        .clk(mgmt_clk),.cold_rst_n(management_cold_rst_n),
        .ready_mgmt(init_complete_async),.fault_mgmt(availability_fault),
        .ready(loader_ready_mgmt));''')
        text=once(text,'.memory_ready_async((&init_complete_async)&&!availability_fault)', '.memory_ready_async(loader_ready_mgmt)')
        (out/top.name).write_text(text)
        tcl=out/'elaborate_hbm_native_join.tcl'
        tcl.write_text(once(tcl.read_text(),'ntrpu_hbm_weight_loader ntrpu_axi_bridge_hbm_controlled}',
            'ntrpu_hbm_weight_loader ntrpu_hbm_loader_readiness ntrpu_axi_bridge_hbm_controlled}'))
        binding=json.loads((out/'WEIGHT_PHYSICAL_BINDING.json').read_text())
        for name,digest in binding['frozen_rtl'].items():assert sha(out/name)==digest
        binding.update(functional_attachment_identical=False,registered_readiness=True,
            original_physical_top_sha256=sha(top),physical_top_sha256=sha(out/top.name),
            status='REGISTERED_READINESS_CANDIDATE_REQUIRES_VALIDATION')
        (out/'WEIGHT_PHYSICAL_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
    else:
        for name in ('ntrpu_control_snapshot.sv','ntrpu_hbm_control_plane.sv',
                     'ntrpu_hbm_weight_loader.sv','ntrpu_hbm_native_availability.sv'):
            shutil.copy2(here/name,out/name)
        text=(here/'tb_hbm_weight_loader.sv').read_text()
        text=once(text,'reg cold_rst_n=0,bus_rst_n=0,mem_cold_rst_n=0,memory_ready_async=0;', '''reg cold_rst_n=0,bus_rst_n=0,mem_cold_rst_n=0;
    reg mgmt_clk=0,mgmt_enable=1,fabric_locked=0;
    reg [1:0] apb_complete=0;
    wire [1:0] ready_mgmt;
    wire availability_fault,availability_armed,memory_ready_async;
    always #5 if(mgmt_enable)mgmt_clk=~mgmt_clk;else mgmt_clk=0;
    ntrpu_hbm_native_availability availability(.mgmt_clk(mgmt_clk),.cold_rst_n(cold_rst_n),
        .fabric_locked_async(fabric_locked),.apb_complete_mgmt(apb_complete),
        .ready_mgmt(ready_mgmt),.armed(availability_armed),.fault(availability_fault));
    ntrpu_hbm_loader_readiness source(.clk(mgmt_clk),.cold_rst_n(cold_rst_n),
        .ready_mgmt(ready_mgmt),.fault_mgmt(availability_fault),.ready(memory_ready_async));
    integer registered_samples=0;
    reg expected_ready;
    always @(posedge mgmt_clk)begin
        expected_ready=cold_rst_n&&(&ready_mgmt)&&!availability_fault;
        #0.001;
        require(memory_ready_async===expected_ready,"registered readiness latency");
        registered_samples++;
        $display("WR SAMPLE t=%0t expect=%b actual=%b",$time,expected_ready,memory_ready_async);
    end''')
        text=text.replace('memory_ready_async=0;', 'fabric_locked=0;apb_complete=0;')
        text=text.replace('memory_ready_async=1;', 'fabric_locked=1;apb_complete=3;')
        text=once(text,'bad_id=0;mem_enable=1;', 'bad_id=0;mem_enable=1;mgmt_enable=1;')
        text=once(text,'        $display("WL PASS seed=',
            '        `include "loader_readiness_scenarios.svh"\n        require(registered_samples>100,"registered source observed");\n        $display("WR SOURCE samples=%0d",registered_samples);\n        $display("WL PASS seed=')
        (out/'tb_hbm_weight_loader.sv').write_text(text)
        shutil.copy2(here/'loader_readiness_scenarios.svh',out/'loader_readiness_scenarios.svh')
        runner=(here/'run_hbm_weight_loader.py').read_text()
        runner=once(runner,"'ntrpu_hbm_weight_loader', 'tb_hbm_weight_loader'", "'ntrpu_hbm_weight_loader', 'ntrpu_hbm_native_availability', 'ntrpu_hbm_loader_readiness', 'tb_hbm_weight_loader'")
        runner=once(runner,"['xvlog', '--sv',", "['xvlog', '--sv', '-i', str(source),")
        runner=once(runner,"int(match[0][1]) != 8", "int(match[0][1]) != 14")
        (out/'run_hbm_weight_loader.py').write_text(runner)
    shutil.copy2(here/'ntrpu_hbm_loader_readiness.sv',out/'ntrpu_hbm_loader_readiness.sv')
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    report=dict(mode=mode,loader_sha256=sha(loader),readiness_sha256=sha(out/'ntrpu_hbm_loader_readiness.sv'),
        source_register_added=True,additional_management_cycles=1,baseline_rtl_unchanged=True,
        native_traffic_rerun=False,board_qualified=False)
    (out/'READINESS_BINDING.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('donor',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=('physical','preflight'),required=True)
    a=p.parse_args();print(json.dumps(prepare(a.donor,a.output,a.mode)))
