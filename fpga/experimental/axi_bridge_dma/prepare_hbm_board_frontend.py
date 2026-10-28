"""Prepare a board-port candidate from the frozen registered-readiness top."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1: raise ValueError('Changed source anchor: '+old[:70])
    return text.replace(old, new, 1)


def prepare(donor, out):
    if out.exists(): raise ValueError('Fresh destination required')
    top=donor/'ntrpu_axi_bridge_hbm_controlled.sv'
    binding=json.loads((donor/'WEIGHT_PHYSICAL_BINDING.json').read_text())
    if sha(top)!=binding['physical_top_sha256'] or not binding['registered_readiness']:
        raise ValueError('Registered-readiness donor differs')
    text=top.read_text()
    text=once(text,'  input wire mgmt_clk_in,','  input wire mgmt_clk_p, mgmt_clk_n,')
    text=once(text,'  input wire fabric_clock_reset,','')
    text=once(text,'  input wire hbm_ref0_clk_in,','  input wire hbm_ref0_clk_p, hbm_ref0_clk_n,')
    text=once(text,'  input wire hbm_ref1_clk_in,','  input wire hbm_ref1_clk_p, hbm_ref1_clk_n,')
    text=once(text,'  output wire fabric_clock_locked,','')
    text=once(text,'  input wire  configuration_rst_n','  input wire supervisor_rst_n_pad')
    anchor='    BUFGCE management_buffer(.I(mgmt_clk_in),.CE(1\'b1),.O(mgmt_clk));'
    text=once(text,'    wire mgmt_clk,reader_clk,mem_clk,hbm_ref0,hbm_ref1;','')
    inserted='''    wire mgmt_clk,reader_clk,mem_clk,hbm_ref0,hbm_ref1;
    wire mgmt_clk_in, hbm_ref0_clk_in, hbm_ref1_clk_in;
    wire supervisor_rst_n, configuration_rst_n, fabric_clock_reset, fabric_clock_locked;
    wire board_lock_fault;
    ntrpu_hbm_board_inputs board_inputs(
        .mgmt_clk_p(mgmt_clk_p),.mgmt_clk_n(mgmt_clk_n),
        .hbm_ref0_clk_p(hbm_ref0_clk_p),.hbm_ref0_clk_n(hbm_ref0_clk_n),
        .hbm_ref1_clk_p(hbm_ref1_clk_p),.hbm_ref1_clk_n(hbm_ref1_clk_n),
        .supervisor_rst_n_pad(supervisor_rst_n_pad),.supervisor_rst_n(supervisor_rst_n),
        .mgmt_clk_in(mgmt_clk_in),.hbm_ref0_clk_in(hbm_ref0_clk_in),.hbm_ref1_clk_in(hbm_ref1_clk_in));
    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_board_startup board_startup(
        .mgmt_clk(mgmt_clk),.supervisor_rst_n(supervisor_rst_n),
        .fabric_locked_async(fabric_clock_locked),.fabric_clock_reset(fabric_clock_reset),
        .configuration_rst_n(configuration_rst_n),.lock_fault(board_lock_fault));
'''
    # Declare the new internal nets before all uses in the generated top.
    text=once(text,'    wire [12:0] ack_bytes;',inserted+'\n    wire [12:0] ack_bytes;')
    if text.count(anchor)!=1: raise ValueError('Existing management buffer differs')
    out.mkdir(parents=True)
    for p in donor.iterdir():
        if p.is_file():shutil.copy2(p,out/p.name)
    (out/top.name).write_text(text)
    here=Path(__file__).parent
    for n in ('ntrpu_hbm_board_inputs.sv','ntrpu_hbm_board_startup.sv'):
        shutil.copy2(here/n,out/n)
    tcl=out/'elaborate_hbm_native_join.tcl'
    script=once(tcl.read_text(),'ntrpu_hbm_loader_readiness ntrpu_axi_bridge_hbm_controlled}',
        'ntrpu_hbm_loader_readiness ntrpu_hbm_board_inputs ntrpu_hbm_board_startup ntrpu_axi_bridge_hbm_controlled}')
    for old,new in [('mgmt_clk_in management_refclk','mgmt_clk_p management_refclk'),
                    ('hbm_ref0_clk_in hbm_reference0_clk','hbm_ref0_clk_p hbm_reference0_clk'),
                    ('hbm_ref1_clk_in hbm_reference1_clk','hbm_ref1_clk_p hbm_reference1_clk')]:
        script=once(script,old,new)
    tcl.write_text(script)
    record=dict(status='BOARD_FRONTEND_PREPARED_NOT_NATIVE_VALIDATED',
        donor_top_sha256=sha(top),candidate_top_sha256=sha(out/top.name),
        source_files={n:sha(out/n) for n in ('ntrpu_hbm_board_inputs.sv','ntrpu_hbm_board_startup.sv')},
        global_clock_buffers_added=0,differential_input_buffers_added=3,
        supervisor_input_buffers_added=1,perst_wiring_unchanged=True,
        startup_only_cold_release=True,lock_loss_clears_epoch=False,
        pinout_released=False,native_synthesis_executed=False,native_traffic_executed=False)
    (out/'BOARD_FRONTEND_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('donor',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.donor,a.output)))
