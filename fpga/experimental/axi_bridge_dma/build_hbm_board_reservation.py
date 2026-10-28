"""Derive a non-release clock/pin reservation from reviewed native evidence."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from review_hbm_board_boundary import review


def build(root, out):
    if out.exists(): raise ValueError('Fresh contract directory required')
    evidence = review(root)
    contract = dict(status='EXPERIMENTAL_VU47P_HBM_BOARD_RESERVATION',
        part='xcvu47p-fsvh2892-2-e',source_checkpoint_sha256=evidence['checkpoint_sha256'],
        memory='16 GiB HBM, two stacks; DDR512GiB is a separate design',
        pcie=dict(profile='AXI Bridge',generation=3,lanes=16,block='PCIE4CE4_X0Y0',
                  reference_p='AR40',reference_n='AR41',reference_mhz=100,
                  perst_pin='BF41',perst_bank=65,connector_selected=False),
        reserved_clocks=evidence['clock_reservations'],
        trial_constraints=[dict(port=port,ball=ball,iostandard=standard)
            for port,ball,standard in [('sys_clk_p','AR40',None),('sys_clk_n','AR41',None),
                ('sys_rst_n','BF41','LVCMOS18'),('fabric_ref_clk_in','BK31','LVCMOS18')]],
        electrical_assumptions=dict(trial_bank64_vcco_v=1.8,trial_bank65_vcco_v=1.8,
            perst_translation_selected=False,clock_generator_selected=False,
            hbm_reference_jitter_requirement_ps_rms_lt=3,
            hbm_reference_source_requirement='GCIO in SLR0, via BUFG or MMCM/BUFG',
            spec='https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking',
            final_clock_io_standards_selected=False),
        boundary=dict(pad_bits_present=68,serial_pad_bits=64,other_pad_bits=4,
            existing_unbuffered_ports=evidence['ooc_boundary_ports'],
            missing_reference_input_buffers=['mgmt_clk_in','hbm_ref0_clk_in','hbm_ref1_clk_in'],
            sequencer_connections_pending=['configuration_rst_n','fabric_clock_reset','fabric_clock_locked']),
        implementation_gates=[
            'Verify trial package assignments with native opt/place reports before routing.',
            'Keep trial placement distinct from a board top: management and HBM reference input pads are absent.',
            'For differential management/HBM references, add input buffers at the reserved P/N pairs and select standards from the electrical design.',
            'Fabric reference currently uses a single-ended input buffer on BK31; BL31 remains reserved, not connected. Differential fabric input requires changing that buffer/IP configuration.',
            'Define clock source, jitter, bank voltages, PERST translation and cold-reset supervisor/sequencer before releasing the pinout.',
            'Repeat synthesis, native traffic, CDC, implementation and routed timing for the complete board wrapper.',
            'Verify all native and private CDC/data constraints after placement/routing; do not hide exceptions or physical DRCs.'],
        board_qualified=False,pinout_released=False,routed_timing_qualified=False,
        native_traffic_rerun=False,serial_lanes=evidence['serial_lanes'])
    out.mkdir(parents=True)
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    for name,key in [('clock_reservation.tsv','reserved_clocks'),('serial_lanes.tsv','serial_lanes'),('trial_pins.tsv','trial_constraints')]:
        table=contract[key]
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t')
            writer.writeheader();writer.writerows(table)
    return contract


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();value=build(a.root,a.out)
    print(json.dumps(dict(status=value['status'],clock_pairs=len(value['reserved_clocks']),serial_bits=len(value['serial_lanes']))))
