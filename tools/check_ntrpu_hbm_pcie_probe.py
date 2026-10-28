"""Read-only audit of the frozen VU47P XDMA generation (not traffic qualification)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


PART = 'xcvu47p-fsvh2892-2-e'
MANIFEST_SHA = '89a57c78f7a54b0504dcc52979b255ea8149ca760abcd63fa364446875a5e7f3'
NATIVE_RESULT = 'fa951962bbf66d1753c6f855b05af669d56a720da2ca31a4232172820f1af62a'
EXPECTED = dict(
    mode_selection='Advanced', functional_mode='DMA',
    pl_link_cap_max_link_width='X16', pl_link_cap_max_link_speed='8.0_GT/s',
    axi_data_width='512_bit', axi_addr_width='64', xdma_axi_intf_mm='AXI_Stream',
    ref_clk_freq='100_MHz', xdma_rnum_chnl='1', xdma_wnum_chnl='2',
    xdma_dsc_bypass='true', dsc_bypass_rd='0001', dsc_bypass_wr='0011',
    xdma_sts_ports='true', xdma_axilite_slave='true', soft_reset_en='false',
    axilite_master_en='true', axilite_master_scale='Kilobytes', axilite_master_size='64',
    enable_tandem_dfx='false', pf0_msix_enabled='true', pf0_msix_cap_table_size='01F',
    pf0_msix_cap_table_bir='BAR_1', pf0_msix_cap_table_offset='00008000',
    pf0_msix_cap_pba_bir='BAR_1', pf0_msix_cap_pba_offset='00008FE0',
    axil_master_64bit_en='false', xdma_pcie_64bit_en='false', axist_bypass_en='true',
    axist_bypass_size='1', axist_bypass_scale='Megabytes', axi_bypass_64bit_en='true',
    axi_bypass_prefetchable='false', pciebar2axibar_axist_bypass='0x0000000100000000')
BAR = [(9, 4), (9, 4), (13, 6), (0, 0), (0, 0), (0, 0)]
MSIX = dict(PF0_MSIX_CAP_TABLE_SIZE=31, PF0_MSIX_CAP_TABLE_BIR=1,
            PF0_MSIX_CAP_TABLE_OFFSET=0x8000, PF0_MSIX_CAP_PBA_BIR=1,
            PF0_MSIX_CAP_PBA_OFFSET=0x8fe0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name
            and ':' not in name and bool(p.parts), 'Unsafe path: ' + name)
    return p


def params(data, group):
    return {k: v[0]['value'] for k, v in
            json.loads(data)['ip_inst']['parameters'][group].items()}


def ports(text):
    result = {}
    for direction, high, low, name in re.findall(
            r'^\s*(input|output)\s+wire\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s+)?(\w+)\s*;',
            text, re.M):
        require(name not in result, 'Duplicate port: ' + name)
        result[name] = (direction, int(high) - int(low) + 1 if high else 1)
    return result


def check_top(text):
    actual = ports(text)
    expected = dict(axi_aclk=('output', 1), axi_aresetn=('output', 1),
                    user_lnk_up=('output', 1), sys_rst_n=('input', 1))
    for ch in range(2):
        for field, width in [('tdata', 512), ('tkeep', 64), ('tlast', 1), ('tvalid', 1)]:
            expected[f's_axis_c2h_{field}_{ch}'] = ('input', width)
        expected[f's_axis_c2h_tready_{ch}'] = ('output', 1)
        for field, width in [('src_addr', 64), ('dst_addr', 64), ('len', 28), ('ctl', 16), ('load', 1)]:
            expected[f'c2h_dsc_byp_{field}_{ch}'] = ('input', width)
        expected[f'c2h_dsc_byp_ready_{ch}'] = ('output', 1)
        expected[f'c2h_sts_{ch}'] = ('output', 8)
    for field, width in dict(awid=4, awaddr=64, awlen=8, awsize=3, awburst=2,
                            wdata=512, wstrb=64, bid=4, bresp=2, arid=4, araddr=64,
                            arlen=8, arsize=3, arburst=2, rid=4, rdata=512, rresp=2,
                            awvalid=1, awready=1, wvalid=1, wready=1, wlast=1,
                            bvalid=1, bready=1, arvalid=1, arready=1, rvalid=1,
                            rready=1, rlast=1).items():
        incoming = field in {'awready', 'wready', 'bid', 'bresp', 'bvalid', 'arready',
                             'rid', 'rdata', 'rresp', 'rvalid', 'rlast'}
        expected['m_axib_' + field] = ('input' if incoming else 'output', width)
    for name, value in expected.items():
        require(actual.get(name) == value, 'Port contract mismatch: ' + name)
    require(not any(re.fullmatch(r's_axis_c2h_tdata_[23]', x) for x in actual), 'Extra C2H channels')
    require(".C_PCIEBAR2AXIBAR_2(64'H0000000100000000)" in text, 'Lost high BAR translation bits')
    clock = re.search(r'XIL_INTERFACENAME CLK\.axi_aclk,[^\n]*FREQ_HZ (\d+)', text)
    require(clock and clock[1] == '125000000', 'Clock metadata changed; review discrepancy again')


def numeric_parameters(text):
    result = {}
    for name, value in re.findall(r'\.(\w+)\(\s*((?:\d+\s*\x27[Hh][0-9A-Fa-f]+)|\d+)\s*\)', text):
        result[name] = int(value.split("'")[1][1:], 16) if "'" in value else int(value)
    return result


def review_exports(blobs):
    snapshots = {}
    for profile in ['serial', 'pipe']:
        prefix = 'export/' + profile + '/'
        files = {k: v.decode('utf-8') for k, v in blobs.items() if k.startswith(prefix)
                 and k.endswith(('.tsv', '.rpt', '.xci', '.sv', '.v', '.xdc'))}
        wanted = dict(EXPECTED, pipe_sim=str(profile == 'pipe').lower())
        rows = [x.split('\t') for x in files[prefix + 'requested.tsv'].splitlines()[1:]]
        require(len(rows) == len(wanted), 'Requested parameter count')
        require({r[0]: r[1:] for r in rows} == {'CONFIG.' + k: [v, v] for k, v in wanted.items()},
                'Requested/actual parameter mismatch')
        config = dict(x.split('\t', 1) for x in files[prefix + 'config.tsv'].splitlines())
        require(config['PART'] == PART, 'Wrong FPGA part')
        for k, v in dict(wanted, axisten_freq='250', pcie_blk_locn='PCIE4C_X0Y0', select_quad='GTY_Quad_127').items():
            require(config['CONFIG.' + k] == v, 'Config mismatch: ' + k)
        status = files[prefix + 'ip_status.rpt']
        require(status.count('Up-to-date') == 5 and status.count('No changes required') == 5,
                'IP status not current')
        for filename, hard in [('ntrpu_vu47p_xdma.xci', False), ('ntrpu_vu47p_xdma_pcie4c_ip.xci', True)]:
            copies = [v for k, v in files.items() if k.endswith('/' + filename)]
            require(len(copies) == 2, 'Missing project/example XCI: ' + filename)
            for data in copies:
                project = params(data, 'project_parameters')
                for k, v in dict(DEVICE='xcvu47p', PACKAGE='fsvh2892', SPEEDGRADE='-2', TEMPERATURE_GRADE='E').items():
                    require(project[k] == v, 'XCI part mismatch: ' + k)
                component = params(data, 'component_parameters')
                model = params(data, 'model_parameters')
                require(model['PIPE_SIM'] == wanted['pipe_sim'].upper(), 'Wrong PIPE mode')
                require(model['PL_LINK_CAP_MAX_LINK_WIDTH'] == '16' and model['PL_LINK_CAP_MAX_LINK_SPEED'] == '4',
                        'Generated link encoding mismatch')
                if hard:
                    require(component['PL_LINK_CAP_MAX_LINK_SPEED'] == '8.0_GT/s', 'Hard IP link speed')
                    require(component['PL_LINK_CAP_MAX_LINK_WIDTH'] == 'X16', 'Hard IP link width')
                    for i, (aperture, control) in enumerate(BAR):
                        require(int(model[f'PF0_BAR{i}_APERTURE_SIZE'], 0) == aperture
                                and int(model[f'PF0_BAR{i}_CONTROL'], 0) == control, 'Hard BAR mismatch')
                    for k, v in MSIX.items():
                        require(int(model[k], 0) == v, 'Hard MSI-X mismatch: ' + k)
                else:
                    for k, v in dict(wanted, axisten_freq='250').items():
                        require(component[k] == v, 'Top XCI mismatch: ' + k)
                    require(int(model['C_PCIEBAR2AXIBAR_2'], 0) == 1 << 32, 'XCI translation mismatch')
                    if profile in snapshots:
                        require(snapshots[profile] == component, 'Example/project configuration differs')
                    snapshots[profile] = component
        for filename, hard in [('ntrpu_vu47p_xdma.sv', False), ('ntrpu_vu47p_xdma_pcie4c_ip.v', True)]:
            copies = [v for k, v in files.items() if k.endswith('/' + filename)]
            require(len(copies) == 4, 'Missing synth/sim wrapper: ' + filename)
            for text in copies:
                if not hard:
                    check_top(text)
                    continue
                values = numeric_parameters(text)
                for i, (aperture, control) in enumerate(BAR):
                    require(values[f'PF0_BAR{i}_APERTURE_SIZE'] == aperture
                            and values[f'PF0_BAR{i}_CONTROL'] == control, 'Wrapper BAR mismatch')
                require(all(values[k] == v for k, v in MSIX.items()), 'Wrapper MSI-X mismatch')
        clocks = [v for k, v in files.items() if k.endswith('/ntrpu_vu47p_xdma_ooc.xdc')]
        require(len(clocks) == 2, 'Missing OOC clock constraints')
        require(all(re.search(r'create_clock\s+-period 4\s+\[get_ports axi_aclk\]', v) for v in clocks),
                'OOC clock changed')
        examples = [v for k, v in files.items() if k.endswith('/xilinx_dma_pcie_ep.sv')]
        require(len(examples) == 1, 'Missing generated example')
        require(re.search(r"\.c2h_dsc_byp_load_1\s*\(1'b0\)", examples[0])
                and re.search(r'\.c2h_sts_1\s*\(\s*\)', examples[0]), 'Example channel 1 boundary changed')
    differences = {k for k in snapshots['serial'] if snapshots['serial'][k] != snapshots['pipe'].get(k)}
    require(differences == {'pipe_sim'}, 'Unexpected profile configuration difference')


def negative_controls(blobs):
    """Mutate in memory, beyond the hash gate, to exercise semantic rejection."""
    results = []
    top = next(k for k in blobs if k.startswith('export/serial/') and k.endswith('/ntrpu_vu47p_xdma.xci'))
    hard = next(k for k in blobs if k.startswith('export/serial/') and k.endswith('/ntrpu_vu47p_xdma_pcie4c_ip.xci'))
    changes = []
    for name, path, group, key, value in [
            ('wrong_part', top, 'project_parameters', 'DEVICE', 'xcvu5p'),
            ('wrong_pipe', top, 'model_parameters', 'PIPE_SIM', 'TRUE'),
            ('narrow_address', top, 'component_parameters', 'axi_addr_width', '32'),
            ('upper_bar_enabled', hard, 'model_parameters', 'PF0_BAR3_CONTROL', '0x4'),
            ('lost_translation_bits', top, 'model_parameters', 'C_PCIEBAR2AXIBAR_2', '0x0')]:
        data = json.loads(blobs[path])
        data['ip_inst']['parameters'][group][key][0]['value'] = value
        changes.append((name, path, json.dumps(data).encode()))
    wrapper = next(k for k in blobs if k.startswith('export/serial/') and k.endswith('/ntrpu_vu47p_xdma.sv'))
    changes.append(('missing_channel_1_status', wrapper, blobs[wrapper].replace(b'output wire [7 : 0] c2h_sts_1;', b'')))
    clock = next(k for k in blobs if k.startswith('export/serial/') and k.endswith('/ntrpu_vu47p_xdma_ooc.xdc'))
    changes.append(('wrong_clock', clock, blobs[clock].replace(b'-period 4 ', b'-period 8 ')))
    changes.append(('missing_profile_config', 'export/pipe/config.tsv', b'PART\tmissing\n'))
    for name, path, replacement in changes:
        require(replacement != blobs[path], 'Ineffective negative control: ' + name)
        changed = dict(blobs, **{path: replacement})
        try:
            review_exports(changed)
        except (ValueError, KeyError) as error:
            results.append(dict(name=name, rejected=True, reason=str(error)))
        else:
            raise ValueError('Accepted corrupt evidence: ' + name)
    return results


def audit(root):
    output = root / 'output'
    raw = (output / 'FILES.json').read_bytes()
    require(sha(raw) == MANIFEST_SHA, 'Not the frozen generation manifest')
    manifest = json.loads(raw)
    require(len(manifest) == 785, 'Wrong output inventory')
    blobs = {}
    for name, digest in manifest.items():
        safe_path(name)
        path = output / name
        require(not path.is_symlink() and path.resolve().is_relative_to(output.resolve()), 'Unsafe source path')
        blobs[name] = path.read_bytes()
        require(sha(blobs[name]) == digest, 'Output hash mismatch: ' + name)
    result = json.loads(blobs['result.json'])
    require(result['returncode'] == 0 and not result['timed_out'], 'Generation failed')
    require(result['status'] == 'EXPORTED_PENDING_REVIEW', 'Unexpected original result')
    require(result['sources'] == result['sources_after'], 'Source changed during generation')
    for name, digest in result['sources'].items():
        safe_path(name)
        require(sha((root / 'repo' / name).read_bytes()) == digest, 'Source hash mismatch: ' + name)
    dependency = json.loads((root / 'repo/DEPENDENCY.json').read_bytes())
    require(dependency['part'] == PART and dependency['native_hbm_not_in_probe'] is True
            and dependency['native_result_sha256'] == NATIVE_RESULT, 'Wrong HBM dependency')
    native = (root / 'support/NATIVE_OUTCOME.json').read_bytes()
    require(sha(native) == dependency['native_outcome_sha256'], 'Native provenance hash mismatch')
    console = blobs['console.log'].decode()
    for marker in ['HP GENERATED serial ' + PART, 'HP GENERATED pipe ' + PART, 'HP_PROBE_COMPLETE']:
        require(console.count(marker) == 1, 'Missing/duplicate generation marker')
    require(not re.search(r'^\s*(?:ERROR|WARNING|CRITICAL WARNING):', console, re.M), 'Vivado diagnostic requires review')
    review_exports(blobs)
    return dict(status='VU47P_XDMA_GENERATION_REVIEWED_NOT_INTEGRATED', part=PART,
                export_files=len(blobs), export_bytes=sum(map(len, blobs.values())),
                manifest_sha256=MANIFEST_SHA, result_sha256=sha(blobs['result.json']),
                source_sha256=result['sources'], requested_parameters_per_profile=len(EXPECTED) + 1,
                profiles=['serial', 'pipe'], checked_xci=8, checked_wrappers=16,
                configured_axi_mhz=250, wrapper_metadata_axi_mhz=125,
                example_channel_1_descriptors_tied_off=True,
                native_hbm_integrated=False, traffic_tested=False, physical_qualified=False,
                negative_controls=negative_controls(blobs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Stage directory containing repo/ and output/')
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
