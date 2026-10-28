"""Review generated XDMA configuration only; never claim functional P2P."""
import hashlib
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameters(xci, group):
    return {k: v[0]['value'] for k, v in xci['ip_inst']['parameters'][group].items()}


def review(evidence, baseline):
    inv = evidence['invocation']
    require(inv.get('exit_code') == 0 and 'finished_utc' in inv, 'not successful terminal generation')
    require(inv['script'] == 'tools/ntrpu_board_pcie_bypass_probe.tcl', 'wrong generator')
    require(inv['arguments'][0] == 'xcvu5p-flva2104-2-i', 'wrong device')
    require(inv['max_threads'] == 4 and inv['timeout_seconds'] == 600, 'resource contract')
    lines = evidence['console'].splitlines()
    require(lines.count('NTRPU_PCIE_BYPASS_EXAMPLE_GENERATED') == 1, 'missing generation outcome')
    require(not any(s.startswith(('ERROR:', 'WARNING:', 'CRITICAL WARNING:')) for s in lines),
            'unreviewed generator diagnostic')
    source = evidence['source_script']
    require(hashlib.sha256(source.encode()).hexdigest() == evidence['source_script_sha256'],
            'generator hash mismatch')
    expected = dict(re.findall(r'CONFIG\.(\w+)\s+([^\s\\\]]+)', source))
    require(len(expected) == 29, 'incomplete request contract')
    require(expected['axist_bypass_en'] == 'true' and expected['axi_bypass_64bit_en'] == 'true'
            and expected['pciebar2axibar_axist_bypass'] == '0x0000000100000000', 'wrong bypass request')
    table = evidence['reports']['requested_properties.tsv'].splitlines()
    require(table[0] == 'property\trequested\tactual' and len(table) == 30, 'request table incomplete')
    actual_table = {}
    for line in table[1:]:
        key, requested, actual = line.split('\t')
        require(key.removeprefix('CONFIG.') not in actual_table and requested == actual, 'request was not honored')
        actual_table[key.removeprefix('CONFIG.')] = actual
    require(actual_table == expected, 'request table differs from generator')
    original = parameters(baseline, 'component_parameters')
    wanted_delta = {'axist_bypass_en': ('false', 'true'),
                    'axi_bypass_64bit_en': ('false', 'true'),
                    'pciebar2axibar_axist_bypass': ('0x0000000000000000', '0x0000000100000000')}
    top = [(n, x) for n, x in evidence['xcis'].items() if n.endswith('/ntrpu_xdma.xci')]
    hard = [(n, x) for n, x in evidence['xcis'].items() if n.endswith('/ntrpu_xdma_pcie4_ip.xci')]
    require(len(top) == len(hard) == 2, 'need project and example XCI')
    for name, xci in top:
        c = parameters(xci, 'component_parameters')
        require(all(c[k] == v for k, v in expected.items()), 'XCI request mismatch: ' + name)
        delta = {k: (original.get(k), c.get(k)) for k in set(original) | set(c) if original.get(k) != c.get(k)}
        require(delta == wanted_delta, 'unexpected baseline configuration change: ' + name)
        m = parameters(xci, 'model_parameters')
        require(m['XDMA_AXIST_BYPASS'] == 'TRUE' and int(m['AXIST_BYPASS_CONTROL'], 16) == 5,
                'bypass wrapper not enabled/64-bit')
        for bar in range(7):
            require(int(m[f'C_PCIEBAR2AXIBAR_{bar}'], 16) == ((1 << 32) if bar == 2 else 0),
                    'wrong BAR translation')
    expected_bars = [(9, 4), (9, 4), (13, 6), (0, 0), (0, 0), (0, 0)]
    for name, xci in hard:
        m = parameters(xci, 'model_parameters')
        require([(int(m[f'PF0_BAR{i}_APERTURE_SIZE'], 16), int(m[f'PF0_BAR{i}_CONTROL'], 16))
                 for i in range(6)] == expected_bars, 'effective hard-block BAR map changed')
        for key, value in {'PF0_MSIX_CAP_TABLE_BIR': 1, 'PF0_MSIX_CAP_PBA_BIR': 1,
                           'PF0_MSIX_CAP_TABLE_SIZE': 31, 'PF0_MSIX_CAP_TABLE_OFFSET': 0x8000,
                           'PF0_MSIX_CAP_PBA_OFFSET': 0x8fe0}.items():
            require(int(m[key], 0) == value, 'effective MSI-X mapping changed')
    widths = {'awid': 4, 'awaddr': 64, 'awlen': 8, 'awsize': 3, 'awburst': 2,
              'wdata': 512, 'wstrb': 64, 'bid': 4, 'bresp': 2,
              'arid': 4, 'araddr': 64, 'arlen': 8, 'arsize': 3, 'arburst': 2,
              'rid': 4, 'rdata': 512, 'rresp': 2}
    checked_top = checked_hard = 0
    for name, text in evidence['wrapper_texts'].items():
        require(hashlib.sha256(text.encode()).hexdigest() == evidence['generated_file_hashes'][name],
                'wrapper provenance mismatch: ' + name)
        if name.endswith('/ntrpu_xdma.sv'):
            checked_top += 1
            for port, width in widths.items():
                direction = 'input' if port in {'bid', 'bresp', 'rid', 'rdata', 'rresp'} else 'output'
                require(re.search(rf'^{direction} wire \[{width-1}\s*:\s*0\] m_axib_{port};$', text, re.M),
                        'missing or narrowed bypass port: ' + port)
            require(".C_PCIEBAR2AXIBAR_2(64'H0000000100000000)" in text, 'RTL translation truncated')
            require('.XDMA_AXIST_BYPASS("TRUE")' in text, 'RTL bypass disabled')
        if name.endswith('/ntrpu_xdma_pcie4_ip.v'):
            checked_hard += 1
            for i, (aperture, control) in enumerate(expected_bars):
                for suffix, expected_value in [('APERTURE_SIZE', aperture), ('CONTROL', control)]:
                    found = re.findall(rf'\.PF0_BAR{i}_{suffix}\(\d+\x27H([0-9A-Fa-f]+)\)', text)
                    require(len(found) == 1 and int(found[0], 16) == expected_value, 'RTL BAR map differs')
    require(checked_top == checked_hard == 4, 'need generated sim/synth wrappers in both projects')
    require(evidence['reports']['ip_status.rpt'].count('| Up-to-date | No changes required') == 5,
            'unreviewed IP status')
    return dict(status='PASS_GENERATED_BYPASS_CONFIGURATION_ONLY', configuration_qualified=True,
                functional_qualified=False, seconds=inv['duration_seconds'],
                semantic_configuration_changes=wanted_delta,
                bars=[dict(index=0, bits=32, bytes=65536, use='AXI-Lite'),
                      dict(index=1, bits=32, bytes=65536, use='DMA and MSI-X'),
                      dict(index=2, upper_index=3, bits=64, bytes=1048576,
                           prefetchable=False, use='AXI-MM bypass', axi_base=1 << 32)],
                bypass_widths=widths, generated_wrappers_checked=checked_top+checked_hard,
                limitations=['No PCIe enumeration or MemWr/MemRd simulation yet',
                             'No peer receiver, DDR, coherent aggregation, P2P or board qualification'])
