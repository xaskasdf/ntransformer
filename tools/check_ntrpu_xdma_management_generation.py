"""Compare generated XDMA wrappers without interpreting protected implementation."""
import re


def need(condition, message):
    if not condition:
        raise ValueError(message)


def named_arguments(text):
    result = {}
    pattern = re.compile(r'\.(\w+)\s*\(([^()]*)\)', re.S)
    for match in pattern.finditer(text):
        name, value = match.groups()
        need(name not in result, 'duplicate named argument: ' + name)
        result[name] = re.sub(r'\s+', '', value)
    need(not re.sub(r'[\s,]+', '', pattern.sub('', text)), 'unparsed instance arguments')
    return result


def wrapper(text):
    """Accept only the generated declaration-plus-single-instance wrapper grammar."""
    timescale = re.findall(r'^\s*`timescale\s+(.+)$', text, re.M)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    attributes = re.findall(r'\(\*.*?\*\)', text, re.S)
    text = re.sub(r'\(\*.*?\*\)', '', text, flags=re.S)
    text = re.sub(r'^\s*`timescale[^\n]*', '', text, flags=re.M)
    match = re.fullmatch(r'\s*module\s+ntrpu_xdma\s*\((.*?)\)\s*;(.*?)'
                         r'\b(\w+)\s*#\s*\((.*?)\)\s*inst\s*\((.*?)\)\s*;\s*endmodule\s*', text, re.S)
    need(match is not None, 'unparsed generated wrapper structure')
    port_text, declarations, core, parameters, connections = match.groups()
    ports = [x.strip() for x in port_text.split(',')]
    need(all(re.fullmatch(r'\w+', x) for x in ports) and len(ports) == len(set(ports)), 'invalid public port list')
    declaration = re.compile(r'\b(input|output|inout)\s+(wire|reg)\s*(\[[^\]]+\])?\s*(\w+)\s*;')
    declared = {}
    for row in declaration.finditer(declarations):
        direction, kind, width, name = row.groups()
        need(name not in declared, 'duplicate public declaration')
        declared[name] = (direction, kind, re.sub(r'\s+', '', width or ''))
    need(not declaration.sub('', declarations).strip(), 'unparsed declaration or wrapper logic')
    need(set(ports) == set(declared), 'declared ports differ from port list')
    return dict(ports=ports, declarations=declared, core=core,
                parameters=named_arguments(parameters), connections=named_arguments(connections),
                timescale=timescale, attributes=attributes)


def compare_wrappers(original, generated):
    a, z = wrapper(original), wrapper(generated)
    need(a['core'] == z['core'] and a['timescale'] == z['timescale'], 'core type or timescale changed')
    expected_parameters = {'XDMA_AXILITE_SLAVE': ('"FALSE"', '"TRUE"'),
                           'XDMA_STS_PORTS': ('"FALSE"', '"TRUE"')}
    delta = {k:(a['parameters'].get(k),z['parameters'].get(k))
             for k in a['parameters'].keys() | z['parameters'].keys()
             if a['parameters'].get(k) != z['parameters'].get(k)}
    need(delta == expected_parameters, 'unexpected functional core parameter change')
    inputs = {'s_axil_awaddr':'[31:0]', 's_axil_awprot':'[2:0]', 's_axil_awvalid':'',
              's_axil_wdata':'[31:0]', 's_axil_wstrb':'[3:0]', 's_axil_wvalid':'', 's_axil_bready':'',
              's_axil_araddr':'[31:0]', 's_axil_arprot':'[2:0]', 's_axil_arvalid':'', 's_axil_rready':''}
    outputs = {'s_axil_awready':'', 's_axil_wready':'', 's_axil_bvalid':'', 's_axil_bresp':'[1:0]',
               's_axil_arready':'', 's_axil_rdata':'[31:0]', 's_axil_rresp':'[1:0]', 's_axil_rvalid':'',
               'h2c_sts_0':'[7:0]', 'c2h_sts_0':'[7:0]'}
    added = {**{n:('input','wire',w) for n,w in inputs.items()},
             **{n:('output','wire',w) for n,w in outputs.items()}}
    need(not (set(a['declarations']) & set(added)), 'baseline already exposes management ports')
    need(z['declarations'] == a['declarations'] | added, 'unexpected public port or width change')
    need([n for n in z['ports'] if n not in added] == a['ports'], 'existing port order changed')
    need(set(a['connections']) == set(z['connections']), 'protected-core interface changed')
    for name, old in a['connections'].items():
        expected = name if name in added else old
        need(z['connections'][name] == expected, 'unexpected core binding: ' + name)
        if name in inputs:
            need(re.fullmatch(r"\d+'[bB]0", old) is not None, 'baseline input was not tied low: ' + name)
        elif name in outputs:
            need(old == '', 'baseline output was not open: ' + name)
    need(set(added) <= set(z['connections']), 'new public port is not bound to core')
    # Attributes are preserved in the evidence, but no physical/XDC approval is inferred.
    return dict(status='PASS_GENERATED_PUBLIC_INTERFACE_ONLY', added_ports=added,
                parameter_changes=delta, original_port_count=len(a['ports']),
                generated_port_count=len(z['ports']), core=a['core'],
                attributes_require_physical_review=True, native_behavior_tested=False)


def parameters(xci, group):
    return {k:[v['value'] for v in values]
            for k,values in xci['ip_inst']['parameters'][group].items()}


def compare_parameters(original, generated):
    need(original['ip_inst']['component_reference'] == generated['ip_inst']['component_reference'] ==
         'xilinx.com:ip:xdma:4.2', 'XDMA version changed')
    result = {}
    for group, expected in [
        ('component_parameters', {'xdma_axilite_slave':(['false'],['true']), 'xdma_sts_ports':(['false'],['true'])}),
        ('model_parameters', {'XDMA_AXILITE_SLAVE':(['FALSE'],['TRUE']), 'XDMA_STS_PORTS':(['FALSE'],['TRUE'])})]:
        a,z = parameters(original,group),parameters(generated,group)
        delta = {k:(a.get(k),z.get(k)) for k in a.keys() | z.keys() if a.get(k) != z.get(k)}
        need(delta == expected, 'unexpected ' + group + ' changes: ' + repr(delta))
        result[group] = delta
    return result
