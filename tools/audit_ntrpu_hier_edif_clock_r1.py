"""Conservative hierarchical EDIF audit with a strictly checked BUFGCE clock tree.

Primitive boundaries and directions use the reviewed flat auditor's whitelist.
All combinational input/output dependencies are over-approximated. Registers
cut the graph. No timing, functional, CDC or physical qualification is inferred.
"""
import collections
import re
from audit_ntrpu_flat_edif_r1 import PORTS as BASE_PORTS, remaining_after_topological_sort
# Keep the previously reviewed whitelist immutable. A BUFGCE is conservatively
# combinational in the DAG; its enable does not introduce a sequential cut.
PORTS = dict(BASE_PORTS)
PORTS['BUFGCE'] = {'I': ('INPUT', 1), 'CE': ('INPUT', 1), 'O': ('OUTPUT', 1)}


def parse(text):
    stack=[];roots=[]
    for token in re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()"]+',text):
        if token=='(':
            node=[]
            (stack[-1] if stack else roots).append(node);stack.append(node)
        elif token==')':
            if not stack:raise ValueError('Unbalanced EDIF')
            stack.pop()
        elif stack:stack[-1].append(token.lower() if not stack[-1] else token)
        else:raise ValueError('Atom outside EDIF')
    if stack or len(roots)!=1 or roots[0][0]!='edif':raise ValueError('Incomplete EDIF')
    return roots[0]


def children(node,kind):
    return [n for n in node if isinstance(n,list) and n and n[0]==kind]


def one(node,kind):
    found=children(node,kind)
    if len(found)!=1:raise ValueError('Expected unique '+kind)
    return found[0]


def name(value):
    if isinstance(value,str):return value
    if len(value)==3 and value[0]=='rename' and isinstance(value[1],str):return value[1]
    raise ValueError('Unsupported EDIF name')


def cell_reference(node):
    return name(one(node,'libraryref')[1]),name(node[1])


def audit(text, require_buffer=True):
    tree=parse(text);cells={};interfaces={}
    for lib in children(tree,'library')+children(tree,'external'):
        library=name(lib[1])
        for cell in children(lib,'cell'):
            key=(library,name(cell[1]))
            if key in cells:raise ValueError('Duplicate cell definition')
            view=one(cell,'view');interface=one(view,'interface');ports={}
            for port in children(interface,'port'):
                p=port[1];width=1
                if isinstance(p,list) and p[0]=='array':
                    if len(p)!=3:raise ValueError('Unsupported port array')
                    width=int(p[2]);p=p[1]
                p=name(p);direction=one(port,'direction')[1]
                if p in ports or width<1 or direction not in ('INPUT','OUTPUT'):raise ValueError('Invalid interface')
                ports[p]=(direction,width)
            cells[key]=view;interfaces[key]=ports
    design=one(tree,'design');root=cell_reference(one(design,'cellref'))
    if root not in cells:raise ValueError('Missing root definition')
    parents={};leafs={};leaf_properties={};pins={};top_inputs=set();scopes=[];net_count=reference_count=0
    empty_net_declarations=0;duplicate_empty_annotations=[]

    def find(x):
        parents.setdefault(x,x)
        while parents[x]!=x:
            parents[x]=parents[parents[x]];x=parents[x]
        return x

    def join(a,b):
        a=find(a);b=find(b)
        if a!=b:parents[b]=a

    def visit(key,path,ancestry):
        nonlocal net_count,reference_count,empty_net_declarations
        if key in ancestry:raise ValueError('Recursive hierarchy')
        if key not in cells:raise ValueError('Undefined hierarchy cell')
        view=cells[key];contents=one(view,'contents');instances={};scopes.append('/'.join(path) or 'TOP')
        for inst in children(contents,'instance'):
            label=name(inst[1]);ref=one(inst,'viewref');target=cell_reference(one(ref,'cellref'))
            if label in instances:raise ValueError('Duplicate scoped instance')
            instances[label]=target
            child=path+(label,)
            if target[0]=='hdi_primitives':
                if target[1] not in PORTS:raise ValueError('Unsupported primitive '+target[1])
                if target in interfaces and interfaces[target]!=PORTS[target[1]]:raise ValueError('Primitive interface mismatch')
                leafs[child]=target[1]
                if target[1]=='BUFGCE':
                    props={}
                    for prop in children(inst,'property'):
                        key=name(prop[1])
                        if key in props:raise ValueError('Duplicate buffer property')
                        if len(prop)!=3:raise ValueError('Unsupported buffer property encoding')
                        props[key]=prop[2]
                    leaf_properties[child]=props
            else:
                visit(target,child,ancestry+(key,))
        seen_nets={};seen_pins=set()
        for net in children(contents,'net'):
            label=name(net[1])
            joined=one(net,'joined');refs=children(joined,'portref')
            if len(refs)!=len(joined)-1:raise ValueError('Unparsed joined entry')
            display=net[1][2] if isinstance(net[1],list) else label
            previous=seen_nets.setdefault(label,[])
            if refs and any(item['references'] for item in previous):
                raise ValueError('Duplicate connected scoped net')
            previous.append(dict(display_name=display,references=len(refs)))
            if not refs:empty_net_declarations+=1
            net_count+=1;netkey=('net',path,label);find(netkey)
            for ref in refs:
                reference_count+=1;port=ref[1];bit=None
                if isinstance(port,list) and port[0]=='member':
                    if len(port)!=3:raise ValueError('Invalid port member')
                    bit=int(port[2]);port=port[1]
                port=name(port);instances_ref=children(ref,'instanceref')
                if len(instances_ref)>1 or len(ref)!=2+len(instances_ref):raise ValueError('Malformed port reference')
                if instances_ref:
                    instance=name(instances_ref[0][1])
                    if instance not in instances:raise ValueError('Unknown instance')
                    target=instances[instance];child=path+(instance,)
                    interface=PORTS[target[1]] if target[0]=='hdi_primitives' else interfaces[target]
                    primitive=target[0]=='hdi_primitives'
                else:
                    child=path;interface=interfaces[key];primitive=False
                if port not in interface:raise ValueError('Unsupported port '+port)
                direction,width=interface[port]
                if (bit is None and width!=1) or (bit is not None and not 0<=bit<width):raise ValueError('Invalid port width/index')
                if width==1:bit=0
                endpoint=('pin' if primitive else 'port',child,port,bit)
                if endpoint in seen_pins:raise ValueError('Pin assigned multiple scoped nets')
                seen_pins.add(endpoint);join(netkey,endpoint)
                if primitive:pins[endpoint]=(child,direction)
                elif not child and direction=='INPUT':top_inputs.add(endpoint)
        for label,declarations in seen_nets.items():
            if len(declarations)>1:
                duplicate_empty_annotations.append(dict(scope='/'.join(path) or 'TOP',
                    canonical_name=label,declarations=declarations))
        return
    visit(root,(),())
    if not leafs or not net_count:raise ValueError('Empty graph')
    # Every contents body must be in the reachable design; never silently skip one.
    defined=[key for key,view in cells.items() if children(view,'contents')]
    reachable=set()
    def mark(key):
        if key in reachable:return
        reachable.add(key)
        for inst in children(one(cells[key],'contents'),'instance'):
            target=cell_reference(one(one(inst,'viewref'),'cellref'))
            if target[0]!='hdi_primitives':mark(target)
    mark(root)
    if set(defined)!=reachable:raise ValueError('Unvisited contents definition')
    connections=collections.defaultdict(lambda:dict(drivers=[],sinks=[]))
    for endpoint,(instance,direction) in pins.items():
        connections[find(endpoint)]['drivers' if direction=='OUTPUT' else 'sinks'].append(instance)
    for endpoint in top_inputs:connections[find(endpoint)]['drivers'].append(('TOP_INPUT',endpoint))
    buffers=[p for p,k in leafs.items() if k=='BUFGCE']
    buffer_review=None
    if require_buffer and len(buffers)!=1:raise ValueError('Expected exactly one BUFGCE')
    if buffers:
        if len(buffers)!=1:raise ValueError('Multiple BUFGCE not supported in this single-clock audit')
        buffer=buffers[0]
        def pin_root(port):
            ep=('pin',buffer,port,0)
            if ep not in pins:raise ValueError('Missing buffer pin '+port)
            return find(ep)
        input_root,enable_root,output_root=[pin_root(p) for p in ('I','CE','O')]
        if len({input_root,enable_root,output_root})!=3:raise ValueError('Buffer pins alias each other')
        groups=collections.defaultdict(list)
        for ep in pins:groups[find(ep)].append(ep)
        input_drivers=connections[input_root]['drivers']
        clk=('port',(),'clk',0)
        if input_drivers!=[('TOP_INPUT',clk)]:raise ValueError('Buffer input must be driven only by top clk')
        if set(groups[input_root])!={('pin',buffer,'I',0)}:raise ValueError('Clock bypass or extra load before buffer')
        enable_drivers=connections[enable_root]['drivers']
        if len(enable_drivers)!=1 or leafs.get(enable_drivers[0])!='VCC':raise ValueError('Buffer enable must be driven by VCC')
        expected={('pin',p,'C',0) for p,k in leafs.items() if k=='FDRE'}
        if not expected:raise ValueError('Missing clocked registers')
        sinks={ep for ep in groups[output_root] if pins[ep][1]=='INPUT'}
        if sinks!=expected:raise ValueError('Clock output does not exclusively reach every register C pin')
        if connections[output_root]['drivers']!=[buffer]:raise ValueError('Clock buffer output driver mismatch')
        if any(ep[0]=='port' and not ep[1] and find(ep)==output_root for ep in parents):
            raise ValueError('Buffered clock unexpectedly reaches top port')
        props=leaf_properties[buffer]
        def value(key,default):
            if key not in props:return default
            val=props[key]
            if len(val)==2 and val[0]=='string':return val[1].strip('"').upper()
            if len(val)==2 and val[0]=='boolean' and val[1] in (['false'],['true']):
                return '0' if val[1]==['false'] else '1'
            raise ValueError('Unsupported effective buffer parameter '+key)
        effective={key:value(key,default) for key,default in
            [('CE_TYPE','SYNC'),('IS_CE_INVERTED',"1'B0"),('IS_I_INVERTED',"1'B0"),('STARTUP_SYNC','FALSE')]}
        if effective['CE_TYPE']!='SYNC':raise ValueError('Unsupported clock enable mode')
        if any(effective[k] not in ("1'B0",'0','FALSE') for k in ['IS_CE_INVERTED','IS_I_INVERTED']):
            raise ValueError('Clock or enable inversion is not permitted')
        if effective['STARTUP_SYNC'] not in ('FALSE','0'):raise ValueError('Unexpected startup synchronization')
        buffer_review=dict(instance='/'.join(buffer),properties=props,effective_control_parameters=effective,
            clock_loads=len(expected),input_top_port='clk',enable_driver='/'.join(enable_drivers[0]),
            output_only_drives_all_fdre_clocks=True,buffer_is_combinational_in_dag=True,
            absent_control_parameter_defaults_require_installed_library_crosscheck=True)
    comb={p for p,k in leafs.items() if k not in ('FDRE','GND','VCC')};graph={p:set() for p in comb}
    for net in connections.values():
        if len(net['drivers'])>1:raise ValueError('Multiple drivers after hierarchy expansion')
        if net['drivers'] and net['drivers'][0] in comb:
            graph[net['drivers'][0]].update(p for p in net['sinks'] if p in comb)
    residual=remaining_after_topological_sort(graph)
    first=next(iter(graph));graph[first].add(first)
    if not remaining_after_topological_sort(graph):raise AssertionError('Cycle control admitted')
    return dict(clock_buffer=buffer_review,instances=len(leafs),primitive_counts=dict(collections.Counter(leafs.values())),
        nets=net_count,aliased_net_groups=len({find(n) for n in parents if n[0]=='net'}),
        empty_net_declarations=empty_net_declarations,duplicate_empty_annotations=duplicate_empty_annotations,
        port_references=reference_count,hierarchy_scopes=scopes,
        combinational_cells=len(comb),combinational_graph_acyclic=not residual,
        residual_cell_count=len(residual),residual_cells=['/'.join(p) for p in residual[:100]],
        injected_cycle_rejected=True,
        scope='Hierarchical primitive graph with explicit parent/child port aliases; conservative combinational DAG only. No timing, functionality, CDC or physical qualification.')
