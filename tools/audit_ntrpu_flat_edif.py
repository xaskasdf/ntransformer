"""Conservative combinational-cycle audit for the inventory's flat primitive EDIF.

Every instance and port reference must parse. An edge connects any input to any
output of a combinational primitive, including CARRY8. Thus a DAG proves absence
of combinational cycles; a residual graph is inconclusive until refined. FDRE
is the only permitted sequential primitive. This does not check timing or logic
equivalence, and rejects hierarchies/unknown primitive interfaces.
"""
import collections,re

ID=r'[^\s()"]+'
NAME=rf'(?:{ID}|\(rename\s+{ID}\s+"[^"]*"\))'
INSTANCE=re.compile(rf'^\s*\(instance\s+({NAME})\s+\(viewref\s+netlist\s+\(cellref\s+({ID})\s+\(libraryref\s+hdi_primitives\)\)\)',re.M)
NET=re.compile(rf'^\s*\(net\s+({NAME})\s+\(joined\b',re.M)
REF=re.compile(rf'\(portref\s+(?:\(member\s+({ID})\s+(\d+)\)|({ID}))(?:\s+\(instanceref\s+({ID})\))?\s*\)')
PORTS={f'LUT{n}':{**{f'I{i}':('INPUT',1) for i in range(n)},'O':('OUTPUT',1)} for n in range(1,7)}
PORTS.update({n:{'I0':('INPUT',1),'I1':('INPUT',1),'S':('INPUT',1),'O':('OUTPUT',1)} for n in ('MUXF7','MUXF8')})
PORTS.update(GND={'G':('OUTPUT',1)},VCC={'P':('OUTPUT',1)},INV={'I':('INPUT',1),'O':('OUTPUT',1)},
    FDRE={**{n:('INPUT',1) for n in ('C','CE','D','R')},'Q':('OUTPUT',1)},
    CARRY8={'CI':('INPUT',1),'CI_TOP':('INPUT',1),'DI':('INPUT',8),'S':('INPUT',8),'CO':('OUTPUT',8),'O':('OUTPUT',8)})


def canonical(name):
    return name.split()[1] if name.startswith('(rename ') else name


def remaining_after_topological_sort(graph):
    degree={n:0 for n in graph}
    for targets in graph.values():
        for target in targets:degree[target]+=1
    todo=collections.deque(n for n,d in degree.items() if not d)
    removed=set()
    while todo:
        n=todo.popleft();removed.add(n)
        for target in graph[n]:
            degree[target]-=1
            if degree[target]==0:todo.append(target)
    return sorted(set(graph)-removed)


def audit(text):
    instances={}
    for match in INSTANCE.finditer(text):
        name,kind=canonical(match[1]),match[2]
        if name in instances or kind not in PORTS:raise ValueError('duplicate/unsupported instance: '+name)
        instances[name]=kind
    if len(instances)!=len(re.findall(r'^\s*\(instance\b',text,re.M)) or not instances:
        raise ValueError('not every flat primitive instance parsed')
    comb={n for n,k in instances.items() if k not in ('FDRE','GND','VCC')}
    graph={n:set() for n in comb}
    nets=list(NET.finditer(text));names=set();refs=0;instance_pins=set()
    if len(nets)!=len(re.findall(r'^\s*\(net\b',text,re.M)) or not nets:
        raise ValueError('not every net parsed')
    for index,net in enumerate(nets):
        name=canonical(net[1])
        if name in names:raise ValueError('duplicate net: '+name)
        names.add(name)
        block=text[net.end():nets[index+1].start() if index+1<len(nets) else len(text)]
        parsed=list(REF.finditer(block));refs+=len(parsed)
        if len(parsed)!=len(re.findall(r'\(portref\b',block)):
            raise ValueError('unparsed port reference in '+name)
        drivers=[];sinks=[]
        for ref in parsed:
            port=ref[1] or ref[3];bit=int(ref[2]) if ref[2] is not None else None;instance=ref[4]
            if instance is None:continue # Top ports cannot form an internal cycle.
            if instance not in instances or port not in PORTS[instances[instance]]:
                raise ValueError('unsupported instance port '+str((instance,port)))
            direction,width=PORTS[instances[instance]][port]
            if (bit is None and width!=1) or (bit is not None and not 0<=bit<width):
                raise ValueError('invalid port width/index')
            key=(instance,port,bit)
            if key in instance_pins:raise ValueError('instance pin assigned multiple nets')
            instance_pins.add(key)
            (drivers if direction=='OUTPUT' else sinks).append(instance)
        if len(drivers)>1:raise ValueError('multiple primitive output drivers on '+name)
        if drivers and drivers[0] in comb:
            graph[drivers[0]].update(n for n in sinks if n in comb)
    if refs!=len(re.findall(r'\(portref\b',text)):
        raise ValueError('port reference outside parsed nets')
    remaining=remaining_after_topological_sort(graph)
    # A deliberate combinational feedback edge must invalidate this same graph.
    first=next(iter(graph));graph[first].add(first)
    if not remaining_after_topological_sort(graph):raise AssertionError('cycle negative control passed')
    return dict(instances=len(instances),primitive_counts=dict(collections.Counter(instances.values())),
        nets=len(nets),port_references=refs,combinational_cells=len(comb),
        combinational_graph_acyclic=not remaining,residual_cell_count=len(remaining),residual_cells=remaining[:100],
        injected_cycle_rejected=True,scope='Flat primitive graph over-approximates all combinational input/output dependencies, stopping at FDRE. Acyclicity is structural only; no timing, functionality, CDC or board qualification.')
