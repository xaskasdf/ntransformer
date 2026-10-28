"""VCD edge samples preserving four-state packet data at explicitly named ports."""
import re
from ntrpu_pcie_straddle import LogicWord, require


def logic_vcd_rows(text, data_paths):
    header, body = text.split('$enddefinitions $end', 1)
    require(re.search(r'\$timescale\s+1ps\s+\$end', header), 'VCD timescale')
    scopes, names, widths = [], {}, {}
    for line in header.splitlines():
        f=line.split()
        if not f:continue
        if f[0]=='$scope':scopes.append(f[2])
        elif f[0]=='$upscope':scopes.pop()
        elif f[0]=='$var':
            name='/'+'/'.join(scopes+[f[4]])
            require(name not in names, 'duplicate VCD name')
            names[name]=f[3]
            widths[f[3]]=int(f[2])
    require(set(data_paths)<=names.keys(), 'missing four-state data paths')
    aliases={code:[n for n in names if names[n]==code] for code in widths}
    current={n:None for n in names}
    changes,now={},0
    for line in body.splitlines():
        line=line.strip()
        if not line or line.startswith('$'):continue
        if line.startswith('#'):
            after={**current,**changes}
            yield now,current,after
            current,changes=after,{}
            new=int(line[1:]);require(new>=now,'VCD reversal');now=new
        else:
            raw,code=line[1:].split() if line.startswith('b') else (line[0],line[1:])
            require(code in widths, 'unknown VCD code')
            logic=LogicWord.parse(raw,widths[code])
            integer=logic.value if logic.known==(1 << logic.width)-1 else None
            for name in aliases[code]:changes[name]=logic if name in data_paths else integer
    yield now,current,{**current,**changes}
