"""Compare both replay traces with every frozen binary expected output."""
import re


def review(logs, vectors):
    expected=[]
    for line in vectors.splitlines():
        fields=line.split()
        if len(fields)!=8:raise ValueError('vector width')
        valid,effect,metadata=map(lambda x:int(x,16),fields[5:])
        expected.append((valid<<87)|(effect<<30)|metadata)
    if len(expected)!=12805:raise ValueError('vector count')
    values={}
    for mode in ['rtl','mapped']:
        log=logs[mode]
        if re.search(r'^\s*(?:Fatal:|ERROR:|Failure:|FAIL )',log,re.M):raise ValueError('simulation failure')
        if log.splitlines().count('EFFECT_BINARY_PASS rows=12805')!=1:raise ValueError('completion')
        rows=[s for s in log.splitlines() if s.startswith('EFFECT_BINARY_ROW ')]
        if len(rows)!=len(expected):raise ValueError('row count')
        result=[]
        for i,row in enumerate(rows):
            m=re.fullmatch(r'EFFECT_BINARY_ROW index=(\d+) value=([0-9a-f]{22})',row)
            if not m or int(m[1])!=i:raise ValueError('row sequence/format')
            value=int(m[2],16)
            if value!=expected[i]:raise ValueError('frozen reference mismatch')
            result.append(value)
        values[mode]=result
    if values['rtl']!=values['mapped']:raise ValueError('RTL/netlist mismatch')
    return dict(rows=12805,observations=25610,all_rtl_netlist_reference_rows_identical=True)
