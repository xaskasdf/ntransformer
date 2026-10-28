"""Supplement numeric/lane review with independent two-entry FIFO coverage."""
import re


def review(text):
    lines=[l for l in text.splitlines() if l.startswith('NTRPU_BLOCK_FIFO ')]
    if len(lines)!=1:
        return dict(status='FAIL',errors=['one FIFO summary required'])
    fields={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',lines[0])}
    required={'pushes','pops','dropped','peak','full_stalls','simultaneous','pending'}
    errors=[]
    if set(fields)!=required:
        errors.append('exact FIFO summary schema')
    else:
        if not (fields['pushes']==fields['pops']+fields['dropped'] and fields['pending']==0):
            errors.append('FIFO conservation and complete drain')
        if not (fields['peak']==2 and fields['full_stalls']>0 and fields['simultaneous']>0 and fields['pops']>0):
            errors.append('FIFO depth/backpressure/simultaneous coverage')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,counts=fields)


def mutations(text):
    def change(field,value):
        return re.sub(r'(^NTRPU_BLOCK_FIFO [^\n]*\b'+field+r'=)\d+',lambda m:m[1]+str(value),text,count=1,flags=re.M)
    return dict(fifo_lost=change('pushes',0),fifo_pending=change('pending',1),
                fifo_depth=change('peak',1),fifo_no_backpressure=change('full_stalls',0),
                fifo_no_simultaneous=change('simultaneous',0))
