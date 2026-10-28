"""Prove the changed combinational predicate for all 64-bit/13-bit binary inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from prepare_hbm_range import sha, variant, SOURCE_SHA


def prove(source, out, deps):
    if out.exists(): raise ValueError('New proof directory required')
    if deps: sys.path.insert(0, str(deps))
    import z3
    original = (source/'reference_hbm_source.sv').read_text().replace('module reference_hbm_source (','module ntrpu_axi_bridge_hbm_source (')
    if hashlib.sha256(original.encode()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Reference is not the frozen original')
    if variant(original) != (source/'ntrpu_axi_bridge_hbm_source.sv').read_text():
        raise ValueError('The checked variant has additional changes')
    out.mkdir(parents=True)
    a = z3.BitVec('address', 64); n = z3.BitVec('length', 13)
    # SV comparison against the unsized decimal 4096 extends this sum to 32 bits.
    page_end = z3.ZeroExt(20, z3.Extract(11, 0, a)) + z3.ZeroExt(19, n)
    common = [n == 0, z3.UGT(n, 4096), z3.Extract(5,0,a) != 0, z3.Extract(5,0,n) != 0,
              z3.UGT(page_end, 4096)]
    old = z3.Or(*common, z3.UGT(z3.ZeroExt(1,a)+z3.ZeroExt(52,n), 1<<34))
    new = z3.Or(*common, z3.And(z3.UGE(a,1<<34), z3.UGT(n,0)))
    queries = [('equivalence',old != new, 'unsat'),
               ('missing_page_check', old != z3.Or(*common[:-1],z3.UGE(a,1<<34)), 'sat'),
               ('wrong_stack_bit', old != z3.Or(*common,z3.UGE(a,1<<33)), 'sat'),
               ('reject_exact_end', old != z3.Or(*common,z3.UGE(z3.ZeroExt(1,a)+z3.ZeroExt(52,n),1<<34)), 'sat')]
    # Three-valued expression model: 0=false, 1=true, 2=unknown (X or Z).
    # Arithmetic and relational operators yield X if any operand bit is X/Z.
    ak=z3.BitVec('address_known_mask',64);nk=z3.BitVec('length_known_mask',13)
    def tri(known, value):return z3.If(known,z3.If(value,1,0),2)
    def tor(*xs):return z3.If(z3.Or(*[x==1 for x in xs]),1,z3.If(z3.Or(*[x==2 for x in xs]),2,0))
    def tand(x,y):return z3.If(z3.Or(x==0,y==0),0,z3.If(z3.Or(x==2,y==2),2,1))
    def reduced(value,mask,bits):
        return z3.If((value&mask&bits)!=0,1,z3.If((mask&bits)==bits,0,2))
    a_known=ak==(1<<64)-1;n_known=nk==8191
    zero=z3.If((n&nk)!=0,0,z3.If(n_known,1,2))
    common_x=[zero,tri(n_known,z3.UGT(n,4096)),reduced(a,ak,63),reduced(n,nk,63),
              tri(z3.And((ak&4095)==4095,n_known),z3.UGT(page_end,4096))]
    old_x=tor(*common_x,tri(z3.And(a_known,n_known),z3.UGT(z3.ZeroExt(1,a)+z3.ZeroExt(52,n),1<<34)))
    new_x=tor(*common_x,tand(tri(a_known,z3.UGE(a,1<<34)),tri(n_known,z3.UGT(n,0))))
    queries += [('four_state_expression_model',old_x!=new_x,'unsat'),
                ('four_state_plain_upper_bits',old_x!=tor(*common_x,reduced(z3.Extract(63,34,a),z3.Extract(63,34,ak),(1<<30)-1)),'sat'),
                ('four_state_missing_positive_length',old_x!=tor(*common_x,tri(a_known,z3.UGE(a,1<<34))),'sat')]
    records=[]; started=time.monotonic()
    for name,expr,expected in queries:
        solver=z3.Solver();solver.set(timeout=30000);solver.add(expr)
        (out/(name+'.smt2')).write_text(solver.to_smt2())
        result=str(solver.check());row=dict(name=name,result=result,expected=expected)
        if result=='sat':
            model=solver.model();row['counterexample']=dict(address=model.eval(a,model_completion=True).as_long(),length=model.eval(n,model_completion=True).as_long())
            if name.startswith('four_state_'):
                row['counterexample'].update(address_known_mask=model.eval(ak,model_completion=True).as_long(),length_known_mask=model.eval(nk,model_completion=True).as_long())
        if result!=expected:raise ValueError('Unproved predicate or ineffective control: '+str(row))
        records.append(row)
    record=dict(status='HBM_RANGE_PREDICATE_PROVED',solver=z3.get_version_string(),queries=records,
                seconds=time.monotonic()-started,source_sha256=sha(source/'ntrpu_axi_bridge_hbm_source.sv'),
                input_bits=77,assumptions=[],scope='Two-state predicate and manually modeled four-state expression; not an HDL parser or sequential equivalence proof')
    (out/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--deps',type=Path)
    a=ap.parse_args();print(json.dumps(prove(a.source,a.output,a.deps),indent=2))
