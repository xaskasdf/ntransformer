from pathlib import Path
import importlib.util
r=Path('/home/ubuntu/ntransformer-h100/ds41-gap-20261015')
s=importlib.util.spec_from_file_location('gap',r/'run.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
b=g.b
args=[b.PREFIX/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-g','-rdynamic','-I'+str(g.CUDA/'include'),'-I'+str(r/'cupti-include'),r/'smoke.cpp',r/'trace.cpp','-L'+str(g.CUDA/'lib'),'-lcudart','-L'+str(g.CU13/'lib'),'-l:libcupti.so.13','-o',r/'smoke']
g.cmd('compile-smoke',args)
with b.admitted():
    b.ENV['GAP_TRACE']=str(r)
    g.cmd('smoke',[r/'smoke'])
