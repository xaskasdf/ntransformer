from common import *
run('compile-probe',[prefix/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
    '-I'+str(source/'src'),'-I'+str(source/'include'),root/'probe.cpp',root/'build/libntransformer_lib.a',
    '-L'+str(prefix/'lib'),'-L'+str(prefix/'targets/x86_64-linux/lib'),'-lcudart','-lcudadevrt','-lcudart_static',
    '-ldl','-lrt','-lpthread','-o',root/'probe'])
