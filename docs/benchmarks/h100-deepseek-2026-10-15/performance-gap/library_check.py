import ctypes
from pathlib import Path
import subprocess
base=Path('/home/ubuntu/brandon-mini/.venv/lib/python3.12/site-packages/nvidia/cu13')
lib=ctypes.CDLL(str(base/'lib/libcupti.so.13')); version=ctypes.c_uint32(); result=lib.cuptiGetVersion(ctypes.byref(version)); print('cupti',result,version.value)
for p in ['/home/ubuntu/ntransformer-h100/ds41-gap-20261015/bench-nt','/home/ubuntu/ntransformer-h100/ds41-gap-20261015/smoke']:
    print(subprocess.run(['readelf','-d',p],capture_output=True,text=True).stdout)
