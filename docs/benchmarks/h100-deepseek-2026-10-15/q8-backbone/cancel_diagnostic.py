from pathlib import Path
import os, signal
pid=1657375
if Path('/proc',str(pid),'cwd').resolve() == Path('/home/ubuntu/ntransformer-h100/ds41-backbone-20261015'):
    assert Path('/proc',str(pid),'cmdline').read_bytes().endswith(b'diagnose.py\x00')
    os.kill(pid, signal.SIGTERM)
