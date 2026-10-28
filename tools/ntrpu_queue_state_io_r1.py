"""Bounded reads of legacy state writers and atomic writes for new owners."""
import json
import os
from pathlib import Path
import time
import uuid

def read_json(path,attempts=40,delay=0.05):
    """Retry a transient partial write; persistent malformed state still fails."""
    assert attempts>0 and delay>=0
    for attempt in range(attempts):
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except (json.JSONDecodeError,PermissionError):
            if attempt+1==attempts:raise
            time.sleep(delay)

def write_json_atomic(path,value,attempts=40,delay=0.05):
    """Readers see the previous complete JSON or its complete replacement."""
    assert attempts>0 and delay>=0
    path=Path(path)
    pending=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.pending')
    with pending.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    for attempt in range(attempts):
        try:os.replace(pending,path);return
        except PermissionError:
            if attempt+1==attempts:raise
            time.sleep(delay)
