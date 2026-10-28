"""Deterministic interleavings for queue state recovery and atomic replacement."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import ntrpu_queue_state_io_r1 as io

class QueueStateTests(unittest.TestCase):
    def test_truncated_legacy_write_is_retried(self):
        with tempfile.TemporaryDirectory(prefix='ntrpu-queue-state-') as d:
            p=Path(d)/'state.json';p.write_text('')
            complete={'active_runs':['real-run'],'all_owned_runs_terminal_and_backed_up':False}
            with patch.object(io.time,'sleep',side_effect=lambda _:p.write_text(json.dumps(complete))) as wait:
                self.assertEqual(io.read_json(p,attempts=2),complete)
                self.assertEqual(wait.call_count,1)

    def test_persistent_malformed_state_fails(self):
        with tempfile.TemporaryDirectory(prefix='ntrpu-queue-state-') as d:
            p=Path(d)/'state.json';p.write_text('{"active_runs":')
            with patch.object(io.time,'sleep') as wait:
                with self.assertRaises(json.JSONDecodeError):io.read_json(p,attempts=3)
                self.assertEqual(wait.call_count,2)
            with self.assertRaises(FileNotFoundError):io.read_json(Path(d)/'absent.json')

    def test_transient_read_denial_is_retried(self):
        with patch.object(Path,'read_text',side_effect=[PermissionError('sharing'),'{"run":1}']),patch.object(io.time,'sleep'):
            self.assertEqual(io.read_json('not-opened',attempts=2),{'run':1})

    def test_replacement_keeps_old_state_until_success(self):
        with tempfile.TemporaryDirectory(prefix='ntrpu-queue-state-') as d:
            p=Path(d)/'state.json';p.write_text('{"run":"old"}')
            original=os.replace;calls=[]
            def replace(source,target):
                self.assertEqual(json.loads(p.read_text()),{'run':'old'})
                self.assertEqual(json.loads(Path(source).read_text()),{'run':'new'})
                calls.append(source)
                if len(calls)==1:raise PermissionError('reader sharing')
                original(source,target)
            with patch.object(io.os,'replace',side_effect=replace),patch.object(io.time,'sleep'):
                io.write_json_atomic(p,{'run':'new'},attempts=2)
            self.assertEqual(json.loads(p.read_text()),{'run':'new'})
            self.assertEqual(list(Path(d).iterdir()),[p])

if __name__=='__main__':unittest.main()
