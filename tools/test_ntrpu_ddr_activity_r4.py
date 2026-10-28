"""Corruption controls for the R2 VCD/SAIF binding; synthetic evidence only."""
import csv
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

import review_ntrpu_ddr_activity_r4 as r
from test_ntrpu_ddr_rank_commands_r1 import waveform, command
from test_ntrpu_ddr_activity_detail_r1 import synthetic_windows


class CaptureR2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'windows.tsv').write_text(synthetic_windows())
        self.windows = r.detail.review_windows(self.root/'windows.tsv')['windows']
        text = waveform([command(bits) for bits in ('001', '101', '100')]*2)
        def offset(match):
            value = int(match[1])
            if value >= 10000:
                return '#90980000'
            return '#'+str(49998125+value+(20476250 if value >= 5000 else 0))
        self.text = re.sub(r'(?m)^#(\d+)$', offset, text)
        self.write_trace(self.text)
        ranks = r.commands.read_vcd(self.root/'rank_commands.vcd')[0]
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['rank', 'pin', 'object'])
        for rank, row in ranks.items():
            for pin in r.commands.PINS:
                writer.writerow([rank, pin, '/ntrpu_ddr_example_activity_tb/'+row['scope']+'/'+pin])
        self.inventory = stream.getvalue()
        (self.root/'rank_pins.tsv').write_text(self.inventory)

    def write_trace(self, text):
        (self.root/'rank_commands.vcd').write_text(text)

    def review(self):
        return r.rank_evidence(self.root, self.windows)

    def test_bound_four_ranks_and_windows(self):
        out = self.review()
        self.assertTrue(out['inventory_matches_vcd'])
        self.assertFalse(out['window_observation_gaps'])
        for row in out['ranks'].values():
            for window in row['windows']:
                self.assertEqual(window['counts'], dict(REF=1, RD=1, WR=1))
        self.assertFalse(out['refresh_schedule_qualified'])
        self.assertFalse(out['ready_for_power_estimate'])

    def test_truncated_trace_margin(self):
        self.write_trace(self.text.replace('#90980000', '#90960000'))
        with self.assertRaisesRegex(ValueError, 'extent'):
            self.review()

    def test_missing_rank_pin(self):
        self.write_trace(self.text.replace(' TEN $end', ' MISSING $end', 1))
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.review()

    def test_wrong_inventory_path(self):
        (self.root/'rank_pins.tsv').write_text(self.inventory.replace('/CK\n', '/CK_BAD\n', 1))
        with self.assertRaisesRegex(ValueError, 'differ'):
            self.review()

    def test_duplicate_inventory(self):
        (self.root/'rank_pins.tsv').write_text(self.inventory+self.inventory.splitlines()[1]+'\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.review()

    def test_missing_refresh_remains_open(self):
        # Change each REF's WE from 1 to 0: MRS must never count as REF.
        self.write_trace(re.sub(r'b1 (v\d+_WE_n_A14)', r'b0 \1', self.text))
        out = self.review()
        self.assertEqual(sum('no refresh' in x for x in out['window_observation_gaps']), 8)
        self.assertFalse(out['ready_for_power_estimate'])

    def test_noninteger_ps_rejected(self):
        self.windows[0]['start']['simulator_seconds'] = '0.0000500000001'
        with self.assertRaisesRegex(ValueError, 'Noninteger'):
            self.review()

    def test_wrong_source_and_revision_rejected(self):
        for change in ({'capture_revision': 1}, {'source_sha256': 'b'*64}, {'exit_code': 1},
                       {'frozen_source_unchanged': False}, {'finished_utc': None}):
            inv = dict(capture_revision=2, source_sha256='a'*64, exit_code=0,
                       frozen_source_unchanged=True, finished_utc='test')
            inv.update(change)
            if change == {'finished_utc': None}:
                del inv['finished_utc']
            (self.root/'invocation.json').write_text(json.dumps(inv))
            out = r.review(self.root, 'a'*64)
            self.assertEqual(out['status'], 'DIAGNOSTIC_DDR_ACTIVITY_R2')
            self.assertTrue(out['errors'])
            self.assertFalse(out['ready_for_power_estimate'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
