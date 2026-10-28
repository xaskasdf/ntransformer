"""Independent small waveforms for DDR pin observation, not DDR model tests."""
from pathlib import Path
import tempfile
import unittest
import review_ntrpu_ddr_rank_commands_r1 as r


def waveform(commands, ranks=4, same_edge=False):
    text = ['$timescale 1 ps $end', '$scope module tb $end']
    codes = {}
    for rank in range(ranks):
        text += ['$scope module dimms[%d].memory $end' % (rank//2),
                 '$scope module rank_instances[%d].rank $end' % (rank%2),
                 '$scope module Micron_model.instance_of_sdram_devices[0].iDDR4 $end']
        for pin, width in r.PINS.items():
            code = 'v%d_%s' % (rank, pin);codes[rank,pin] = code
            text.append('$var wire %d %s %s $end' % (width, code, pin))
        text += ['$upscope $end']*3
    text += ['$upscope $end', '$enddefinitions $end', '#0', '$dumpvars']
    base = {pin:'0'*width for pin,width in r.PINS.items()}
    base.update(CK='01', CKE='1')
    for (rank,pin),code in codes.items():text.append('b%s %s' % (base[pin],code))
    text += ['$end', '#625']
    for rank in range(ranks):text.append('b10 '+codes[rank,'CK'])
    for index, changes in enumerate(commands):
        rise = 1875+1250*index
        text.append('#'+str(rise-625))
        for rank in range(ranks):text.append('b01 '+codes[rank,'CK'])
        text.append('#'+str(rise if same_edge else rise-100))
        pins = dict(RESET_n='1', CKE='1', CS_n='0', ACT_n='1', TEN='0',
                    RAS_n_A16='0', CAS_n_A15='0', WE_n_A14='0', ADDR='0'*14)
        pins.update(changes)
        for rank in range(ranks):
            for pin,value in pins.items():text.append('b%s %s' % (value,codes[rank,pin]))
        if not same_edge:text.append('#'+str(rise))
        for rank in range(ranks):text.append('b10 '+codes[rank,'CK'])
    text += ['#'+str(2500+1250*len(commands))]
    return '\n'.join(text)+'\n'


def command(bits, **changes):
    return dict(zip(('RAS_n_A16','CAS_n_A15','WE_n_A14'),bits),**changes)


class RankCommands(unittest.TestCase):
    def analyze(self, text, windows=None):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.vcd';path.write_text(text)
            return r.review(path,windows)

    def test_four_ranks_exact_encodings(self):
        rows=[command(bits) for bits in ('000','001','010','011','100','101','110','111')]
        rows += [command('111',ACT_n='0'), command('010',ADDR=bin(1<<10)[2:]),
                 command('101',ADDR=bin(1<<10)[2:]),command('100',ADDR=bin(1<<10)[2:]),
                 command('110',ADDR=bin(1<<10)[2:]),command('111',CS_n='1')]
        out=self.analyze(waveform(rows))
        expected=dict.fromkeys(('RESET','MRS','REF','PRE','RESERVED','WR','RD','ZQCS','NOP',
                                'ACT','PREA','RDA','WRA','ZQCL','DES'),1)
        self.assertTrue(out['all_four_rank_interfaces_present'])
        for row in out['ranks'].values():self.assertEqual(row['windows'][0]['counts'],expected)
        for key in ('protocol_timing_qualified','refresh_schedule_qualified','ready_for_power_estimate',
                    'native_activity_r1_qualified','board_power_qualified'):self.assertFalse(out[key])

    def test_self_refresh_and_cke_exit_are_not_refresh(self):
        rows=[command('001',CKE='0'), command('001',CKE='0'),command('001'),command('001')]
        counts=self.analyze(waveform(rows))['ranks']['slot0/rank0']['windows'][0]['counts']
        self.assertEqual(counts,dict(RESET=1,SRE_ENCODING=1,CKE_LOW=1,CKE_EXIT=1,REF=1))

    def test_same_time_changes_are_ambiguous(self):
        out=self.analyze(waveform([command('001'),command('101')],same_edge=True))
        counts=out['ranks']['slot0/rank0']['windows'][0]['counts']
        self.assertEqual(counts,dict(RESET=1,AMBIGUOUS_EDGE=2))

    def test_unknown_and_test_mode_are_not_refresh(self):
        for pin,value in [('CS_n','x'),('ACT_n','x'),('RAS_n_A16','z'),('BG','x'),
                          ('BA','z'),('ADDR','x'),('TEN','1'),('RESET_n','x')]:
            with self.subTest(pin=pin):
                out=self.analyze(waveform([command('001',**{pin:value})]))
                self.assertNotIn('REF',out['ranks']['slot0/rank0']['windows'][0]['counts'])

    def test_missing_ranks_never_inferred(self):
        out=self.analyze(waveform([command('001')],ranks=1))
        self.assertFalse(out['all_four_rank_interfaces_present'])
        self.assertEqual(sum(row['observed'] for row in out['ranks'].values()),1)

    def test_missing_pin_is_explicit(self):
        text=waveform([command('001')],ranks=1).replace(' TEN $end',' NOT_TEN $end')
        out=self.analyze(text)
        self.assertEqual(out['ranks']['slot0/rank0']['missing_pins'],['TEN'])

    def test_half_open_window_and_refresh_gap(self):
        text=waveform([command('001')]*3)
        windows=[dict(name='a',start_ps=1875,end_ps=3125),dict(name='b',start_ps=3125,end_ps=5000)]
        a,b=self.analyze(text,windows)['ranks']['slot0/rank0']['windows']
        self.assertEqual(a['refresh_times_ps'],[1875]);self.assertEqual(b['refresh_times_ps'],[3125,4375])
        self.assertIsNone(a['maximum_observed_refresh_gap_ps'])
        self.assertEqual(b['maximum_observed_refresh_gap_ps'],1250)

    def test_invalid_extent_and_window_contract(self):
        text=waveform([command('001')])
        for window in [dict(name='a',start_ps=-1,end_ps=100),dict(name='a',start_ps=0,end_ps=100000),
                       dict(name='a',start_ps=100,end_ps=100),dict(name='a',start_ps=0.0,end_ps=100)]:
            with self.subTest(window=window),self.assertRaises(ValueError):self.analyze(text,[window])

    def test_corrupt_vcd_rejected(self):
        text=waveform([command('001')],ranks=1)
        variants=[text.replace('1 ps','10 ps'),text.replace('$enddefinitions $end',''),
                  text+'#1\n',text.replace('wire 2 v0_CK','wire 1 v0_CK'),
                  text+'$dumpoff\n',text+'b101 v0_CK\n',text+'1undeclared\n']
        for index,bad in enumerate(variants):
            with self.subTest(index=index),self.assertRaises(ValueError):self.analyze(bad)

    def test_ambiguous_clock_does_not_count_command(self):
        text=waveform([command('001')],ranks=1).replace('#1875\nb10 v0_CK','#1875\nbxx v0_CK\nb10 v0_CK')
        out=self.analyze(text)['ranks']['slot0/rank0']['windows'][0]
        self.assertNotIn('REF',out['counts']);self.assertGreater(out['clock_problem_events'],0)


if __name__=='__main__':
    unittest.main(verbosity=2)
