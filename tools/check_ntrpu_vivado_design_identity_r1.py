"""Distinguish the Vivado design object's NAME from its synthesized top module."""
import copy
import re


def review(evidence, expected):
    assert re.fullmatch(r'ntrpu_[a-z0-9_]+', expected)
    files = evidence['files']
    counts = dict(line.split('\t') for line in files['output/primitive_counts.tsv'].splitlines())
    assert counts['top'].strip(), 'Missing design object name'
    assert counts['part'].replace('-', '') == 'xcvu5pflva21042i'
    reports = {}
    for name in ('utilization.rpt', 'hierarchy.rpt', 'timing.rpt'):
        text = files['output/'+name]
        top = re.findall(r'^\|\s*Design\s*:\s*(\S+)\s*$', text, re.M)
        state = re.findall(r'^\|\s*Design State\s*:\s*(\S+)\s*$', text, re.M)
        assert top == [expected] and state == ['Synthesized'], name
        reports[name] = top[0]
    commands = re.findall(r'^Command: synth_design -top (\S+) -mode out_of_context -part (\S+)\s*$',
                          evidence['console'], re.M)
    assert commands == [(expected, 'xcvu5p-flva2104-2-i')], 'Actual synthesis command mismatch'
    return dict(verified_top=expected, design_object_name=counts['top'], report_tops=reports,
                synthesis_command_verified=True)


def controls(evidence, expected):
    rejected = []
    for kind in ('utilization', 'hierarchy', 'timing', 'state', 'command', 'part', 'empty_name'):
        e = copy.deepcopy(evidence)
        if kind in ('utilization', 'hierarchy', 'timing'):
            name = 'output/'+kind+'.rpt'
            e['files'][name] = re.sub(r'(^\|\s*Design\s*:\s*)\S+', r'\1wrong_top', e['files'][name], flags=re.M)
        elif kind == 'state':
            e['files']['output/utilization.rpt'] = e['files']['output/utilization.rpt'].replace('Synthesized','Elaborated')
        elif kind == 'command':
            e['console'] = e['console'].replace('Command: synth_design -top '+expected,
                                                'Command: synth_design -top wrong_top')
        elif kind == 'part':
            e['files']['output/primitive_counts.tsv'] = e['files']['output/primitive_counts.tsv'].replace('xcvu5p-flva2104-2-i','xcvu9p-flga2104-2-i')
        else:
            e['files']['output/primitive_counts.tsv'] = re.sub(r'^top\t[^\n]*', 'top\t', e['files']['output/primitive_counts.tsv'], flags=re.M)
        assert e != evidence
        try: review(e,expected)
        except AssertionError: rejected.append(kind)
        else: raise AssertionError('Accepted '+kind)
    return rejected
