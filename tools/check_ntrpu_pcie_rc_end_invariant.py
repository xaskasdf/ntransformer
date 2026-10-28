"""Check the settled-state monitor and its actual corrupted-register mutant."""
import re


def review_normal(log, minimum_checks):
    pattern = (r'RC_END_INVARIANT_CHECKS total=(\d+) cold=(\d+) active=(\d+) '
               r'inactive=(\d+) hold=(\d+) headers=(\d+)')
    lines = [line for line in log.splitlines() if line.startswith('RC_END_INVARIANT_CHECKS')]
    if len(lines) != 1 or not (match := re.fullmatch(pattern, lines[0])):
        raise ValueError('Missing/duplicate/malformed retained-end invariant summary')
    result = dict(zip(['total', 'cold', 'active', 'inactive', 'hold', 'headers'],
                      map(int, match.groups())))
    if result['total'] < minimum_checks or any(value <= 0 for value in result.values()):
        raise ValueError('Missing retained-end invariant coverage')
    if result['active'] + result['inactive'] != result['total']:
        raise ValueError('Inconsistent retained-end invariant sample counts')
    if any(result[key] > result['total'] for key in ('cold', 'hold', 'headers')):
        raise ValueError('Impossible retained-end transition count')
    if re.search(r'^\s*(Fatal:|Failure:|ERROR:)', log, re.M):
        raise ValueError('Invariant bank contains a failure')
    return result


def review_mutant(log):
    if log.splitlines().count('RC_END_CACHE_MUTATION_INJECTED') != 1:
        raise ValueError('Missing actual retained-end corruption')
    fatals = [line for line in log.splitlines() if line.startswith('Fatal:')]
    if (len(fatals) != 1 or not re.fullmatch(
            r'Fatal: RC_END_CACHE_INVARIANT end=3fff delivered=[0-9a-fx]+ low=[0-9a-fx]+',
            fatals[0]) or 'RC_STAGE_SUITE_PASS' in log):
        raise ValueError('Corrupted retained-end register was not rejected by its invariant')
    if log.index('RC_END_CACHE_MUTATION_INJECTED') > log.index(fatals[0]):
        raise ValueError('Mutation happened after its claimed detection')
    return dict(name='corrupt_context_end', rejected=True, fatal=fatals[0])


def exercise_controls(normal, mutant, minimum_checks):
    """Reject damaged evidence independently of the original event-row controls."""
    counts = review_normal(normal, minimum_checks)
    review_mutant(mutant)
    summary = next(s for s in normal.splitlines() if s.startswith('RC_END_INVARIANT_CHECKS'))
    normal_cases = [
        ('missing_summary', normal.replace(summary, '', 1)),
        ('duplicate_summary', normal + '\n' + summary + '\n'),
        ('inconsistent_counts', normal.replace('total=' + str(counts['total']),
                                             'total=' + str(counts['total'] + 1), 1)),
        ('hidden_failure', normal + '\nFatal: hidden invariant failure\n'),
    ]
    for key in ('cold', 'hold', 'headers'):
        normal_cases.append(('missing_' + key, normal.replace(
            key + '=' + str(counts[key]), key + '=0', 1)))
    mutant_cases = [
        ('missing_injection', mutant.replace('RC_END_CACHE_MUTATION_INJECTED', 'IGNORED', 1)),
        ('suppressed_invariant_fatal', mutant.replace('Fatal: RC_END_CACHE_INVARIANT',
                                                    'Ignored: RC_END_CACHE_INVARIANT', 1)),
        ('false_mutant_pass', mutant + '\nRC_STAGE_SUITE_PASS\n'),
    ]
    controls = []
    for cases, original, checker in ((normal_cases, normal, lambda log: review_normal(log, minimum_checks)),
                                     (mutant_cases, mutant, review_mutant)):
        for name, damaged in cases:
            if damaged == original:
                raise AssertionError('Control did not change the evidence: ' + name)
            try:
                checker(damaged)
            except ValueError as exc:
                controls.append(dict(name=name, rejected=True, reason=str(exc)))
            else:
                raise AssertionError('Damaged invariant evidence admitted: ' + name)
    return controls
