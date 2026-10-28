"""Adversarial transcript checks shared by native and XSim review."""
import re
from check_ntrpu_kv_cache_r1 import review

def controls(text, seed, watchdog):
    controls = []
    
    def corrupt(label, pattern, replace):
        bad, count = re.subn(pattern, replace, text, count=1, flags=re.M)
        assert count == 1 and bad != text, label
        try: review(bad, seed, watchdog)
        except (AssertionError, ValueError, KeyError, IndexError) as exc:
            controls.append(dict(name=label, rejected=True, reason=str(exc)))
        else: raise AssertionError('Accepted corruption '+label)
    
    corrupt('aw_address', r'^(NTRPU_KVC_AW .* address=)[0-9a-f]+', r'\g<1>0000000000000000')
    corrupt('write_data', r'^(NTRPU_KVC_W .* data=)[0-9a-f]+', r'\g<1>0')
    corrupt('write_last', r'^(NTRPU_KVC_W .* last=)1', r'\g<1>0')
    corrupt('b_error_hidden', r'^(NTRPU_KVC_B .* resp=)2', r'\g<1>0')
    corrupt('early_publication', r'^(NTRPU_KVC_APPEND_DONE .* committed=)0', r'\g<1>1')
    corrupt('lease_identity', r'^(NTRPU_KVC_LEASE .* lease=)[0-9a-f]+', r'\g<1>0')
    corrupt('lease_prefix', r'^(NTRPU_KVC_LEASE_DONE .* prefix=)1', r'\g<1>0')
    corrupt('ar_address', r'^(NTRPU_KVC_AR .* address=)[0-9a-f]+', r'\g<1>0')
    corrupt('r_data', r'^(NTRPU_KVC_R .* data=)[0-9a-f]+', r'\g<1>0')
    corrupt('tile_lease', r'^(NTRPU_KVC_TILE .* lease=)[0-9a-f]+', r'\g<1>0')
    corrupt('tile_payload', r'^(NTRPU_KVC_TILE .* data=)[0-9a-f]+', r'\g<1>0')
    corrupt('read_completion_bytes', r'^(NTRPU_KVC_READ_DONE .* bytes=)512', r'\g<1>448')
    corrupt('read_replay_position', r'^(NTRPU_KVC_READ_DONE .* next=)1', r'\g<1>0')
    corrupt('reset_phase', r'^(NTRPU_KVC_READ_ABORT .* phase=)0', r'\g<1>1')
    corrupt('missing_cancel', r'^NTRPU_KVC_CANCEL .*\n', '')
    corrupt('missing_operation', r'^NTRPU_KVC_READ .*\n', '')
    corrupt('duplicate_operation', r'^(NTRPU_KVC_READ .*\n)', r'\1\1')
    corrupt('wrong_epoch_revocation', r'^(NTRPU_KVC_EPOCH .* expected=)[0-9a-f]+', r'\g<1>0')
    corrupt('missing_terminal', r'^NTRPU_KVC_PASS .*\n', '')
    return controls
