"""Explicit vendor-model limits and complete trace boundaries for AXI4 pair."""
from check_ntrpu_axi4_pair import review as review_transactions,parse
from check_ntrpu_axi4_drain import require

FIFO_WARNING='WARNING: Behavioral models for independent clock FIFO configurations do not model synchronization delays. The behavioral models are functionally correct, and will represent the behavior of the configured FIFO. See the FIFO Generator User Guide for more information.'


def review(text,period):
    require(text.splitlines().count(FIFO_WARNING)==5,'five identified FIFO model notices')
    # Account for the exact notices; retain the original text in evidence.
    result=review_transactions('\n'.join(x for x in text.splitlines() if x!=FIFO_WARNING),period)
    events={x['name']:x for x in (parse(line) for line in text.splitlines() if line.startswith('AXI4_PAIR_EVENT '))}
    for side,clock,first_phase in [('H',7000,3500),('D',period,600+period//2)]:
        times=[parse(line)['t'] for line in text.splitlines() if line.startswith('AXI4_CDC_BUS side='+side+' ')]
        first=first_phase+((events['START']['t']-first_phase)//clock+1)*clock
        require(times[0]==first,'first clock observation missing')
        require(0<=events['TERMINAL']['t']-times[-1]<clock,'last clock observation missing')
    result['vendor_model_notices']=5
    result['data_cdc_synchronization_delay_modeled']=False
    result['timing_limit']='FIFO Generator behavioral models represent functional transactions but explicitly omit synchronization delays. Data-channel latency and throughput here are simulation measurements, not physical CDC timing.'
    return result
