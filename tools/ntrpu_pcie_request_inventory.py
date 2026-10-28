"""Offline memory RQ/RC inventory for PG213 client tags, RQ2/RC4, DWORD alignment.

No source fence, host posted-write receipt, reset approval or hardware ACK is
inferred here. Configured requester IDs and effective attributes must be bound
to the native core profile by the caller. Protocol violations abort review.
"""
from ntrpu_pcie_straddle import (
    StraddleDecoder, completion_header, decode_completion, decode_memory_request, require,
)


class AcceptedBeatMonitor:
    """Validate offered beats on every rising edge; emit only handshakes."""
    def __init__(self, interface):
        require(interface in ('rq','rc'), 'stream interface')
        self.interface = interface
        self.retained = None
        self.stalled_cycles = self.accepted_beats = 0

    def sample(self, valid, ready, data, user):
        require(valid in (0,1) and ready in ((0,15) if self.interface=='rq' else (0,1)),
                'known and replicated PCIe handshake')
        payload = (data,user)
        if self.retained is not None:
            require(valid == 1 and payload == self.retained, 'PCIe payload changed under backpressure')
        if not valid:
            return None
        require(None not in payload, 'unknown PCIe payload/sideband')
        if not ready:
            self.retained = payload
            self.stalled_cycles += 1
            return None
        self.retained = None
        self.accepted_beats += 1
        return payload

    def finish(self):
        require(self.retained is None, 'capture ends with unaccepted offered PCIe beat')


class RequestInventory:
    def __init__(self, requester_ids, capacity=256, attribute_mask=3, ari_enabled=False):
        self.function_mask = 255 if ari_enabled else 7
        require(requester_ids and all(0 <= f <= self.function_mask and 0 <= i < 65536 and i & self.function_mask == f
                                     for f, i in requester_ids.items()), 'requester identity configuration')
        require(1 <= capacity <= 256 and 0 <= attribute_mask <= 3, 'inventory profile capacity/attributes')
        self.requester_ids = dict(requester_ids)
        self.capacity = capacity
        self.attribute_mask = attribute_mask
        self.rq = StraddleDecoder('rq')
        self.rc = StraddleDecoder('rc')
        self.tags = {}
        self.live = {}
        self.payloads = {}
        self.requests = []
        self.completions = []
        self.completed_reads = []
        self.faults = []
        self.timeline = []
        self.last_time = None
        self.peak_tags = self.peak_reads = self.peak_span_bytes = 0

    def _sample(self, time_ps, event):
        row = dict(time_ps=time_ps, event=event, allocated_tags=len(self.tags),
                   outstanding_reads=len(self.live), partial_completions=len(self.payloads),
                   remaining_span_bytes=sum(r['remaining'] for r in self.live.values()),
                   partial_rq=int(self.rq.pending is not None), partial_rc=int(self.rc.pending is not None))
        self.peak_tags = max(self.peak_tags, row['allocated_tags'])
        self.peak_reads = max(self.peak_reads, row['outstanding_reads'])
        self.peak_span_bytes = max(self.peak_span_bytes, row['remaining_span_bytes'])
        self.timeline.append(row)

    def cycle(self, time_ps, *, rq=None, rc=None):
        """Pass only accepted beats as (data, user); RC precedes same-edge RQ reuse."""
        require(self.last_time is None or time_ps > self.last_time, 'non-increasing inventory cycle')
        self.last_time = time_ps
        if rc is not None:
            self.rc.accept(time_ps, *rc)
            for event, packet in self.rc.events:
                if event == 'start':
                    self._completion_start(packet)
                else:
                    self._completion_end(packet)
                self._sample(time_ps, 'rc_'+event)
        if rq is not None:
            for packet in self.rq.accept(time_ps, *rq):
                self._request(packet)
                self._sample(time_ps, 'rq_end')
        self._sample(time_ps, 'cycle_end')

    def _request(self, packet):
        row = decode_memory_request(packet)
        row['serial'] = len(self.requests)
        require(row['address_type'] == 0, 'unqualified translated address profile')
        require((row['address'] & 4095) + row['dword_count']*4 <= 4096, 'memory request crosses 4KiB boundary')
        function = row['requester_id'] & self.function_mask
        require(function in self.requester_ids, 'unconfigured requester function')
        expected_id = self.requester_ids[function]
        row['effective_requester_id'] = row['requester_id'] if row['requester_id_enable'] else expected_id
        require(row['effective_requester_id'] == expected_id, 'requester ID override mismatch')
        row['effective_attributes'] = row['attributes'] & self.attribute_mask
        self.requests.append(row)
        if row['poisoned'] or row['discontinued']:
            self.faults.append(dict(kind='request_error', serial=row['serial'], time_ps=packet.end_ps))
            return
        if row['kind'] == 'write':
            return  # Accepted posted traffic is not evidence of host receipt.
        key = (function, row['tag'])
        require(key not in self.tags, 'request tag reused before terminal descriptor')
        require(len(self.tags) < self.capacity, 'request tag capacity exceeded')
        entry = dict(request=row, remaining=row['completion_span_bytes'],
                     next_address=row['first_byte_address'], logical_payload=bytearray(),
                     completion_count=0, terminal_header_ps=None, failed=False)
        self.tags[key] = entry
        self.live[row['serial']] = entry

    def _completion_start(self, packet):
        row = completion_header(packet)
        key = (row['requester_function'] & self.function_mask, row['tag'])
        require(key in self.tags, 'completion for unknown or already retired tag')
        entry = self.tags[key]
        request = entry['request']
        # All error paths remain sticky failures; timeout has only function/tag/terminal valid.
        if row['normal']:
            require(row['requester_id'] == request['effective_requester_id'], 'completion requester ID mismatch')
            require(row['tc'] == request['tc'] and row['attributes'] == request['effective_attributes'],
                    'completion TC/attributes mismatch')
            require(row['byte_count'] == entry['remaining'], 'completion remaining byte count mismatch')
            require(row['lower_address'] == entry['next_address'] & 4095, 'completion lower address mismatch')
            require(1 <= row['dword_count'] <= 1024, 'successful memory completion has no valid payload')
            available = row['dword_count']*4 - (entry['next_address'] & 3)
            delivered = min(entry['remaining'], available)
            require(delivered > 0, 'empty completion progress')
            require(row['dword_count'] == ((entry['next_address'] & 3)+delivered+3)//4,
                    'completion carries excess payload DWORDs')
            require(row['request_completed'] == (delivered == entry['remaining']),
                    'terminal descriptor disagrees with remaining bytes')
        else:
            delivered = 0
            entry['failed'] = True
            self.faults.append(dict(kind='completion_error', serial=request['serial'],
                                   error_code=row['error_code'], time_ps=packet.start_ps))
        self.payloads[id(packet)] = (entry, row, delivered)
        if row['request_completed']:
            # Free tag immediately, but keep this transaction until its payload EOP.
            del self.tags[key]
            entry['terminal_header_ps'] = packet.start_ps

    def _completion_end(self, packet):
        require(id(packet) in self.payloads, 'completion end without retained descriptor')
        entry, header, delivered = self.payloads.pop(id(packet))
        row = decode_completion(packet)
        request = entry['request']
        row['request_serial'] = request['serial']
        if row['normal']:
            require(header['normal'], 'completion error disappeared')
            first = entry['next_address'] & 3
            wanted_enable = [int(first <= n < first+delivered) for n in range(len(row['payload']))]
            require(packet.byte_en[12:] == wanted_enable, 'completion enabled byte range mismatch')
            require(len(row['enabled_payload']) == delivered, 'completion payload byte count mismatch')
            offsets = set(request['byte_offsets'])
            origin = entry['next_address'] - request['address']
            entry['logical_payload'].extend(v for n, v in enumerate(row['enabled_payload']) if origin+n in offsets)
            entry['remaining'] -= delivered
            entry['next_address'] += delivered
        elif header['normal']:
            entry['failed'] = True
            self.faults.append(dict(kind='payload_error', serial=request['serial'], time_ps=packet.end_ps))
        entry['completion_count'] += 1
        self.completions.append(row)
        if header['request_completed']:
            del self.live[request['serial']]
            self.completed_reads.append(dict(serial=request['serial'], tag=request['tag'],
                request_ps=request['start_ps'], terminal_header_ps=entry['terminal_header_ps'],
                payload_end_ps=packet.end_ps, payload=bytes(entry['logical_payload']),
                completion_count=entry['completion_count'], remaining_span_bytes=entry['remaining'],
                successful=not entry['failed'] and row['normal'] and entry['remaining'] == 0))

    @property
    def empty(self):
        return not (self.live or self.tags or self.payloads or self.rq.pending or self.rc.pending)

    def finish(self):
        self.rq.finish()
        self.rc.finish()
        require(self.empty, 'capture ends with outstanding PCIe obligations')
        require(not self.faults, 'PCIe inventory has sticky errors')
