"""PG213 512-bit RQ2/RC4 packet reconstruction for the generated XDMA profile.

This is a software evidence decoder, not a synthesized drain provider. It
consumes accepted beats; callers must separately validate VALID/READY stalls.
Reference fields: docs/fpga/board/XDMA_PCIE_INVENTORY_PLAN.md.
"""
from dataclasses import dataclass, field


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bits(value, low, width):
    return (value >> low) & ((1 << width) - 1)


@dataclass(frozen=True)
class LogicWord:
    value: int
    known: int
    width: int

    @classmethod
    def parse(cls, raw, width):
        require(raw and set(raw) <= set('01xz') and len(raw) <= width, 'invalid four-state word')
        raw = raw.rjust(width, raw[0] if raw[0] in 'xz' else '0')
        value = int(''.join(c if c in '01' else '0' for c in raw), 2)
        known = int(''.join('1' if c in '01' else '0' for c in raw), 2)
        return cls(value, known, width)


@dataclass
class Packet:
    interface: str
    start_ps: int
    start_lane: int
    first_be: int | None = None
    last_be: int | None = None
    end_ps: int | None = None
    words: list[int] = field(default_factory=list)
    byte_en: list[int] = field(default_factory=list)
    discontinued: bool = False
    sequence_number: int | None = None
    known_words: list[int] = field(default_factory=list)

    @property
    def data(self):
        return b''.join(w.to_bytes(4, 'little') for w in self.words)

    @property
    def header(self):
        size = 16 if self.interface == 'rq' else 12
        require(len(self.data) >= size, 'truncated packet descriptor')
        return int.from_bytes(self.data[:size], 'little')

    def require_known(self, mask, message):
        known = sum(word << (32*n) for n, word in enumerate(self.known_words))
        require(known & mask == mask, message)


class StraddleDecoder:
    """Retain a partial packet; emit every packet ending on an accepted beat."""

    def __init__(self, interface):
        require(interface in ('rq', 'rc'), 'interface must be rq or rc')
        self.interface = interface
        self.pending = None
        self.last_time = None
        self.accepted_beats = 0
        self.completed_packets = 0
        self.max_starts_per_beat = 0
        self.max_ends_per_beat = 0
        self.discontinued = False
        self.events = []

    def accept(self, time_ps, data, user):
        require(isinstance(time_ps, int) and time_ps >= 0, 'invalid beat timestamp')
        require(self.last_time is None or time_ps > self.last_time, 'non-increasing accepted-beat time')
        if isinstance(data, LogicWord):
            require(data.width == 512, 'four-state data width')
            known, data = data.known, data.value
        else:
            known = (1 << 512)-1
        require(isinstance(data, int) and 0 <= data < 1 << 512, 'unknown or oversized packet data')
        width = 137 if self.interface == 'rq' else 161
        require(isinstance(user, int) and 0 <= user < 1 << width, 'unknown or oversized packet sideband')
        rq = self.interface == 'rq'
        slots = 2 if rq else 4
        sop_bit, sop_ptr_bit, eop_bit, eop_ptr_bit = (20, 22, 26, 28) if rq else (64, 68, 76, 80)
        sop = bits(user, sop_bit, slots)
        eop = bits(user, eop_bit, slots)
        require(sop in [(1 << n)-1 for n in range(slots+1)], 'reserved SOP mask')
        require(eop in [(1 << n)-1 for n in range(slots+1)], 'reserved EOP mask')
        starts = [4 * bits(user, sop_ptr_bit + 2*n, 2) for n in range(sop.bit_count())]
        ends = [bits(user, eop_ptr_bit + 4*n, 4) for n in range(eop.bit_count())]
        require(starts == sorted(set(starts)), 'unordered or duplicate SOP pointers')
        require(ends == sorted(set(ends)), 'unordered or duplicate EOP pointers')
        if rq:
            require(all(n in (0, 8) for n in starts), 'RQ SOP outside lanes 0/32')
            require(len(starts) < 2 or starts == [0, 8], 'RQ second SOP must start at lane32')
            if self.pending is None and starts:
                require(starts[0] == 0, 'RQ first packet without continuation must start at lane0')
        discontinue = bool(bits(user, 36 if rq else 96, 1))
        require(not (discontinue and starts), 'discontinue cannot accompany a new straddled packet')
        if not rq and discontinue:
            require(ends, 'RC discontinue without final beat')
        require(self.pending is not None or starts, 'accepted beat contains no packet')
        require(not starts or self.pending is None or starts[0] != 0, 'SOP overlaps a partial packet')
        result = []
        self.events = []
        header_words = 4 if rq else 3
        for lane in range(16):
            if lane in starts:
                require(self.pending is None, 'new SOP before previous EOP')
                index = starts.index(lane)
                self.pending = Packet(self.interface, time_ps, lane*4,
                    bits(user, index*4, 4) if rq else None,
                    bits(user, 8+index*4, 4) if rq else None)
                if rq:
                    require(bits(user, 16, 4) == 0, 'RQ profile requires DWORD alignment')
                    self.pending.sequence_number = bits(user, 61+index*6, 6)
                self.events.append(('start', self.pending))
            if self.pending is not None:
                p = self.pending
                p.words.append(bits(data, lane*32, 32))
                p.known_words.append(bits(known, lane*32, 32))
                require(len(p.words) <= header_words + 1024, 'packet exceeds maximum payload size')
                if not rq:
                    enable = bits(user, lane*4, 4)
                    require(len(p.words) > header_words or enable == 0, 'RC descriptor bytes enabled as payload')
                    p.byte_en.extend((enable >> n) & 1 for n in range(4))
                p.discontinued |= discontinue
            elif not rq:
                require(bits(user, lane*4, 4) == 0, 'RC byte enable outside a packet')
            if lane in ends:
                require(self.pending is not None, 'EOP without packet')
                p = self.pending
                require(len(p.words) >= header_words, 'EOP truncates descriptor')
                p.end_ps = time_ps
                result.append(p)
                self.events.append(('end', p))
                self.pending = None
        self.discontinued |= discontinue
        self.last_time = time_ps
        self.accepted_beats += 1
        self.completed_packets += len(result)
        self.max_starts_per_beat = max(self.max_starts_per_beat, len(starts))
        self.max_ends_per_beat = max(self.max_ends_per_beat, len(ends))
        return result

    def finish(self):
        require(self.pending is None, 'capture ends with an incomplete PCIe packet')


def decode_memory_request(packet):
    """Decode this project's memory RQ traffic; reject unsupported request types."""
    require(packet.interface == 'rq' and packet.end_ps is not None, 'not a complete RQ packet')
    h = packet.header
    packet.require_known((1 << 128)-1, 'unknown RQ descriptor bits')
    kind, count = bits(h, 75, 4), bits(h, 64, 11)
    require(kind in (0, 1), 'unsupported endpoint request type')
    require(1 <= count <= 1024, 'invalid memory request DWORD count')
    require(packet.first_be is not None and packet.last_be is not None, 'missing RQ byte enables')
    require(count != 1 or packet.last_be == 0, 'one-DWORD request has last byte enables')
    require(count == 1 or packet.first_be != 0 and packet.last_be != 0, 'empty multi-DWORD request boundary')
    expected_words = 4 + (count if kind == 1 else 0)
    if not packet.discontinued:
        require(len(packet.words) == expected_words, 'RQ packet length differs from descriptor')
    enable = []
    for dw in range(count):
        mask = packet.first_be if dw == 0 else packet.last_be if dw == count-1 else 15
        enable.extend((mask >> n) & 1 for n in range(4))
    payload = packet.data[16:]
    if kind == 1 and not packet.discontinued:
        packet.require_known(sum(255 << (128+8*n) for n,v in enumerate(enable) if v),
                             'unknown enabled RQ payload bits')
    address = bits(h, 2, 62) << 2
    offsets = [n for n, v in enumerate(enable) if v]
    first = offsets[0] if offsets else 0
    span = offsets[-1] - first + 1 if offsets else 1
    return dict(kind='read' if kind == 0 else 'write', address=address,
                address_type=bits(h, 0, 2), dword_count=count,
                requested_bytes=sum(enable), byte_offsets=offsets,
                first_byte_address=address+first, completion_span_bytes=span,
                tag=bits(h, 96, 8), requester_id=bits(h, 80, 16),
                sequence_number=packet.sequence_number,
                requester_id_enable=bits(h, 120, 1), tc=bits(h, 121, 3), attributes=bits(h, 124, 3),
                poisoned=bool(bits(h, 79, 1)), discontinued=packet.discontinued,
                start_ps=packet.start_ps, end_ps=packet.end_ps,
                payload=payload, enabled_payload=bytes(v for v,e in zip(payload, enable) if e) if kind else b'')


def completion_header(packet):
    """The descriptor is available on SOP, before a long payload finishes."""
    require(packet.interface == 'rc', 'not an RC packet')
    h = packet.header
    packet.require_known(15 << 12, 'unknown RC error code')
    error = bits(h, 12, 4)
    timeout = error == 9
    valid_header = (1 << 30 | 255 << 48 | 255 << 64) if timeout else (1 << 96)-1
    packet.require_known(valid_header, 'unknown valid RC descriptor bits')
    count = bits(h, 32, 11)
    return dict(tag=bits(h, 64, 8), requester_function=bits(h, 48, 8),
                request_completed=bool(bits(h, 30, 1)), error_code=error,
                requester_id=None if timeout else bits(h, 48, 16),
                lower_address=None if timeout else bits(h, 0, 12),
                byte_count=None if timeout else bits(h, 16, 13),
                dword_count=None if timeout else count,
                completion_status=None if timeout else bits(h, 43, 3),
                poisoned=None if timeout else bool(bits(h, 46, 1)),
                tc=None if timeout else bits(h, 89, 3),
                attributes=None if timeout else bits(h, 92, 3),
                discontinued=packet.discontinued,
                start_ps=packet.start_ps, end_ps=packet.end_ps,
                normal=error == 0 and bits(h, 43, 3) == 0 and not bits(h, 46, 1) and not packet.discontinued)


def decode_completion(packet):
    """Decode RC without treating terminal error completions as successful reads."""
    require(packet.end_ps is not None, 'not a complete RC packet')
    row = completion_header(packet)
    if row['error_code'] != 9 and not packet.discontinued:
        require(row['dword_count'] <= 1024 and len(packet.words) == 3 + row['dword_count'],
                'RC packet length differs from descriptor')
    payload = packet.data[12:]
    require(len(packet.byte_en) == len(packet.data), 'RC enabled-byte accounting')
    if row['normal']:
        packet.require_known(sum(255 << (96+8*n) for n,e in enumerate(packet.byte_en[12:]) if e),
                             'unknown enabled RC payload bits')
    row.update(payload=payload,
               enabled_payload=bytes(v for v,e in zip(payload, packet.byte_en[12:]) if e))
    return row
