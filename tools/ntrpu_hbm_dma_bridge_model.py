"""Bounded HBM-to-C2H contract model; no RTL, CDC primitive or PCIe emulation.

Calls represent accepted events in their respective clock domains. The caller
schedules reader and DMA edges independently. Route authorization, remote ACK
authentication and physical isolation are responsibilities outside this model.
"""
from collections import deque
from dataclasses import dataclass

from ntrpu_transport_model import RegionHandle, TransferId, TransportError, require, unsigned


U64 = 1 << 64


def bounded(base, size, limit=U64):
    unsigned(base)
    unsigned(size, positive=True)
    require(base < limit and size <= limit - base, 'RANGE')


def overlaps(a, size_a, b, size_b):
    return a < b + size_b and b < a + size_a


@dataclass(frozen=True)
class Route:
    source: RegionHandle
    target: RegionHandle
    window_generation: int
    model_epoch: int
    source_base: int
    source_bytes: int
    target_base: int
    target_bytes: int
    writeback_base: int
    writeback_bytes: int = 256


@dataclass(frozen=True)
class Span:
    identity: TransferId
    source: RegionHandle
    target: RegionHandle
    window_generation: int
    model_epoch: int
    source_address: int
    target_address: int
    length: int


@dataclass(frozen=True)
class Descriptor:
    channel: int
    identity: TransferId
    src_addr: int  # C2H metadata writeback, NOT the HBM address.
    dst_addr: int
    length: int
    control: int = 0x10  # EOP; no Stop, reserved bits zero.


@dataclass(frozen=True)
class Beat:
    identity: TransferId
    address: int
    data: bytes
    last: bool
    keep: int = (1 << 64) - 1


@dataclass(frozen=True)
class Event:
    identity: TransferId
    target: RegionHandle
    window_generation: int
    length: int
    status: str = 'OK'


@dataclass
class _Context:
    span: Span
    descriptor: Descriptor
    slot: int
    accepted: int = 0
    sent: int = 0
    issued: bool = False
    local_done: bool = False
    done_observed: bool = False
    dma_done: bool = False
    ack: bool = False
    failure: str | None = None


class BridgeModel:
    """Two C2H channels, two contexts and two completion credits per channel.

    Payload queues are abstract CDC mailboxes, not a metastability model.
    Failed contexts remain owned for the lifetime of this instance. There is
    deliberately no warm-reset/rearm or timeout-based resource release API.
    """
    def __init__(self, *, payload_depth=4):
        unsigned(payload_depth, positive=True)
        self.payload_depth = payload_depth
        self.route = None
        self.contexts = {}
        self.order = [deque(), deque()]
        self.pending = [deque(), deque()]
        self.input_order = [deque(), deque()]
        self.outstanding = [deque(), deque()]
        self.payload = [deque(), deque()]
        self.events = [deque(), deque()]
        self.run = [False, False]
        self.halted = False
        self.link_up = True
        self.watermark = 0
        self.faults = []
        self.trace = []

    def _log(self, kind, channel=None, identity=None, **fields):
        self.trace.append(dict(index=len(self.trace), kind=kind, channel=channel,
                               sequence=None if identity is None else identity.sequence, **fields))

    def _channel(self, channel):
        require(type(channel) is int and channel in (0, 1), 'CHANNEL')

    def _fault(self, reason):
        self.halted = True
        self.faults.append(reason)
        for context in self.contexts.values():
            context.failure = context.failure or reason
        self._log('FAULT', reason=reason)

    def _protocol(self, condition, reason):
        if not condition:
            self._fault(reason)
            raise TransportError(reason)

    def _context(self, identity):
        self._protocol(identity in self.contexts, 'UNKNOWN_EVENT')
        return self.contexts[identity]

    def configure(self, route):
        require(not self.contexts, 'MAP_BUSY')
        require(not self.halted, 'HALTED')
        require(isinstance(route, Route), 'ROUTE')
        for handle in [route.source, route.target]:
            require(isinstance(handle, RegionHandle), 'HANDLE')
            for value in [handle.session, handle.generation]:
                unsigned(value, positive=True)
            for value in [handle.node, handle.slot]:
                unsigned(value)
        require(route.source.session == route.target.session, 'SESSION')
        unsigned(route.window_generation, positive=True)
        unsigned(route.model_epoch, positive=True)
        bounded(route.source_base, route.source_bytes, 1 << 34)
        bounded(route.target_base, route.target_bytes)
        bounded(route.writeback_base, route.writeback_bytes)
        require(route.writeback_bytes >= 256 and route.writeback_base % 64 == 0, 'WRITEBACK_RANGE')
        require(not overlaps(route.target_base, route.target_bytes,
                             route.writeback_base, route.writeback_bytes), 'WRITEBACK_OVERLAP')
        if self.route is not None:
            require(route.source.session == self.route.source.session, 'NEW_SESSION_REQUIRES_ISOLATION')
            require(route.window_generation > self.route.window_generation, 'MAP_GENERATION')
        self.route = route
        self._log('CONFIG', generation=route.window_generation)

    def reserve(self, channel, span):
        self._channel(channel)
        require(not self.halted and self.link_up, 'HALTED')
        route = self.route
        require(route is not None and isinstance(span, Span), 'UNCONFIGURED')
        require(isinstance(span.identity, TransferId), 'IDENTITY')
        unsigned(span.identity.sequence, positive=True)
        require(span.source == route.source and span.target == route.target
                and span.window_generation == route.window_generation
                and span.model_epoch == route.model_epoch
                and span.identity.session == route.source.session
                and span.identity.source == route.source.node, 'STALE_CONTEXT')
        require(span.identity.sequence > self.watermark, 'REPLAY')
        require(len(self.order[channel]) < 2, 'CONTEXT_FULL')
        bounded(span.source_address, span.length, 1 << 34)
        bounded(span.target_address, span.length)
        require(span.length <= 4096 and span.length % 64 == 0
                and span.source_address % 64 == 0 and span.target_address % 64 == 0, 'SPAN_ALIGNMENT')
        require(span.source_address // 4096 == (span.source_address + span.length - 1) // 4096,
                'SOURCE_PAGE')
        for address, base, size in [(span.source_address, route.source_base, route.source_bytes),
                                    (span.target_address, route.target_base, route.target_bytes)]:
            require(address >= base and span.length <= size - (address - base), 'ROUTE_RANGE')
        require(not any(overlaps(span.target_address, span.length, c.span.target_address, c.span.length)
                        for c in self.contexts.values()), 'TARGET_BUSY')
        used = {self.contexts[i].slot for i in self.order[channel]}
        slot = next(i for i in range(2) if i not in used)
        writeback = route.writeback_base + (channel * 2 + slot) * 64
        descriptor = Descriptor(channel, span.identity, writeback, span.target_address, span.length)
        self.contexts[span.identity] = _Context(span, descriptor, slot)
        for queue in [self.order, self.pending, self.input_order]:
            queue[channel].append(span.identity)
        self.watermark = span.identity.sequence
        self._log('RESERVE', channel, span.identity, destination=span.target_address,
                  writeback=writeback, length=span.length)
        return descriptor

    def set_run(self, channel, run):
        self._channel(channel)
        self.run[channel] = bool(run)

    def descriptor(self, channel):
        self._channel(channel)
        return self.contexts[self.pending[channel][0]].descriptor if self.pending[channel] else None

    def issue(self, channel, *, ready):
        """Conservative bypass subset: load only while Ready and Run are high."""
        descriptor = self.descriptor(channel)
        if descriptor is None or not ready or not self.run[channel] or not self.link_up or self.halted:
            return None
        # Credit is reserved at issue: Done pulses cannot be backpressured.
        require(len(self.outstanding[channel]) + len(self.events[channel]) < 2, 'EVENT_CREDIT')
        identity = self.pending[channel].popleft()
        self.contexts[identity].issued = True
        self.outstanding[channel].append(identity)
        self._log('ISSUE', channel, identity)
        return descriptor

    def push(self, channel, identity, data, *, last):
        """Reader edge. Accepted obligations may continue draining after halt."""
        self._channel(channel)
        context = self._context(identity)
        self._protocol(bool(self.input_order[channel]) and self.input_order[channel][0] == identity,
                       'PACKET_INTERLEAVE')
        self._protocol(isinstance(data, bytes) and len(data) == 64, 'BEAT_SIZE')
        self._protocol(type(last) is bool and last == (context.accepted + 64 == context.span.length),
                       'LAST_MISMATCH')
        if len(self.payload[channel]) == self.payload_depth:
            return False
        beat = Beat(identity, context.span.target_address + context.accepted, data, last)
        self.payload[channel].append(beat)
        context.accepted += 64
        if last:
            self.input_order[channel].popleft()
        self._log('PUSH', channel, identity, address=beat.address, last=last)
        return True

    def peek(self, channel):
        self._channel(channel)
        if not self.payload[channel]:
            return None
        beat = self.payload[channel][0]
        return beat if self.contexts[beat.identity].issued else None

    def take(self, channel, *, ready):
        """DMA edge; a halt does not withdraw a previously offered payload."""
        beat = self.peek(channel)
        if beat is None or not ready or not self.run[channel]:
            return None
        self.payload[channel].popleft()
        self.contexts[beat.identity].sent += 64
        self._log('SEND', channel, beat.identity, address=beat.address, last=beat.last)
        return beat

    def local_complete(self, identity, length, *, status='OK'):
        context = self._context(identity)
        self._protocol(not context.local_done and length == context.span.length
                       and context.accepted == length and status == 'OK', 'LOCAL_CPL')
        context.local_done = True
        self._log('LOCAL_CPL', context.descriptor.channel, identity)

    def observe_done(self, channel, *, error=None):
        """One status[3] pulse in axi_aclk plus abstract error-policy outcome.

        There is no ID on the real pulse. Association is descriptor FIFO order;
        the future RTL must validate the register/error policy independently.
        """
        self._channel(channel)
        self._protocol(bool(self.outstanding[channel]), 'UNEXPECTED_DONE')
        identity = self.outstanding[channel][0]
        context = self.contexts[identity]
        self._protocol(error is not None or context.sent == context.span.length, 'EARLY_DONE')
        self._protocol(len(self.events[channel]) < 2, 'EVENT_OVERFLOW')
        self.outstanding[channel].popleft()
        context.done_observed = True
        event = Event(identity, context.span.target, context.span.window_generation,
                      context.span.length if error is None else 0, error or 'OK')
        self.events[channel].append(event)
        if error is not None:
            self._fault('DMA_ERROR')
        self._log('DONE_PULSE', channel, identity, status=event.status)
        return event

    def poll_done(self, channel, *, ready):
        self._channel(channel)
        if not self.events[channel] or not ready:
            return None
        event = self.events[channel].popleft()
        context = self.contexts[event.identity]
        context.dma_done = event.status == 'OK'
        self._log('DMA_EVENT', channel, event.identity, status=event.status)
        return event

    def acknowledge(self, event):
        context = self._context(event.identity)
        span = context.span
        self._protocol(not context.ack and event.target == span.target
                       and event.window_generation == span.window_generation
                       and event.length == span.length and event.status == 'OK'
                       and context.sent == span.length, 'BAD_ACK')
        context.ack = True
        self._log('ACK', context.descriptor.channel, event.identity)

    def result(self, channel):
        self._channel(channel)
        if not self.order[channel]:
            return None
        context = self.contexts[self.order[channel][0]]
        span = context.span
        if context.failure:
            return Event(span.identity, span.target, span.window_generation, 0, context.failure)
        if context.local_done and context.dma_done and context.ack:
            return Event(span.identity, span.target, span.window_generation, span.length)
        return None

    def retire(self, channel):
        event = self.result(channel)
        require(event is not None, 'NOT_COMPLETE')
        require(event.status == 'OK', 'QUARANTINED')
        self.order[channel].popleft()
        del self.contexts[event.identity]
        self._log('RETIRE', channel, event.identity)
        return event

    def halt(self, reason='STOP'):
        self._fault(reason)

    def set_link(self, up):
        self.link_up = bool(up)
        if not up:
            self._fault('LINK_DOWN')

    def resources(self):
        return dict(contexts=[len(q) for q in self.order], payload=[len(q) for q in self.payload],
                    outstanding=[len(q) for q in self.outstanding], events=[len(q) for q in self.events],
                    halted=self.halted, quarantined=sum(c.failure is not None for c in self.contexts.values()))
