"""Behavioral posted-write/AXI adapter for the NTRPU protocol reference.

Event-level only: no TLP encoder, RTL, timing, PCIe IP or physical memory.
Conservative profile: 512-bit full writes, AxCACHE=0000, fixed AXI ID and
one active burst per receiver. The environment exposes final-memory commits.
"""

from collections import deque
from dataclasses import dataclass

from ntrpu_transport_model import (
    ALIGNMENT, U64_MAX, PacketId, TransportError, TransportModel, require, unsigned,
)


@dataclass(frozen=True)
class WriteSpan:
    pcie_address: int
    axi_address: int
    offset: int
    length: int


def plan_writes(pcie_address, axi_address, length, *, mps=256, max_axi_beats=256):
    """Split full-width writes at BOTH address-space page boundaries and limits."""
    for value in (pcie_address, axi_address):
        unsigned(value)
    unsigned(length, positive=True)
    require(length <= 1 << 20, "PLAN_LIMIT")
    require(type(mps) is int and mps in (128, 256, 512, 1024, 2048, 4096), "INVALID_MPS")
    require(type(max_axi_beats) is int and 1 <= max_axi_beats <= 256, "AXI_BEAT_LIMIT")
    require(all(value % ALIGNMENT == 0 for value in (pcie_address, axi_address, length)),
            "ALIGNMENT")
    require(length <= U64_MAX - pcie_address + 1 and
            length <= U64_MAX - axi_address + 1, "ADDRESS_OVERFLOW")
    offset = 0
    spans = []
    while offset < length:
        pcie, axi = pcie_address + offset, axi_address + offset
        size = min(length - offset, mps, max_axi_beats * ALIGNMENT,
                   4096 - pcie % 4096, 4096 - axi % 4096)
        spans.append(WriteSpan(pcie, axi, offset, size))
        offset += size
    return tuple(spans)


@dataclass(frozen=True)
class BurstId:
    packet: PacketId
    index: int


@dataclass
class _Burst:
    identity: BurstId
    span: WriteSpan
    payload: bytes
    local_ack: bool = False
    delivered: bool = False
    launched: bool = False
    aw: bool = False
    w_beats: int = 0
    visible_beats: int = 0
    bresp: str | None = None
    dropped: bool = False


class PcieAxiModel(TransportModel):
    """Two endpoints with FIFO PCIe arrivals and a serial AXI write path each.

    API events represent successful handshakes, not signal levels or cycles.
    AW/W can arrive in either order; no second burst launches until B retires.
    Destination memory visibility is an explicit environment event. Responses
    violating the selected non-bufferable profile are rejected by this oracle.
    """

    def __init__(self, *, bar_bases=(0x100000000, 0x200000000), mps=256,
                 max_axi_beats=256, bridge_depth=4, **kwargs):
        super().__init__(**kwargs)
        require(len(bar_bases) == 2, "TWO_BARS_REQUIRED")
        unsigned(bridge_depth, positive=True)
        for base in bar_bases:
            # The same validation applies even before the first transfer.
            plan_writes(base, 0, ALIGNMENT, mps=mps, max_axi_beats=max_axi_beats)
        self.bar_bases = tuple(bar_bases)
        self.mps = mps
        self.max_axi_beats = max_axi_beats
        self.bridge_depth = bridge_depth
        self._bursts = {}
        self._packet_bursts = {}
        self._posted = [deque(), deque()]
        self._received = [deque(), deque()]
        self._active = [None, None]
        self._source_stopped = [False, False]
        self._ingress_isolated = [False, False]
        self._revoked = False

    def _burst(self, identity):
        require(isinstance(identity, BurstId), "INVALID_BURST")
        self._session(identity.packet.transfer.session)
        burst = self._bursts.get(identity)
        require(burst is not None, "UNKNOWN_BURST")
        return burst

    @staticmethod
    def _receiver(identity):
        return 1 - identity.packet.transfer.source

    def _active_burst(self, identity):
        burst = self._burst(identity)
        require(self._active[self._receiver(identity)] == identity and not burst.dropped,
                "NOT_ACTIVE")
        return burst

    def send(self, *args, **kwargs):
        raise TransportError("USE_POST")

    def receive(self, *args, **kwargs):
        raise TransportError("USE_DELIVER")

    def commit(self, *args, **kwargs):
        raise TransportError("USE_MEMORY_STEP")

    def post(self, identity, fragment_index):
        job = self._job(identity)
        require(type(fragment_index) is int and fragment_index >= 0, "FRAGMENT_RANGE")
        fragment_offset = fragment_index * self.fragment_bytes
        require(fragment_offset < job.length, "FRAGMENT_RANGE")
        region = self._region(job.window.region, closing_ok=True)
        node = region.handle.node
        offset = job.target_offset + fragment_offset
        pcie = self.bar_bases[node] + offset - job.window.offset
        axi = region.base + offset
        length = min(self.fragment_bytes, job.length - fragment_offset)
        spans = plan_writes(pcie, axi, length, mps=self.mps, max_axi_beats=self.max_axi_beats)
        packet_id = super().send(identity, fragment_index)
        payload = self._packet(packet_id).payload
        identities = tuple(BurstId(packet_id, i) for i in range(len(spans)))
        self._packet_bursts[packet_id] = identities
        for burst_id, span in zip(identities, spans):
            self._bursts[burst_id] = _Burst(burst_id, span,
                                           payload[span.offset:span.offset + span.length])
            self._posted[node].append(burst_id)
        self._event("POST", source=identity.source, sequence=identity.sequence,
                    fragment=fragment_index, bursts=len(identities))
        return identities

    def describe(self, identity):
        burst = self._burst(identity)
        span = burst.span
        return dict(pcie_address=span.pcie_address, axi_address=span.axi_address,
                    length=span.length, awlen=span.length // ALIGNMENT - 1,
                    awsize=6, awburst="INCR", awcache=0, awid=0,
                    wstrb=(1 << ALIGNMENT) - 1)

    def acknowledge_posted(self, identity):
        """Source-side bridge acknowledgement; does NOT confirm remote memory."""
        burst = self._burst(identity)
        require(not burst.local_ack, "DUPLICATE_LOCAL_ACK")
        burst.local_ack = True
        self._event("LOCAL_POST_ACK", source=identity.packet.transfer.source,
                    sequence=identity.packet.transfer.sequence, burst=identity.index)

    def deliver(self, identity):
        burst = self._burst(identity)
        node = self._receiver(identity)
        require(not self._ingress_isolated[node], "INGRESS_ISOLATED")
        require(not burst.delivered and not burst.dropped, "DUPLICATE_DELIVERY")
        require(self._posted[node] and self._posted[node][0] == identity, "PCIE_ORDER")
        require(len(self._received[node]) + (self._active[node] is not None) < self.bridge_depth,
                "BRIDGE_FULL")
        if not self._packet(identity.packet).received:
            super().receive(identity.packet)
        burst.delivered = True
        self._posted[node].popleft()
        self._received[node].append(identity)
        self._event("INGRESS", receiver=node, sequence=identity.packet.transfer.sequence,
                    burst=identity.index)

    def launch(self, identity):
        burst = self._burst(identity)
        node = self._receiver(identity)
        require(self._active[node] is None, "AXI_BUSY")
        require(not self._ingress_isolated[node] and not burst.dropped, "INGRESS_ISOLATED")
        require(self._received[node] and self._received[node][0] == identity, "AXI_ORDER")
        burst.launched = True
        self._received[node].popleft()
        self._active[node] = identity
        self._event("AXI_LAUNCH", receiver=node, address=burst.span.axi_address,
                    length=burst.span.length)

    def accept_aw(self, identity):
        burst = self._active_burst(identity)
        require(not burst.aw, "DUPLICATE_AW")
        burst.aw = True
        self._event("AW", receiver=self._receiver(identity), address=burst.span.axi_address)

    def accept_w(self, identity, *, last=None):
        burst = self._active_burst(identity)
        beats = burst.span.length // ALIGNMENT
        require(burst.w_beats < beats, "EXTRA_W")
        expected_last = burst.w_beats == beats - 1
        require(last is None or type(last) is bool and last == expected_last, "WLAST")
        burst.w_beats += 1
        self._event("W", receiver=self._receiver(identity), beat=burst.w_beats,
                    last=expected_last)

    def memory_step(self, identity):
        """Environment event: a 64-byte beat reaches the final memory destination."""
        burst = self._active_burst(identity)
        require(burst.aw, "AW_PENDING")
        require(burst.visible_beats < burst.w_beats, "W_PENDING")
        offset = burst.visible_beats * ALIGNMENT
        try:
            self._write_memory(self._receiver(identity), burst.span.axi_address + offset,
                               burst.payload[offset:offset + ALIGNMENT])
        except TransportError:
            self.fail("MEMORY_ERROR")
            raise
        burst.visible_beats += 1
        self._event("MEMORY_VISIBLE", receiver=self._receiver(identity),
                    address=burst.span.axi_address + offset, bytes=ALIGNMENT)
        members = (self._bursts[key] for key in self._packet_bursts[identity.packet])
        if all(b.visible_beats * ALIGNMENT == b.span.length for b in members):
            self._complete_packet(identity.packet)

    def accept_b(self, identity, response="OKAY"):
        burst = self._active_burst(identity)
        require(response in ("OKAY", "SLVERR", "DECERR"), "INVALID_BRESP")
        require(burst.aw and burst.w_beats * ALIGNMENT == burst.span.length, "AW_W_PENDING")
        if response == "OKAY":
            require(burst.visible_beats * ALIGNMENT == burst.span.length, "EARLY_BRESP")
        else:
            # A final error may leave earlier beats written; remaining beats stop here.
            self.fail("MEMORY_ERROR")
        burst.bresp = response
        self._active[self._receiver(identity)] = None
        self._event("B", receiver=self._receiver(identity), response=response,
                    address=burst.span.axi_address)

    def confirm_remote(self, identity):
        self._job(identity)
        members = [b for key, b in self._bursts.items() if key.packet.transfer == identity]
        require(members and all(b.bresp == "OKAY" for b in members), "RESPONSES_PENDING")
        return super().confirm_remote(identity)

    def retire(self, identity):
        result = super().retire(identity)
        if result.status == "SUCCESS":
            for packet in [key for key in self._packet_bursts if key.transfer == identity]:
                for burst in self._packet_bursts.pop(packet):
                    del self._bursts[burst]
        return result

    def source_stop_ack(self, source):
        """Environment guarantees that this source can issue no further old DMA."""
        require(not self.ready, "LINK_STILL_READY")
        self._node(source)
        self._source_stopped[source] = True
        self._event("SOURCE_STOP_ACK", source=source)

    def isolate_ingress_ack(self, receiver):
        """Gate old arrivals and discard only transactions never launched on AXI."""
        require(not self.ready, "LINK_STILL_READY")
        self._node(receiver)
        self._ingress_isolated[receiver] = True
        discarded = list(self._posted[receiver]) + list(self._received[receiver])
        for identity in discarded:
            self._bursts[identity].dropped = True
        self._posted[receiver].clear()
        self._received[receiver].clear()
        # A launched transaction is retained even if AW/W have not handshaken yet.
        self._event("INGRESS_ISOLATED", receiver=receiver, discarded=len(discarded))

    def revoke_inflight(self):
        require(not self.ready, "LINK_STILL_READY")
        require(all(self._source_stopped), "SOURCE_STOP_PENDING")
        require(all(self._ingress_isolated), "INGRESS_ISOLATION_PENDING")
        require(all(active is None for active in self._active), "AXI_DRAIN_PENDING")
        super().revoke_inflight()
        self._revoked = True

    def recover(self):
        if not self._revoked:
            self.revoke_inflight()
        super().recover()
        self._bursts.clear()
        self._packet_bursts.clear()
        self._posted = [deque(), deque()]
        self._received = [deque(), deque()]
        self._active = [None, None]
        self._source_stopped = [False, False]
        self._ingress_isolated = [False, False]
        self._revoked = False

    def resources(self):
        result = super().resources()
        result.update(posted=tuple(map(len, self._posted)),
                      ingress=tuple(map(len, self._received)),
                      axi_active=tuple(item is not None for item in self._active),
                      source_stopped=tuple(self._source_stopped),
                      ingress_isolated=tuple(self._ingress_isolated))
        return result
