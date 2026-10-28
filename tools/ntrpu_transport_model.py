"""Bounded, event-driven NTRPU transport reference; not PCIe/AXI/RTL emulation.

Two nodes, one submission queue per direction and one leased window per node.
The caller explicitly schedules delivery, memory commit and remote confirmation.
No wall clock, threads, CUDA, network, or hardware access is used.
"""

from collections import deque
from dataclasses import dataclass, field


U64_MAX = (1 << 64) - 1
ALIGNMENT = 64
PROTOCOL_VERSION = 1


class TransportError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise TransportError(code)


def unsigned(value, positive=False):
    require(type(value) is int and (1 if positive else 0) <= value <= U64_MAX,
            "INVALID_U64")


def interval(offset, size, capacity):
    unsigned(offset)
    unsigned(size, positive=True)
    require(offset <= capacity and size <= capacity - offset, "OUT_OF_RANGE")
    require(offset % ALIGNMENT == 0 and size % ALIGNMENT == 0, "ALIGNMENT")


@dataclass(frozen=True)
class RegionHandle:
    session: int
    node: int
    slot: int
    generation: int


@dataclass(frozen=True)
class Window:
    region: RegionHandle
    generation: int
    offset: int
    size: int


@dataclass(frozen=True)
class TransferId:
    session: int
    source: int
    sequence: int


@dataclass(frozen=True)
class PacketId:
    transfer: TransferId
    index: int


@dataclass(frozen=True)
class Completion:
    transfer: TransferId
    status: str
    valid_bytes: int


@dataclass(frozen=True)
class Fence:
    session: int
    source: int
    watermark: int


@dataclass
class _Region:
    handle: RegionHandle
    base: int
    size: int
    permissions: str
    valid_blocks: set = field(default_factory=set)
    readers: set = field(default_factory=set)
    writers: set = field(default_factory=set)
    closing: bool = False


@dataclass
class _Transfer:
    identity: TransferId
    source: RegionHandle
    window: Window
    source_offset: int
    target_offset: int
    length: int
    issued: set = field(default_factory=set)
    committed: set = field(default_factory=set)
    result: Completion | None = None
    retired: bool = False


@dataclass
class _Packet:
    identity: PacketId
    payload: bytes
    received: bool = False


class TransportModel:
    """Conservative region leases and bounded queues for two logical endpoints.

    IDs increase strictly per source within a session. Validity is tracked at
    64-byte granularity; physical sparse memory survives logical recovery.
    All configuration arguments and opaque handles are model inputs, not ABI.
    """

    def __init__(self, capacities=(512 << 30, 512 << 30), *, queue_depth=4,
                 credits=2, fragment_bytes=64, max_transfer=1 << 20,
                 max_regions=16, max_blocks=16384, model_epoch=1):
        require(len(capacities) == 2, "TWO_NODES_REQUIRED")
        for capacity in capacities:
            unsigned(capacity, positive=True)
        for value in (queue_depth, credits, fragment_bytes, max_transfer,
                      max_regions, max_blocks, model_epoch):
            unsigned(value, positive=True)
        require(fragment_bytes % ALIGNMENT == 0 and
                max_transfer % ALIGNMENT == 0 and fragment_bytes <= max_transfer,
                "ALIGNMENT")
        self.capacities = tuple(capacities)
        self.queue_depth = queue_depth
        self.credit_limit = credits
        self.fragment_bytes = fragment_bytes
        self.max_transfer = max_transfer
        self.max_regions = max_regions
        self.max_blocks = max_blocks
        self.model_epoch = model_epoch
        self.session = 1
        self.ready = True
        self._regions = [{}, {}]
        self._memory = [{}, {}]
        self._generation = [0, 0]
        self._window_generation = [0, 0]
        self._windows = [None, None]
        self._window_users = [set(), set()]
        self._last_sequence = [0, 0]
        self._first_failure = [None, None]
        self._credits = [credits, credits]
        self._jobs = {}
        self._packets = {}
        self.events = deque(maxlen=4096)
        self._event_number = 0

    def _event(self, kind, **details):
        self._event_number += 1
        self.events.append(dict(event=self._event_number, session=self.session,
                                kind=kind, **details))

    def _node(self, node):
        require(type(node) is int and node in (0, 1), "INVALID_NODE")

    def _session(self, session):
        unsigned(session, positive=True)
        require(session == self.session, "STALE_SESSION")

    def _region(self, handle, *, closing_ok=False):
        require(isinstance(handle, RegionHandle), "INVALID_HANDLE")
        self._session(handle.session)
        self._node(handle.node)
        region = self._regions[handle.node].get(handle.slot)
        require(region is not None and region.handle == handle, "STALE_REGION")
        require(closing_ok or not region.closing, "REGION_CLOSING")
        return region

    def _job(self, identity, *, retired_ok=False):
        require(isinstance(identity, TransferId), "INVALID_ID")
        self._session(identity.session)
        job = self._jobs.get(identity)
        require(job is not None and (retired_ok or not job.retired), "UNKNOWN_TRANSFER")
        return job

    def _packet(self, identity):
        require(isinstance(identity, PacketId), "INVALID_PACKET")
        self._session(identity.transfer.session)
        packet = self._packets.get(identity)
        require(packet is not None, "UNKNOWN_PACKET")
        return packet

    @staticmethod
    def _blocks(offset, length):
        return range(offset // ALIGNMENT, (offset + length) // ALIGNMENT)

    def _read_memory(self, node, address, length):
        return b"".join(self._memory[node].get(block, bytes(ALIGNMENT))
                        for block in self._blocks(address, length))

    def _write_memory(self, node, address, payload):
        blocks = self._blocks(address, len(payload))
        new_blocks = sum(block not in self._memory[node] for block in blocks)
        require(len(self._memory[node]) + new_blocks <= self.max_blocks, "MEMORY_BUDGET")
        for index, block in enumerate(blocks):
            self._memory[node][block] = payload[index * ALIGNMENT:(index + 1) * ALIGNMENT]

    def _conflicts(self, region, offset, length, *, write, published_ok=False):
        for identity in region.writers:
            job = self._job(identity, retired_ok=True)
            if published_ok and job.result is not None and job.result.status == "SUCCESS":
                continue
            if offset < job.target_offset + job.length and job.target_offset < offset + length:
                return True
        if write:
            for identity in region.readers:
                job = self._job(identity, retired_ok=True)
                if offset < job.source_offset + job.length and job.source_offset < offset + length:
                    return True
        return False

    def discover(self, version=PROTOCOL_VERSION):
        require(version == PROTOCOL_VERSION, "VERSION_MISMATCH")
        return dict(version=PROTOCOL_VERSION, session=self.session, ready=self.ready,
                    capacities=self.capacities, alignment=ALIGNMENT,
                    max_transfer=self.max_transfer, fragment_bytes=self.fragment_bytes,
                    queue_depth=self.queue_depth, credits_per_receiver=self.credit_limit)

    def register(self, node, base, size, permissions="rw"):
        require(self.ready, "LINK_DOWN")
        self._node(node)
        interval(base, size, self.capacities[node])
        require(permissions in ("r", "w", "rw"), "PERMISSIONS")
        regions = self._regions[node]
        require(len(regions) < self.max_regions, "REGION_LIMIT")
        require(all(base + size <= r.base or r.base + r.size <= base
                    for r in regions.values()), "OVERLAP")
        require(self._generation[node] < U64_MAX, "GENERATION_EXHAUSTED")
        slot = next(i for i in range(self.max_regions) if i not in regions)
        self._generation[node] += 1
        handle = RegionHandle(self.session, node, slot, self._generation[node])
        regions[slot] = _Region(handle, base, size, permissions)
        self._event("REGISTER", node=node, slot=slot, base=base, size=size)
        return handle

    def unregister(self, handle):
        region = self._region(handle, closing_ok=True)
        region.closing = True
        require(not region.readers and not region.writers, "REGION_BUSY")
        window = self._windows[handle.node]
        if window is not None and window.region == handle:
            require(not self._window_users[handle.node], "WINDOW_BUSY")
            self._windows[handle.node] = None
        del self._regions[handle.node][handle.slot]
        self._event("UNREGISTER", node=handle.node, slot=handle.slot)

    def host_write(self, handle, offset, payload):
        region = self._region(handle)
        require(self.ready, "LINK_DOWN")
        require(isinstance(payload, bytes), "INVALID_PAYLOAD")
        interval(offset, len(payload), region.size)
        require(len(payload) <= self.max_transfer, "TRANSFER_LIMIT")
        require("w" in region.permissions, "PERMISSIONS")
        require(not self._conflicts(region, offset, len(payload), write=True), "REGION_BUSY")
        self._write_memory(handle.node, region.base + offset, payload)
        region.valid_blocks.update(self._blocks(offset, len(payload)))

    def read(self, handle, offset, length):
        region = self._region(handle)
        interval(offset, length, region.size)
        require(length <= self.max_transfer, "TRANSFER_LIMIT")
        require("r" in region.permissions, "PERMISSIONS")
        require(not self._conflicts(region, offset, length, write=False, published_ok=True),
                "REGION_BUSY")
        require(all(block in region.valid_blocks for block in self._blocks(offset, length)),
                "INVALID_DATA")
        return self._read_memory(handle.node, region.base + offset, length)

    def debug_read_raw(self, handle, offset, length):
        """Test-only inspection of physical bytes, including invalid/partial data."""
        region = self._region(handle, closing_ok=True)
        interval(offset, length, region.size)
        require(length <= self.max_transfer, "TRANSFER_LIMIT")
        return self._read_memory(handle.node, region.base + offset, length)

    def bind_window(self, handle, offset=0, size=None):
        require(self.ready, "LINK_DOWN")
        region = self._region(handle)
        if size is None:
            size = min(region.size, self.max_transfer)
        interval(offset, size, region.size)
        require(size <= self.max_transfer, "WINDOW_LIMIT")
        require(not self._window_users[handle.node], "WINDOW_BUSY")
        require(self._window_generation[handle.node] < U64_MAX, "GENERATION_EXHAUSTED")
        self._window_generation[handle.node] += 1
        window = Window(handle, self._window_generation[handle.node], offset, size)
        self._windows[handle.node] = window
        self._event("BIND", node=handle.node, generation=window.generation)
        return window

    def submit(self, session, sequence, source, window, length, *, source_offset=0,
               target_offset=0, model_epoch=1):
        self._session(session)
        require(self.ready, "LINK_DOWN")
        unsigned(sequence, positive=True)
        unsigned(model_epoch, positive=True)
        require(model_epoch == self.model_epoch, "MODEL_EPOCH")
        origin = self._region(source)
        require(isinstance(window, Window), "INVALID_WINDOW")
        target = self._region(window.region)
        require(source.node != target.handle.node, "REMOTE_ONLY")
        require(self._windows[target.handle.node] == window, "STALE_WINDOW")
        require(sequence > self._last_sequence[source.node], "REPLAY")
        interval(source_offset, length, origin.size)
        interval(target_offset, length, target.size)
        require(length <= self.max_transfer, "TRANSFER_LIMIT")
        require(target_offset >= window.offset and
                length <= window.size - (target_offset - window.offset), "WINDOW_RANGE")
        require("r" in origin.permissions and "w" in target.permissions, "PERMISSIONS")
        require(not self._conflicts(origin, source_offset, length, write=False) and
                not self._conflicts(target, target_offset, length, write=True), "REGION_BUSY")
        require(all(block in origin.valid_blocks
                    for block in self._blocks(source_offset, length)), "INVALID_DATA")
        require(sum(not j.retired and j.identity.source == source.node
                    for j in self._jobs.values()) < self.queue_depth, "QUEUE_FULL")
        identity = TransferId(session, source.node, sequence)
        self._jobs[identity] = _Transfer(identity, source, window, source_offset,
                                         target_offset, length)
        self._last_sequence[source.node] = sequence
        origin.readers.add(identity)
        target.writers.add(identity)
        target.valid_blocks.difference_update(self._blocks(target_offset, length))
        self._window_users[target.handle.node].add(identity)
        self._event("SUBMIT", source=source.node, sequence=sequence, length=length)
        return identity

    def send(self, identity, index):
        require(self.ready, "LINK_DOWN")
        job = self._job(identity)
        require(job.result is None, "ALREADY_COMPLETE")
        count = (job.length + self.fragment_bytes - 1) // self.fragment_bytes
        require(type(index) is int and 0 <= index < count, "FRAGMENT_RANGE")
        require(index not in job.issued, "DUPLICATE_FRAGMENT")
        node = job.window.region.node
        require(self._credits[node] > 0, "NO_CREDIT")
        region = self._region(job.source, closing_ok=True)
        offset = index * self.fragment_bytes
        payload = self._read_memory(job.source.node,
                                    region.base + job.source_offset + offset,
                                    min(self.fragment_bytes, job.length - offset))
        packet_id = PacketId(identity, index)
        self._packets[packet_id] = _Packet(packet_id, payload)
        job.issued.add(index)
        self._credits[node] -= 1
        self._event("SEND", source=identity.source, sequence=identity.sequence, fragment=index)
        return packet_id

    def receive(self, packet_id, *, integrity_ok=True):
        packet = self._packet(packet_id)
        require(not packet.received, "DUPLICATE_RECEIVE")
        if not integrity_ok:
            self.fail("INTEGRITY_ERROR")
            raise TransportError("INTEGRITY_ERROR")
        packet.received = True
        self._event("RECEIVE", source=packet_id.transfer.source,
                    sequence=packet_id.transfer.sequence, fragment=packet_id.index)

    def commit(self, packet_id):
        packet = self._packet(packet_id)
        require(packet.received, "NOT_RECEIVED")
        job = self._job(packet_id.transfer, retired_ok=True)
        region = self._region(job.window.region, closing_ok=True)
        address = region.base + job.target_offset + packet_id.index * self.fragment_bytes
        try:
            self._write_memory(region.handle.node, address, packet.payload)
        except TransportError:
            self.fail("MEMORY_ERROR")
            raise
        self._complete_packet(packet_id)

    def _complete_packet(self, packet_id):
        """Adapter hook: caller must have written ALL bytes at the final destination."""
        packet = self._packet(packet_id)
        require(packet.received, "NOT_RECEIVED")
        job = self._job(packet_id.transfer, retired_ok=True)
        region = self._region(job.window.region, closing_ok=True)
        job.committed.add(packet_id.index)
        del self._packets[packet_id]
        self._credits[region.handle.node] += 1
        self._event("COMMIT", source=packet_id.transfer.source,
                    sequence=packet_id.transfer.sequence, fragment=packet_id.index,
                    bytes=len(packet.payload), late=job.result is not None)

    def confirm_remote(self, identity):
        require(self.ready, "LINK_DOWN")
        job = self._job(identity)
        require(job.result is None, "ALREADY_COMPLETE")
        count = (job.length + self.fragment_bytes - 1) // self.fragment_bytes
        require(len(job.committed) == count, "NOT_VISIBLE")
        region = self._region(job.window.region, closing_ok=True)
        region.valid_blocks.update(self._blocks(job.target_offset, job.length))
        job.result = Completion(identity, "SUCCESS", job.length)
        self._event("SUCCESS", source=identity.source, sequence=identity.sequence)
        return job.result

    def completion(self, identity):
        result = self._job(identity).result
        require(result is not None, "NOT_COMPLETE")
        return result

    def retire(self, identity):
        job = self._job(identity)
        result = self.completion(identity)
        job.retired = True
        if result.status == "SUCCESS":
            self._region(job.source, closing_ok=True).readers.remove(identity)
            self._region(job.window.region, closing_ok=True).writers.remove(identity)
            self._window_users[job.window.region.node].remove(identity)
            del self._jobs[identity]
        # Failed jobs keep leases: posted traffic can arrive even after RETIRE.
        self._event("RETIRE", source=identity.source, sequence=identity.sequence,
                    status=result.status)
        return result

    def fence(self, source):
        self._node(source)
        return Fence(self.session, source, self._last_sequence[source])

    def fence_status(self, fence):
        require(isinstance(fence, Fence), "INVALID_FENCE")
        self._session(fence.session)
        self._node(fence.source)
        unsigned(fence.watermark)
        require(fence.watermark <= self._last_sequence[fence.source], "INVALID_FENCE")
        failed = self._first_failure[fence.source]
        if failed is not None and failed <= fence.watermark:
            return "FAILED"
        if any(j.identity.source == fence.source and j.identity.sequence <= fence.watermark
               and j.result is None for j in self._jobs.values()):
            return "PENDING"
        return "SUCCESS"

    def fail(self, reason="LINK_DOWN"):
        require(reason in ("LINK_DOWN", "TIMEOUT", "INTEGRITY_ERROR", "MEMORY_ERROR"),
                "INVALID_FAILURE")
        self.ready = False
        for job in self._jobs.values():
            if job.result is None:
                job.result = Completion(job.identity, reason, 0)
                source = job.identity.source
                previous = self._first_failure[source]
                self._first_failure[source] = min(previous or job.identity.sequence,
                                                   job.identity.sequence)
        self._event("FAIL", reason=reason, in_flight=len(self._packets))

    def revoke_inflight(self):
        """Model an acknowledged transport revocation, not a hardware procedure.

        Its production counterpart must prove that old reads/writes can no longer
        arrive. Recovery alone must never silently supply this acknowledgement.
        """
        require(not self.ready, "LINK_STILL_READY")
        count = len(self._packets)
        self._packets.clear()
        self._credits = [self.credit_limit, self.credit_limit]
        self._event("REVOKE_ACK", packets=count)

    def recover(self):
        require(not self.ready, "LINK_STILL_READY")
        require(not self._packets, "INFLIGHT_NOT_DRAINED")
        require(all(j.retired for j in self._jobs.values()), "COMPLETIONS_PENDING")
        require(self.session < U64_MAX, "SESSION_EXHAUSTED")
        self.session += 1
        self._regions = [{}, {}]
        self._windows = [None, None]
        self._window_users = [set(), set()]
        self._jobs.clear()
        self._last_sequence = [0, 0]
        self._first_failure = [None, None]
        self._credits = [self.credit_limit, self.credit_limit]
        self.ready = True
        self._event("RECOVER")

    def resources(self):
        return dict(credits=tuple(self._credits), packets=len(self._packets),
                    queue_used=tuple(sum(not j.retired and j.identity.source == node
                                         for j in self._jobs.values()) for node in (0, 1)),
                    materialized_bytes=tuple(len(m) * ALIGNMENT for m in self._memory))
