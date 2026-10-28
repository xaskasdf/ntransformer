"""Pre-reserved peer activation blocks and a two-buffer backward prefetch pipeline.

One sequential forward/backward microbatch at a time. All parameters stay on the
compute GPU. Peer storage holds checkpoint inputs, not stale copies of weights.
"""
import math

import torch
from torch.autograd.function import once_differentiable

from .ops import checked, library


class PeerActivationPool:
    def __init__(self, compute, peer, layers, shape, reserve_bytes=2 * 1024**3):
        if compute == peer or layers < 1 or min(shape) < 1:
            raise ValueError('Peer pool requires distinct GPUs, positive layers and shape')
        if not torch.cuda.can_device_access_peer(compute, peer):
            raise RuntimeError('Native CUDA P2P is required for this experiment')
        self.compute, self.peer = compute, peer
        self.shape, self.layers = tuple(shape), layers
        self.block_bytes = math.prod(shape) * 4
        with torch.cuda.device(compute):
            checked(library().nt_train_enable_peer(peer))
            self.write_stream = torch.cuda.Stream(device=compute)
            self.read_stream = torch.cuda.Stream(device=compute)
            self.produced = [torch.cuda.Event() for _ in range(layers)]
            self.saved = [torch.cuda.Event() for _ in range(layers)]
            self.ready = [torch.cuda.Event() for _ in range(layers)]
            self.consumed = [torch.cuda.Event() for _ in range(2)]
            self.staging = [torch.empty(shape, dtype=torch.float32, device=compute) for _ in range(2)]
            # New allocations can reuse storage with pending work on their creation stream.
            self.read_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.device(peer):
            checked(library().nt_train_enable_peer(compute))
            free, _ = torch.cuda.mem_get_info()
            if layers * self.block_bytes + reserve_bytes > free:
                raise RuntimeError('Insufficient peer VRAM for the requested speculative reservation')
            self.storage = [torch.empty(shape, dtype=torch.float32, device=peer) for _ in range(layers)]
            peer_allocated = torch.cuda.Event()
            peer_allocated.record()
        self.write_stream.wait_event(peer_allocated)
        self.state = ['free'] * layers
        self.owner = [None, None]
        self.consumed_recorded = [False, False]
        self.next_save = 0
        self.generation = 0
        self.copies_out = self.copies_in = self.prefetches = self.microbatches = 0

    def begin(self):
        if any(state != 'free' for state in self.state) or any(owner is not None for owner in self.owner):
            raise RuntimeError('Previous microbatch has live peer activations')
        self.next_save = 0
        self.generation += 1
        self.microbatches += 1

    def save(self, layer, x):
        if (layer != self.next_save or self.state[layer] != 'free' or tuple(x.shape) != self.shape
                or x.dtype != torch.float32 or x.device.index != self.compute or not x.is_contiguous()):
            raise ValueError('Peer checkpoint input violates the reservation or sequential schedule')
        stream = torch.cuda.current_stream(self.compute)
        self.produced[layer].record(stream)
        self.write_stream.wait_event(self.produced[layer])
        # Reusing a peer block in a later microbatch must wait for its previous read.
        if self.generation > 1:
            self.write_stream.wait_event(self.ready[layer])
        checked(library().nt_train_peer_copy(self.storage[layer].data_ptr(), self.peer,
                x.data_ptr(), self.compute, self.block_bytes, self.write_stream.cuda_stream))
        self.saved[layer].record(self.write_stream)
        x.record_stream(self.write_stream)
        self.state[layer] = 'saved'
        self.next_save += 1
        self.copies_out += 1

    def _prefetch(self, layer):
        if self.state[layer] == 'ready':
            return
        if self.state[layer] != 'saved':
            raise RuntimeError('Cannot prefetch an activation which has not been saved')
        slot = layer % 2
        if self.owner[slot] is not None:
            raise RuntimeError('Attempted to overwrite a live speculative staging block')
        if self.consumed_recorded[slot]:
            self.read_stream.wait_event(self.consumed[slot])
        self.read_stream.wait_event(self.saved[layer])
        checked(library().nt_train_peer_copy(self.staging[slot].data_ptr(), self.compute,
                self.storage[layer].data_ptr(), self.peer, self.block_bytes, self.read_stream.cuda_stream))
        self.ready[layer].record(self.read_stream)
        self.owner[slot] = layer
        self.state[layer] = 'ready'
        self.copies_in += 1

    def acquire(self, layer, generation):
        if generation != self.generation:
            raise RuntimeError('Backward attempted to reuse an old peer activation generation')
        self._prefetch(layer)
        torch.cuda.current_stream(self.compute).wait_event(self.ready[layer])
        # Layer i-1 is staged while recomputation/backward for i uses the other buffer.
        if layer > 0:
            self._prefetch(layer - 1)
            self.prefetches += 1
        return self.staging[layer % 2]

    def release(self, layer):
        slot = layer % 2
        if self.owner[slot] != layer or self.state[layer] != 'ready':
            raise RuntimeError('Invalid peer activation release')
        self.consumed[slot].record(torch.cuda.current_stream(self.compute))
        self.consumed_recorded[slot] = True
        self.owner[slot] = None
        self.state[layer] = 'free'

    def finish(self):
        if self.next_save != self.layers or any(state != 'free' for state in self.state):
            raise RuntimeError('Incomplete activation forward/backward schedule')

    def synchronize(self):
        self.write_stream.synchronize()
        self.read_stream.synchronize()
        for event, recorded in zip(self.consumed, self.consumed_recorded):
            if recorded:
                event.synchronize()

    def statistics(self):
        return dict(peer_device=self.peer, peer_blocks=self.layers, block_bytes=self.block_bytes,
                    reserved_peer_bytes=self.layers * self.block_bytes,
                    reserved_staging_bytes=2 * self.block_bytes,
                    d2d_writes=self.copies_out, d2d_reads=self.copies_in,
                    transferred_bytes=(self.copies_in + self.copies_out) * self.block_bytes,
                    lookahead_prefetches=self.prefetches, microbatches=self.microbatches)


class _BlockCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, freqs, pool, layer, *parameters):
        ctx.module, ctx.pool, ctx.layer = module, pool, layer
        ctx.autocast = torch.is_autocast_enabled('cuda')
        ctx.autocast_dtype = torch.get_autocast_dtype('cuda')
        if pool is None:
            ctx.save_for_backward(x, freqs, *parameters)
        else:
            pool.save(layer, x)
            ctx.generation = pool.generation
            ctx.save_for_backward(freqs, *parameters)
        # Dropout and configuration variants are rejected by the integration layer.
        return module(x, freqs, mask=None, cache=None, v_first=None)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.pool is None:
            saved_x, freqs, *parameters = ctx.saved_tensors
        else:
            freqs, *parameters = ctx.saved_tensors
            saved_x = ctx.pool.acquire(ctx.layer, ctx.generation)
        with torch.enable_grad(), torch.autocast('cuda', enabled=ctx.autocast, dtype=ctx.autocast_dtype):
            x = saved_x.detach().requires_grad_(True)
            y = ctx.module(x, freqs, mask=None, cache=None, v_first=None)[0]
        # Return parameter gradients through this Function's outer graph. This
        # lets DDP accumulate/reduce them once, instead of mutating .grad here.
        grads = torch.autograd.grad(y, (x, *parameters), dy, create_graph=False)
        if ctx.pool is not None:
            ctx.pool.release(ctx.layer)
        return (grads[0], None, None, None, None, *grads[1:])


def checkpoint_block(x, module, freqs, pool, layer):
    return _BlockCheckpoint.apply(x, module, freqs, pool, layer, *module.parameters())
