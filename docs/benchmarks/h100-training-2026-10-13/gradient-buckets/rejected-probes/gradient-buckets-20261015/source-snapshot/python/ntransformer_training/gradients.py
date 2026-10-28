"""Explicit first-order accumulation into persistent FP32 communication buckets.

The model graph produces reusable gradients; these buckets own independent
storage until reduction and the optimizer finish. Each update overwrites its
first microbatch and adds subsequent microbatches in their original order.
"""
import torch
import torch.distributed as dist


class GradientBuckets:
    def __init__(self, graph, microbatches, bucket_mb=25, process_group=None):
        if microbatches < 1 or not 1 <= bucket_mb <= 1024:
            raise ValueError('Positive microbatch count and 1..1024 MiB buckets required')
        self.graph = graph
        self.parameters = graph.parameters
        if any(p.dtype != torch.float32 or not p.is_contiguous() or not p.requires_grad
               for p in self.parameters):
            raise ValueError('Gradient buckets require contiguous trainable FP32 parameters')
        self.group = process_group
        self.world = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self.microbatches = microbatches
        self.device = graph.device
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self.buffers, self.views = [], []
        capacity = bucket_mb * 1024 * 1024 // 4
        pending, count = [], 0

        def reserve():
            buffer = torch.empty(count,device=self.device,dtype=torch.float32)
            offset = 0
            for parameter in pending:
                view = buffer[offset:offset+parameter.numel()].view_as(parameter)
                parameter.grad = view
                self.views.append(view)
                offset += parameter.numel()
            self.buffers.append(buffer)

        for parameter in self.parameters:
            if pending and count + parameter.numel() > capacity:
                reserve()
                pending, count = [], 0
            pending.append(parameter)
            count += parameter.numel()
        if pending:
            reserve()
        self.views = tuple(self.views)
        self.state = 'ready'
        self.index = self.updates = self.backward_calls = self.collectives = 0

    def _check_storage(self):
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError('Gradient buckets require their owner stream')
        if any(p.grad is not view for p,view in zip(self.parameters,self.views)):
            raise RuntimeError('Gradient bucket ownership changed; do not reset parameter.grad')
        if self.graph.parameter_addresses != tuple(p.data_ptr() for p in self.parameters):
            raise RuntimeError('Captured parameter storage changed')

    def begin_step(self):
        self._check_storage()
        if self.state not in ('ready','reduced') or self.graph.pending:
            raise RuntimeError('Previous gradient accumulation is incomplete')
        self.index = 0
        self.state = 'accumulating'

    def backward(self, scaled_loss):
        self._check_storage()
        if self.state != 'accumulating' or not self.graph.pending:
            raise RuntimeError('Begin an update and run a graph forward before backward')
        # autograd.grad stops at the parameter surface, avoiding AccumulateGrad
        # and DDP hooks. Returned tensors are borrowed only until the next replay.
        gradients = torch.autograd.grad(scaled_loss,self.parameters)
        if self.index == 0:
            torch._foreach_copy_(self.views,gradients)
        else:
            torch._foreach_add_(self.views,gradients)
        self.index += 1
        self.backward_calls += 1
        if self.index == self.microbatches:
            self.finish()

    def finish(self):
        self._check_storage()
        if self.state != 'accumulating' or self.index != self.microbatches or self.graph.pending:
            raise RuntimeError('All configured microbatches must finish before reduction')
        if self.world > 1:
            # Match DDP's pre-sum scaling. Never sum first then divide: that
            # changes overflow behavior and may change floating-point rounding.
            torch._foreach_div_(self.buffers,self.world)
            work = [dist.all_reduce(buffer,group=self.group,async_op=True) for buffer in self.buffers]
            for operation in work:
                operation.wait()
            self.collectives += len(work)
        self.state = 'reduced'
        self.updates += 1

    def statistics(self):
        self._check_storage()
        return dict(updates=self.updates,backward_calls=self.backward_calls,
                    collective_calls=self.collectives,buckets=len(self.buffers),
                    bucket_bytes=[x.numel()*x.element_size() for x in self.buffers],
                    reserved_bytes=sum(x.numel()*x.element_size() for x in self.buffers),
                    state=self.state,world_size=self.world,microbatches=self.microbatches,
                    accumulation='first copy; ordered foreach adds; divide before NCCL sum',
                    storage='independent persistent buckets; optimizer parameter.grad views')
