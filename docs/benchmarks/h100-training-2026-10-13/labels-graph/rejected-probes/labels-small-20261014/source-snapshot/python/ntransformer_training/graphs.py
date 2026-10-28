"""Opt-in fixed-shape forward/backward replay with graph-owned CUDA storage.

DDP and AdamW stay outside capture. Only one training forward may be live;
the scalar loss is cloned so deferred logging survives subsequent replays.
"""
from types import MethodType

import torch


class _LossCallable(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.original_forward = model.forward

    def forward(self, tokens, targets):
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            return self.original_forward(tokens, targets, targets_shifted=True, loss_only=True)[1]


class TrainingGraph:
    def __init__(self, model, tokens, targets):
        if not model.training or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Capture must precede training and DDP construction')
        if getattr(model, '_nt_peer_pool', None) is not None or hasattr(model, '_nt_original_run_layers'):
            raise ValueError('Whole-model graph replay does not capture the checkpoint/P2P host schedule')
        if getattr(model, '_nt_loss_mode', None) != 'fused' or not model._nt_fused_labels:
            raise ValueError('Training graph requires fused loss and fused labels')
        if tokens.shape != targets.shape or tokens.dtype != torch.int64 or targets.dtype != torch.int64:
            raise ValueError('Capture requires equal integer token/target batches')
        self.tokens, self.targets = tokens.clone(), targets.clone()
        self.original_forward = model.forward
        self.parameters = tuple(model.parameters())
        self.parameter_addresses = tuple(p.data_ptr() for p in self.parameters)
        self.pending = False
        self.forward_replays = self.backward_replays = 0
        self.device = tokens.device
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        with torch.random.fork_rng(devices=[self.device.index]):
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                self.callable = torch.cuda.make_graphed_callables(
                    _LossCallable(model), (self.tokens, self.targets), num_warmup_iters=3)
        assert all(p.grad is None for p in self.parameters)

    def stage(self, tokens, targets):
        if self.pending:
            raise RuntimeError('Training graph buffers still belong to a live forward')
        if (tokens.shape != self.tokens.shape or targets.shape != self.targets.shape
                or tokens.dtype != self.tokens.dtype or targets.dtype != self.targets.dtype):
            raise ValueError('Training graph input metadata changed; recapture explicitly')
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError('Training graph input staging requires its owner stream')
        self.tokens.copy_(tokens)
        self.targets.copy_(targets)
        return self.tokens, self.targets

    def run(self, tokens, targets):
        if self.pending:
            raise RuntimeError('Complete backward before the next training graph forward')
        if (tokens.shape != self.tokens.shape or targets.shape != self.targets.shape
                or tokens.dtype != self.tokens.dtype or targets.dtype != self.targets.dtype
                or tokens.device != self.device or targets.device != self.device):
            raise ValueError('Training graph input metadata changed; recapture explicitly')
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError('Training graph replay requires its owner stream')
        self.pending = True
        try:
            loss = self.callable(tokens, targets).clone()
            loss.register_hook(self._backward_started)
        except Exception:
            self.pending = False
            raise
        self.forward_replays += 1
        return loss

    def _backward_started(self, gradient):
        self.pending = False
        self.backward_replays += 1
        return gradient

    def statistics(self):
        return dict(forward_replays=self.forward_replays, backward_replays=self.backward_replays,
                    input_reserved_bytes=sum(x.numel()*x.element_size() for x in (self.tokens,self.targets)),
                    input_addresses=[self.tokens.data_ptr(), self.targets.data_ptr()],
                    parameter_addresses_unchanged=self.parameter_addresses == tuple(p.data_ptr() for p in self.parameters),
                    captured='model forward and backward', collective_optimizer='eager',
                    live_forward=self.pending)


def _graph_forward(self, tokens, targets=None, target_mask=None, n_predict_override=None,
                   label_smoothing=0., unlikelihood_alpha=0., entropy_reg_beta=0.,
                   targets_shifted=False, *, loss_only=False):
    runner = self._nt_training_graph
    if (self.training and torch.is_grad_enabled() and targets is not None and loss_only
            and targets_shifted and target_mask is None and n_predict_override in (None, 1)
            and label_smoothing == unlikelihood_alpha == entropy_reg_beta == 0.):
        if not torch.is_autocast_enabled('cuda') or torch.get_autocast_dtype('cuda') != torch.bfloat16:
            raise ValueError('Training graph requires its captured BF16 autocast policy')
        return None, runner.run(tokens, targets)
    return runner.original_forward(tokens, targets, target_mask, n_predict_override,
                                   label_smoothing, unlikelihood_alpha, entropy_reg_beta,
                                   targets_shifted, loss_only=loss_only)


def install_training_graph(model, tokens, targets):
    if hasattr(model, '_nt_training_graph'):
        raise ValueError('Training graph already installed')
    runner = TrainingGraph(model, tokens, targets)
    # Plain owner object: do not register its model reference as a child module.
    model._nt_training_graph = runner
    model.forward = MethodType(_graph_forward, model)
    return runner
