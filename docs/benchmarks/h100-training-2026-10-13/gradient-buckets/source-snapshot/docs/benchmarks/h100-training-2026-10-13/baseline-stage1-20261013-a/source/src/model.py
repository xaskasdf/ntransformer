"""
TinyLlama Model Architecture v2

Optimized based on MobileLLM, SmolLM2, and Phi research:
- Deep-narrow architecture (more layers, smaller dim)
- Flash Attention via F.scaled_dot_product_attention
- Better weight initialization (1/sqrt(fan_in))
- Block-wise weight sharing (MobileLLM)
- KV cache for fast generation
- RoPE with configurable theta
- SwiGLU activation, RMSNorm, GQA
- Multi-Token Prediction (MTP) - predict next k tokens simultaneously
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Model configuration."""
    dim: int = 48
    n_layers: int = 7
    n_heads: int = 4
    n_kv_heads: int = 2  # For GQA; if equal to n_heads, standard MHA
    vocab_size: int = 1024
    hidden_dim: int = 128  # FFN hidden dimension
    max_seq_len: int = 128
    dropout: float = 0.15
    weight_tying: bool = True
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    block_sharing: bool = False  # MobileLLM block-wise weight sharing
    n_predict: int = 1  # Multi-Token Prediction: predict next k tokens (1=standard)
    dense_former: bool = False  # DenseFormer DWA (weighted averaging of all layer outputs)
    value_residual: bool = False  # Value Residual Learning (add layer 0's V to all layers)
    n_registers: int = 0  # Register tokens (learnable tokens prepended to sequence)
    # Dream Architecture fields
    n_loops: int = 1  # Looped transformer: loop through layers multiple times (1=standard)
    ternary: bool = False  # BitNet 1.58b: quantize weights to {-1, 0, +1}
    relu_squared: bool = False  # Use ReLU² FFN (2 matrices) instead of SwiGLU (3 matrices)
    step_embedding: bool = False  # Add per-loop-iteration embedding (depth embedding)
    feedback: bool = False  # Feed last layer output back to first layer between loops

    def __post_init__(self):
        """Ensure proper types after initialization."""
        self.dim = int(self.dim)
        self.n_layers = int(self.n_layers)
        self.n_heads = int(self.n_heads)
        self.n_kv_heads = int(self.n_kv_heads)
        self.vocab_size = int(self.vocab_size)
        self.hidden_dim = int(self.hidden_dim)
        self.max_seq_len = int(self.max_seq_len)
        self.dropout = float(self.dropout)
        self.weight_tying = bool(self.weight_tying)
        self.norm_eps = float(self.norm_eps)
        self.rope_theta = float(self.rope_theta)
        self.block_sharing = bool(self.block_sharing)
        self.n_predict = int(self.n_predict)
        self.dense_former = bool(self.dense_former)
        self.value_residual = bool(self.value_residual)
        self.n_registers = int(self.n_registers)
        self.n_loops = int(self.n_loops)
        self.ternary = bool(self.ternary)
        self.relu_squared = bool(self.relu_squared)
        self.step_embedding = bool(self.step_embedding)
        self.feedback = bool(self.feedback)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config['model'])

    def param_count(self) -> int:
        """Estimate parameter count."""
        embed = self.vocab_size * self.dim
        head_dim = self.dim // self.n_heads

        attn_q = self.dim * self.dim
        attn_kv = 2 * self.dim * (self.n_kv_heads * head_dim)
        attn_o = self.dim * self.dim
        attn = attn_q + attn_kv + attn_o

        ffn = 3 * self.dim * self.hidden_dim
        norms_per_layer = 2 * self.dim

        layer_params = attn + ffn + norms_per_layer

        # Block sharing: only half the layers have unique params
        if self.block_sharing:
            unique_layers = (self.n_layers + 1) // 2
        else:
            unique_layers = self.n_layers
        all_layers = layer_params * unique_layers

        final_norm = self.dim
        output = 0 if self.weight_tying else self.vocab_size * self.dim

        return embed + all_layers + final_norm + output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


class TernaryQuantize(torch.autograd.Function):
    """BitNet 1.58b: quantize weights to {-1, 0, +1} with straight-through estimator."""

    @staticmethod
    def forward(ctx, weight):
        # Absmean quantization: scale = mean(|W|)
        scale = weight.abs().mean() + 1e-8
        # Quantize to {-1, 0, +1}
        weight_q = torch.clamp(torch.round(weight / scale), -1, 1)
        # Return scaled ternary weights
        return weight_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient unchanged
        return grad_output


class TernaryLinear(nn.Module):
    """Linear layer with BitNet 1.58b ternary weight quantization.

    Weights are stored in full precision for optimizer updates,
    but quantized to {-1, 0, +1} in forward pass.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        # Initialize with standard scaling
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = TernaryQuantize.apply(self.weight)
        return F.linear(x, w, self.bias)


class FeedForwardReLU2(nn.Module):
    """ReLU² Feed-Forward Network (2 matrices instead of SwiGLU's 3).

    Compatible with ternary quantization. Uses ReLU² activation
    which produces sparse activations ideal for ternary weights.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        Linear = TernaryLinear if config.ternary else nn.Linear
        self.w1 = Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = Linear(config.hidden_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU²: sparse, gradient-friendly activation
        return self.dropout(self.w2(F.relu(self.w1(x)).square()))


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotary embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys."""
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention with GQA + Flash Attention + KV cache."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout_p = config.dropout

        Linear = TernaryLinear if config.ternary else nn.Linear
        self.wq = Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        v_first: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Value Residual Learning: store raw V, add layer 0's V as residual
        v_out = xv
        if v_first is not None:
            xv = xv + v_first

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose for attention: [B, n_heads, T, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # KV cache for generation
        new_cache = None
        if cache is not None:
            cached_k, cached_v = cache
            xk = torch.cat([cached_k, xk], dim=2)
            xv = torch.cat([cached_v, xv], dim=2)
            new_cache = (xk, xv)
        elif not self.training:
            new_cache = (xk, xv)

        # Repeat KV for GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=1)
            xv = xv.repeat_interleave(self.n_rep, dim=1)

        # Flash Attention via PyTorch SDPA
        dropout_p = self.dropout_p if self.training else 0.0

        if cache is not None:
            # Generation mode: single query token, no causal mask needed
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=dropout_p, is_causal=False
            )
        elif mask is not None:
            # Training with explicit mask
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=mask, dropout_p=dropout_p
            )
        else:
            # Training with built-in causal mask (most efficient)
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=dropout_p, is_causal=True
            )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output), new_cache, v_out


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        Linear = TernaryLinear if config.ternary else nn.Linear
        self.w1 = Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForwardReLU2(config) if config.relu_squared else FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        v_first: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        attn_out, new_cache, v_out = self.attention(self.attention_norm(x), freqs_cis, mask, cache, v_first)
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_cache, v_out


class TinyLlama(nn.Module):
    """TinyLlama language model v2."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        if config.block_sharing:
            # MobileLLM: share weights between adjacent layer pairs
            unique_layers = nn.ModuleList([
                TransformerBlock(config) for _ in range((config.n_layers + 1) // 2)
            ])
            self.layers = nn.ModuleList()
            for i in range(config.n_layers):
                self.layers.append(unique_layers[i // 2])
            # Keep unique_layers alive for parameter counting
            self._unique_layers = unique_layers
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.n_layers)
            ])

        # Final norm
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # Output projection
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        if config.weight_tying:
            self.output.weight = self.tok_embeddings.weight

        # DenseFormer DWA: learnable weights for cross-layer averaging
        # Note: with looping, DWA is applied per-loop (same weights each loop)
        if config.dense_former:
            n_eff = config.n_layers  # layers per loop
            self.dwa_weights = nn.Parameter(torch.zeros(n_eff, n_eff + 1))
            for i in range(n_eff):
                self.dwa_weights.data[i, i + 1] = 1.0

        # Register tokens: learnable tokens prepended to sequence
        # With looping, these act as resonance buffers (refined each loop)
        if config.n_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, config.n_registers, config.dim) * 0.02
            )
            # Resonance gate: blend updated registers with original between loops
            if config.n_loops > 1:
                self.resonance_gate = nn.Parameter(torch.zeros(config.dim))

        # Dream Architecture: looping components
        if config.step_embedding and config.n_loops > 1:
            # Per-loop-iteration embedding (like depth embedding)
            self.step_embeddings = nn.Embedding(config.n_loops, config.dim)
            nn.init.normal_(self.step_embeddings.weight, std=0.02)

        if config.feedback and config.n_loops > 1:
            # Feedback projection: compress last layer output for next loop input
            self.feedback_norm = RMSNorm(config.dim, config.norm_eps)

        # Multi-Token Prediction: extra heads for predicting t+2, t+3, ..., t+k
        # Each head is a dim->dim transform; all share self.output for final projection
        if config.n_predict > 1:
            self.mtp_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.dim, config.dim, bias=False),
                    RMSNorm(config.dim, config.norm_eps),
                )
                for _ in range(config.n_predict - 1)
            ])

        # Precompute RoPE frequencies
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len * 2, config.rope_theta),
            persistent=False
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers)
        self._init_residual_scaling()

    def _init_weights(self, module: nn.Module):
        """Initialize weights with 1/sqrt(fan_in) scaling."""
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            std = 1.0 / math.sqrt(fan_in)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 1.0 / math.sqrt(module.weight.shape[1])
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_residual_scaling(self):
        """Scale residual output projections to prevent growth."""
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for layer in self.layers:
            # Scale attention output projection
            layer.attention.wo.weight.data *= scale
            # Scale FFN output projection
            layer.feed_forward.w2.weight.data *= scale

    def _run_layers(self, h, freqs_cis, v_first_in=None):
        """Run one pass through all transformer layers.

        Returns: (h, v_first) — output hidden states and V anchor for value residual.
        """
        all_h = [h] if self.config.dense_former else None
        v_first = v_first_in

        for i, layer in enumerate(self.layers):
            vf = v_first if self.config.value_residual else None
            h, _, v_out = layer(h, freqs_cis, mask=None, cache=None, v_first=vf)

            # Value Residual: store layer 0's V on first encounter
            if v_first is None and i == 0 and self.config.value_residual:
                v_first = v_out

            # DenseFormer DWA: weighted average of layer outputs (resets each loop)
            if self.config.dense_former:
                all_h.append(h)
                w = self.dwa_weights[i, :i + 2]
                h = sum(w[j] * all_h[j] for j in range(i + 2))

        return h, v_first

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        n_predict_override: Optional[int] = None,
        label_smoothing: float = 0.0,
        unlikelihood_alpha: float = 0.0,
        entropy_reg_beta: float = 0.0,
        targets_shifted: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for training.

        Args:
            tokens: Input token IDs [B, T]
            targets: Target token IDs [B, T]
            target_mask: Mask for target-only loss (fine-tuning)
            n_predict_override: Override number of MTP heads to use (for curriculum)
            label_smoothing: Label smoothing epsilon (0.0=off, 0.1 recommended)
            unlikelihood_alpha: Unlikelihood training weight (0.0=off, 0.5 recommended)
            entropy_reg_beta: Entropy regularization weight (0.0=off, 0.01 recommended)
            targets_shifted: True when labels already contain the next token.
        """
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.config.max_seq_len, \
            f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}"

        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # Register tokens: prepend learnable tokens
        n_reg = self.config.n_registers
        if n_reg > 0:
            regs = self.register_tokens.expand(batch_size, -1, -1)
            h = torch.cat([regs, h], dim=1)

        freqs_cis = self.freqs_cis[:seq_len + n_reg]
        n_loops = self.config.n_loops
        v_first = None

        for loop in range(n_loops):
            # Step embedding: add per-loop depth embedding
            if self.config.step_embedding and n_loops > 1:
                step_emb = self.step_embeddings(
                    torch.tensor(loop, device=h.device)
                )
                h = h + step_emb

            # Feedback: add previous loop's output (normalized) to current input
            if self.config.feedback and n_loops > 1 and loop > 0:
                h = h + self.feedback_norm(h_prev)

            h_prev = h  # save for feedback in next loop

            # Run through all layers
            h, v_first = self._run_layers(h, freqs_cis, v_first_in=v_first)

            # Resonance gating: blend updated registers with original between loops
            if n_reg > 0 and n_loops > 1 and loop < n_loops - 1:
                gate = torch.sigmoid(self.resonance_gate)
                orig_regs = self.register_tokens.expand(batch_size, -1, -1)
                h_regs = h[:, :n_reg]
                h = torch.cat([gate * h_regs + (1 - gate) * orig_regs, h[:, n_reg:]], dim=1)

        # Strip register tokens before output
        if n_reg > 0:
            h = h[:, n_reg:]

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            V = self.config.vocab_size
            # Text datasets supply t+1 labels; chat datasets supply t labels.
            # Shift exactly once, preserving the existing chat/MTP convention.
            offset = 0 if targets_shifted else 1
            end = seq_len - offset
            n_predict_active = n_predict_override if n_predict_override is not None else self.config.n_predict

            if target_mask is not None or n_predict_active <= 1:
                # Standard next-token loss (fine-tuning or MTP disabled/curriculum k=1)
                shift_logits = logits[..., :end, :].contiguous().view(-1, V)
                shift_targets = targets[..., offset:].contiguous().view(-1)

                if target_mask is not None:
                    shift_mask = target_mask[..., offset:].contiguous().view(-1)
                    loss = F.cross_entropy(
                        shift_logits, shift_targets,
                        reduction='none', label_smoothing=label_smoothing
                    )
                    loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
                else:
                    loss = F.cross_entropy(
                        shift_logits, shift_targets,
                        label_smoothing=label_smoothing
                    )
            else:
                # Multi-Token Prediction: average loss over k prediction heads
                total_loss = torch.zeros(1, device=tokens.device)
                n_heads_used = 0

                # Head 0: standard next-token (shift=1)
                shift_logits = logits[..., :end, :].contiguous().view(-1, V)
                shift_targets = targets[..., offset:].contiguous().view(-1)
                total_loss += F.cross_entropy(
                    shift_logits, shift_targets,
                    label_smoothing=label_smoothing
                )
                n_heads_used += 1

                # Extra heads: predict t+2, ..., t+k (limited by curriculum)
                n_extra = min(n_predict_active - 1, len(self.mtp_heads))
                for i in range(n_extra):
                    head = self.mtp_heads[i]
                    shift = i + 1 + offset
                    if shift >= seq_len:
                        break
                    head_logits = self.output(head(h))
                    s_logits = head_logits[..., :-shift, :].contiguous().view(-1, V)
                    s_targets = targets[..., shift:].contiguous().view(-1)
                    total_loss += F.cross_entropy(
                        s_logits, s_targets,
                        label_smoothing=label_smoothing
                    )
                    n_heads_used += 1

                loss = total_loss / n_heads_used

            # --- Unlikelihood Training Loss ---
            if unlikelihood_alpha > 0.0:
                shift_logits_ul = logits[..., :end, :].contiguous()
                shift_input = tokens[..., :end].contiguous()
                B, T_ul, _ = shift_logits_ul.shape

                probs_ul = torch.softmax(shift_logits_ul, dim=-1)
                one_minus = torch.clamp(1.0 - probs_ul, min=1e-5)
                ul_log = -torch.log(one_minus)  # (B, T, V)

                # Build mask: for each position, mark tokens seen in previous context
                prev_mask = torch.zeros_like(ul_log)
                for t in range(1, T_ul):
                    # Tokens seen in positions 0..t-1
                    seen_tokens = shift_input[:, :t]  # (B, t)
                    prev_mask[:, t].scatter_(1, seen_tokens, 1.0)

                # Apply target mask if present (only penalize on target positions)
                if target_mask is not None:
                    shift_tmask = target_mask[..., offset:].contiguous().unsqueeze(-1)
                    ul_loss = (ul_log * prev_mask * shift_tmask).sum() / (shift_tmask.sum() * V + 1e-8)
                else:
                    ul_loss = (ul_log * prev_mask).sum() / (B * T_ul + 1e-8)

                loss = loss + unlikelihood_alpha * ul_loss

            # --- Entropy Regularization ---
            if entropy_reg_beta > 0.0:
                shift_logits_ent = logits[..., :end, :].contiguous().view(-1, V)
                probs_ent = torch.softmax(shift_logits_ent, dim=-1)
                log_probs_ent = torch.log_softmax(shift_logits_ent, dim=-1)
                entropy = -(probs_ent * log_probs_ent).sum(dim=-1)

                if target_mask is not None:
                    ent_mask = target_mask[..., offset:].contiguous().view(-1)
                    avg_entropy = (entropy * ent_mask).sum() / (ent_mask.sum() + 1e-8)
                else:
                    avg_entropy = entropy.mean()

                # Subtract because we want to MAXIMIZE entropy (minimize negative entropy)
                loss = loss - entropy_reg_beta * avg_entropy

        return logits, loss

    def _run_layers_cached(self, h, freqs_cis, loop_caches, v_first_in=None):
        """Run one pass through layers with KV cache (for generation).

        Returns: (h, loop_caches, v_first)
        """
        all_h = [h] if self.config.dense_former else None
        v_first = v_first_in

        for i, layer in enumerate(self.layers):
            vf = v_first if self.config.value_residual else None
            h, loop_caches[i], v_out = layer(h, freqs_cis, mask=None, cache=loop_caches[i], v_first=vf)
            if v_first is None and i == 0 and self.config.value_residual:
                v_first = v_out
            if self.config.dense_former:
                all_h.append(h)
                w = self.dwa_weights[i, :i + 2]
                h = sum(w[j] * all_h[j] for j in range(i + 2))

        return h, loop_caches, v_first

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_tokens: Optional[list] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        contrastive_search: bool = False,
        contrastive_alpha: float = 0.6,
        contrastive_k: int = 6,
    ) -> torch.Tensor:
        """Generate tokens with KV cache for speed. Supports looped models.

        Anti-repetition features:
            repetition_penalty: Divide logits of seen tokens by this (1.0=off, 1.1-1.3 recommended)
            no_repeat_ngram_size: Block repeated n-grams (3 recommended, 0=off)
            contrastive_search: Use contrastive search decoding (overrides sampling)
            contrastive_alpha: Degeneration penalty weight for contrastive search (0.6 recommended)
            contrastive_k: Top-k candidates for contrastive search (4-6 recommended)
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        stop_tokens = stop_tokens or []
        self.eval()

        batch_size, seq_len = tokens.shape
        n_reg = self.config.n_registers
        n_loops = self.config.n_loops
        n_layers = len(self.layers)

        # === PREFILL: process all input tokens through all loops ===
        h = self.tok_embeddings(tokens)

        if n_reg > 0:
            regs = self.register_tokens.expand(batch_size, -1, -1)
            h = torch.cat([regs, h], dim=1)

        freqs_cis = self.freqs_cis[:seq_len + n_reg]

        # KV cache: [n_loops][n_layers]
        all_caches = [[None] * n_layers for _ in range(n_loops)]
        v_first = None

        for loop in range(n_loops):
            if self.config.step_embedding and n_loops > 1:
                step_emb = self.step_embeddings(torch.tensor(loop, device=h.device))
                h = h + step_emb

            if self.config.feedback and n_loops > 1 and loop > 0:
                h = h + self.feedback_norm(h_prev)

            h_prev = h
            h, all_caches[loop], v_first = self._run_layers_cached(
                h, freqs_cis, all_caches[loop], v_first_in=v_first
            )

            if n_reg > 0 and n_loops > 1 and loop < n_loops - 1:
                gate = torch.sigmoid(self.resonance_gate)
                orig_regs = self.register_tokens.expand(batch_size, -1, -1)
                h = torch.cat([gate * h[:, :n_reg] + (1 - gate) * orig_regs, h[:, n_reg:]], dim=1)

        # Strip registers for logits
        if n_reg > 0:
            h = h[:, n_reg:]

        h = self.norm(h)
        logits = self.output(h[:, -1:, :])

        # === DECODE: generate tokens one at a time ===
        generated = tokens
        # Track hidden states for contrastive search
        if contrastive_search:
            prev_hidden_states = []
            # Save last hidden state from prefill
            prev_hidden_states.append(h[:, -1:, :].clone())

        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :].clone()

            # --- Repetition penalty ---
            if repetition_penalty != 1.0:
                generated_tokens = generated[0].tolist()
                seen = set(generated_tokens)
                for token_id in seen:
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            # --- N-gram blocking ---
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                gen_list = generated[0].tolist()
                ngram_len = no_repeat_ngram_size
                # Current (n-1)-gram that would form the start of a repeated n-gram
                current_ngram_prefix = tuple(gen_list[-(ngram_len - 1):])
                # Scan all previous n-grams
                for i in range(len(gen_list) - ngram_len + 1):
                    prev_prefix = tuple(gen_list[i:i + ngram_len - 1])
                    if prev_prefix == current_ngram_prefix:
                        # Ban the token that would complete this n-gram
                        banned_token = gen_list[i + ngram_len - 1]
                        next_logits[0, banned_token] = float('-inf')

            # --- Contrastive search decoding ---
            if contrastive_search and len(prev_hidden_states) > 0:
                # Get top-k candidates by model confidence
                cs_k = min(contrastive_k, next_logits.size(-1))
                top_k_logits, top_k_ids = torch.topk(next_logits, cs_k, dim=-1)
                top_k_probs = F.softmax(top_k_logits, dim=-1)  # (1, k)

                # Compute hidden states for each candidate
                best_score = float('-inf')
                best_token = top_k_ids[0, 0].unsqueeze(0).unsqueeze(0)
                best_hidden = None

                # Stack all previous hidden states: (1, num_prev, dim)
                prev_h_stack = torch.cat(prev_hidden_states, dim=1)

                for j in range(cs_k):
                    cand_id = top_k_ids[0, j]
                    model_confidence = top_k_probs[0, j].item()

                    # Get candidate's hidden state (approximate from embedding)
                    cand_h = self.tok_embeddings(cand_id.unsqueeze(0).unsqueeze(0))
                    cand_h = F.normalize(cand_h, dim=-1)
                    prev_h_norm = F.normalize(prev_h_stack, dim=-1)

                    # Max cosine similarity with previous tokens
                    cos_sim = torch.matmul(cand_h, prev_h_norm.transpose(-1, -2))
                    max_sim = cos_sim.max().item()

                    # Score = model_confidence - alpha * max_similarity
                    score = model_confidence - contrastive_alpha * max_sim

                    if score > best_score:
                        best_score = score
                        best_token = cand_id.unsqueeze(0).unsqueeze(0)
                        best_hidden = cand_h

                next_token = best_token
                if best_hidden is not None:
                    prev_hidden_states.append(best_hidden)
                    # Keep window to prevent memory issues
                    if len(prev_hidden_states) > 128:
                        prev_hidden_states = prev_hidden_states[-64:]

            # --- Standard sampling/greedy ---
            elif temperature > 0:
                scaled_logits = next_logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() in stop_tokens:
                break

            if step < max_new_tokens - 1:
                pos = seq_len + n_reg + step
                if pos >= self.config.max_seq_len + n_reg:
                    break

                h = self.tok_embeddings(next_token)
                freqs_cis_step = self.freqs_cis[pos:pos+1]

                # Decode through all loops
                v_first_step = None
                for loop in range(n_loops):
                    if self.config.step_embedding and n_loops > 1:
                        step_emb = self.step_embeddings(torch.tensor(loop, device=h.device))
                        h = h + step_emb

                    if self.config.feedback and n_loops > 1 and loop > 0:
                        h = h + self.feedback_norm(h_prev_decode)

                    h_prev_decode = h
                    h, all_caches[loop], v_first_step = self._run_layers_cached(
                        h, freqs_cis_step, all_caches[loop], v_first_in=v_first_step
                    )

                h = self.norm(h)
                logits = self.output(h)

        return generated

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str, optimizer=None, step: int = 0, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'step': step,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cuda') -> "TinyLlama":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        if isinstance(config, dict):
            config = ModelConfig(**config)
        # Handle missing config fields from older checkpoints
        for attr, default in [('dense_former', False), ('value_residual', False),
                              ('n_registers', 0), ('n_predict', 1),
                              ('n_loops', 1), ('ternary', False),
                              ('relu_squared', False), ('step_embedding', False),
                              ('feedback', False)]:
            if not hasattr(config, attr):
                setattr(config, attr, default)
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)


def test_model():
    """Test model instantiation and forward pass."""
    print("Testing TinyLlama v2 model...")

    # Test deep-narrow 10M config
    config_10m = ModelConfig(
        dim=192, n_layers=16, n_heads=6, n_kv_heads=2,
        vocab_size=8192, hidden_dim=512, max_seq_len=512,
        dropout=0.05, weight_tying=True, block_sharing=True
    )

    model = TinyLlama(config_10m)
    params = model.count_parameters()
    print(f"10M v2 model parameters: {params:,}")
    print(f"  Estimated: {config_10m.param_count():,}")

    # Test forward
    batch = torch.randint(0, 8192, (2, 64))
    logits, loss = model(batch, batch)
    print(f"  Forward pass OK: logits {logits.shape}, loss {loss.item():.4f}")

    # Test generation with KV cache
    gen = model.generate(batch[0, :5], max_new_tokens=20, temperature=0.8)
    print(f"  Generation OK (KV cache): {gen.shape}")

    # Test 30M deep-narrow
    config_30m = ModelConfig(
        dim=320, n_layers=20, n_heads=8, n_kv_heads=2,
        vocab_size=8192, hidden_dim=864, max_seq_len=512,
        dropout=0.0, weight_tying=True, block_sharing=True
    )

    model_30m = TinyLlama(config_30m)
    print(f"\n30M v2 model parameters: {model_30m.count_parameters():,}")

    # Test Dream Architecture: Looped + Ternary + ReLU² + all features
    print("\n--- Dream Architecture Test ---")
    config_dream = ModelConfig(
        dim=576, n_layers=4, n_heads=8, n_kv_heads=2,
        vocab_size=8192, hidden_dim=1536, max_seq_len=512,
        dropout=0.0, weight_tying=True,
        # Dream features
        n_loops=4,
        ternary=True,
        relu_squared=True,
        step_embedding=True,
        feedback=True,
        dense_former=True,
        value_residual=True,
        n_registers=4,
    )

    model_dream = TinyLlama(config_dream)
    params_dream = model_dream.count_parameters()
    print(f"Dream model unique params: {params_dream:,}")
    print(f"  Effective depth: {config_dream.n_layers} layers × {config_dream.n_loops} loops = {config_dream.n_layers * config_dream.n_loops}")
    print(f"  Logical params: ~{params_dream * config_dream.n_loops:,} (shared across loops)")

    # Test forward
    batch_dream = torch.randint(0, 8192, (2, 64))
    logits_d, loss_d = model_dream(batch_dream, batch_dream)
    print(f"  Forward pass OK: logits {logits_d.shape}, loss {loss_d.item():.4f}")

    # Test generation with KV cache + looping
    gen_d = model_dream.generate(batch_dream[0, :5], max_new_tokens=20, temperature=0.8)
    print(f"  Generation OK (looped KV cache): {gen_d.shape}")

    # Test without ternary (full precision variant)
    config_dream_fp = ModelConfig(
        dim=576, n_layers=4, n_heads=8, n_kv_heads=2,
        vocab_size=8192, hidden_dim=1536, max_seq_len=512,
        dropout=0.0, weight_tying=True,
        n_loops=4, ternary=False, relu_squared=True,
        step_embedding=True, feedback=True,
        dense_former=True, value_residual=True, n_registers=4,
    )
    model_dream_fp = TinyLlama(config_dream_fp)
    logits_fp, loss_fp = model_dream_fp(batch_dream, batch_dream)
    gen_fp = model_dream_fp.generate(batch_dream[0, :5], max_new_tokens=10)
    print(f"  Full-precision dream: params={model_dream_fp.count_parameters():,}, loss={loss_fp.item():.4f}, gen={gen_fp.shape}")

    # Test backward compatibility
    print("\n--- Backward Compatibility ---")
    old_ckpt = model.state_dict()
    model_reload = TinyLlama(config_10m)
    model_reload.load_state_dict(old_ckpt)
    print("  Old checkpoint reload: OK")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_model()
