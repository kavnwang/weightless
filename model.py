"""Starter model for the FineWeb challenge.

Your goal: achieve val loss < 3.3 with the lowest bytes_per_token_infer score.
Modify this model architecture to be as sparse/efficient as possible.

Six variants are provided:
  - baseline:       dense transformer (the starting point)
  - gqa_only:       grouped-query attention only (dense FFN)
  - topk_only:      top-k FFN activation sparsity only (full MHA)
  - baseline_plus:  GQA + top-k FFN activation sparsity (shows clear improvement)
  - mla:            DeepSeek-style Multi-Head Latent Attention (MLA)
  - loop_top4x3_attnres: loop top 4 layers 3x with inter-block attention residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryPositionalEmbedding
from metric import InferenceProfile

# GPT-2 tokenizer vocab size
VOCAB_SIZE = 50257
SEQ_LEN = 512  # 513 - 1 for causal LM


# ============================================================================
# Baseline: dense transformer
# ============================================================================

class SimpleTransformer(nn.Module):
    """A minimal transformer for language modeling.
    
    This is a basic starter -- you should modify/replace this
    to minimize bytes_per_token_infer while achieving val loss < 3.3.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads  # MHA: all heads are KV heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.weight_tied = True
        
        # Token embeddings (no learned positional embedding - using RoPE)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE for positional encoding (applied in attention)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        self._init_weights(n_layers)
    
    def _init_weights(self, n_layers):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                std = fan_in ** -0.5
                if "proj" in name or ".w2" in name:
                    std = std / (2 * n_layers) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=module.weight.shape[1] ** -0.5)
    
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def count_parameters(self, count_zeros: bool = False):
        """Count model parameters.
        
        Args:
            count_zeros: If False, only count non-zero parameters
        
        Returns:
            Total parameter count
        """
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        """Compute bytes_per_token_infer breakdown for single-token decode.

        Candidates: update this method when you change the architecture so
        the metric accurately reflects your design.

        Args:
            weight_dtype_bytes: bytes per weight element (2=bf16, 1=fp8, 0.5=4-bit)
            kv_dtype_bytes: bytes per KV cache element
        """
        d = self.d_model
        h = self.n_heads
        kv_h = self.n_kv_heads
        hd = self.head_dim
        L = self.n_layers
        V = self.vocab_size
        wb = weight_dtype_bytes
        kb = kv_dtype_bytes

        # LM head: logits = hidden @ W.T reads the FULL vocab x d_model matrix
        # This is always a full-matrix read regardless of weight tying.
        lm_head_bytes = V * d * wb

        # Embedding: lookup one row (d_model bytes).
        # When tied with lm_head, that one row is a subset of the full-matrix
        # read above, so with count_reuse=False we don't double-count it.
        if self.weight_tied and not count_reuse:
            embedding_bytes = 0  # subsumed by lm_head full-matrix read
        else:
            embedding_bytes = d * wb

        # Attention projections per layer (all layers summed)
        # Q: d_model -> n_heads * head_dim = d_model
        attn_q_bytes = L * d * (h * hd) * wb
        # K: d_model -> n_kv_heads * head_dim
        attn_k_bytes = L * d * (kv_h * hd) * wb
        # V: same as K
        attn_v_bytes = L * d * (kv_h * hd) * wb
        # O: n_heads * head_dim -> d_model
        attn_o_bytes = L * (h * hd) * d * wb

        # FFN per layer (SwiGLU: w1, w3 are d->d_ff, w2 is d_ff->d)
        ffn_bytes = L * (
            d * self.d_ff  # w1
            + self.d_ff * d  # w2
            + d * self.d_ff  # w3
        ) * wb

        # Norms: 2 per layer (ln1, ln2) + 1 final (ln_f), each has gamma+beta
        norm_bytes = (2 * L + 1) * 2 * d * wb

        # KV cache
        kv_cache_read_bytes = 2 * kv_h * hd * seq_len * L * kb
        kv_cache_write_bytes = 2 * kv_h * hd * L * kb

        # Unique parameters
        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="baseline",
            d_model=d,
            n_layers=L,
            n_heads=h,
            n_kv_heads=kv_h,
            head_dim=hd,
            d_ff=self.d_ff,
            vocab_size=V,
            seq_len=seq_len,
            weight_dtype_bytes=wb,
            kv_dtype_bytes=kb,
            count_reuse=count_reuse,
            embedding_bytes=embedding_bytes,
            attn_q_bytes=attn_q_bytes,
            attn_k_bytes=attn_k_bytes,
            attn_v_bytes=attn_v_bytes,
            attn_o_bytes=attn_o_bytes,
            ffn_bytes=ffn_bytes,
            norm_bytes=norm_bytes,
            lm_head_bytes=lm_head_bytes,
            kv_cache_read_bytes=kv_cache_read_bytes,
            kv_cache_write_bytes=kv_cache_write_bytes,
            unique_param_bytes=unique_numel * wb,
            unique_opt_state_bytes=unique_numel * 12,
        )


# ============================================================================
# Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional GQA and RoPE + Flash Attention.
    
    When n_kv_heads < n_heads, uses Grouped Query Attention:
    Q has n_heads, K/V have n_kv_heads, heads are repeated for the
    dot product.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // n_kv_heads  # how many Q heads per KV head

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        B, T, C = x.shape
        
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = rope(q, positions)
        k = rope(k, positions)
        
        # Expand KV heads for GQA: (B, n_kv_heads, T, hd) -> (B, n_heads, T, hd)
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)
            k = k.reshape(B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)
            v = v.reshape(B, self.n_heads, T, self.head_dim)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MultiLatentAttention(nn.Module):
    """DeepSeek-style Multi-Head Latent Attention (MLA).

    Implements low-rank latent compression for keys/values and queries with
    decoupled RoPE components:
      - KV: h -> c_kv -> (k_c, v_c), and k_r = RoPE(W_kr h)
      - Q:  h -> c_q  -> q_c,         and q_r = RoPE(W_qr c_q)
      - attention uses q=[q_c;q_r], k=[k_c;k_r], v=v_c
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        kv_lora_rank: int,
        q_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        assert self.qk_rope_head_dim % 2 == 0, "qk_rope_head_dim must be even for RoPE"
        assert self.kv_lora_rank > 0 and self.q_lora_rank > 0
        assert self.qk_nope_head_dim > 0 and self.v_head_dim > 0

        # KV path: h -> c_kv -> k_c / v_c
        self.kv_down = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.k_up = nn.Linear(kv_lora_rank, n_heads * qk_nope_head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)
        # Decoupled RoPE key (shared across heads)
        self.k_rope_proj = nn.Linear(d_model, qk_rope_head_dim, bias=False)

        # Q path: h -> c_q -> q_c / q_r
        self.q_down = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_up = nn.Linear(q_lora_rank, n_heads * qk_nope_head_dim, bias=False)
        self.q_rope_proj = nn.Linear(q_lora_rank, n_heads * qk_rope_head_dim, bias=False)

        self.proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def forward(self, x, causal_mask, attention_mask, rope, positions):
        B, T, _ = x.shape

        # Query latent path
        c_q = self.q_down(x)
        q_c = self.q_up(c_q).reshape(B, T, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        q_r = self.q_rope_proj(c_q).reshape(B, T, self.n_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_r = rope(q_r, positions)
        q = torch.cat([q_c, q_r], dim=-1)

        # Key/Value latent path
        c_kv = self.kv_down(x)
        k_c = self.k_up(c_kv).reshape(B, T, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        v_c = self.v_up(c_kv).reshape(B, T, self.n_heads, self.v_head_dim).transpose(1, 2)

        # RoPE key shared across heads
        k_r = self.k_rope_proj(x).reshape(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_r = rope(k_r, positions).expand(B, self.n_heads, T, self.qk_rope_head_dim)
        k = torch.cat([k_c, k_r], dim=-1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v_c,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.v_head_dim)
        return self.proj(out)


# ============================================================================
# FFN
# ============================================================================

class SwiGLUFF(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, device=device, dtype=dtype, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TopKSwiGLUFF(nn.Module):
    """SwiGLU FFN with top-k activation sparsity.

    After computing the gate activations (w1, w3), only the top-k
    neurons are kept.  During inference this means only k rows of w2
    need to be read from memory (instead of all d_ff rows), reducing
    bytes_per_token_infer for the FFN component.

    Training uses the full d_ff (via straight-through or just dense)
    to keep gradients flowing; the top-k mask is applied to the
    activation values so the model learns which neurons matter.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        top_k: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k if top_k is not None else d_ff // 4
        self.w1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, device=device, dtype=dtype, bias=False)

    def forward(self, x):
        gate = F.silu(self.w1(x)) * self.w3(x)  # (B, T, d_ff)
        # Top-k: zero out all but the top-k activations
        if self.top_k < self.d_ff:
            topk_vals, topk_idx = torch.topk(gate.abs(), self.top_k, dim=-1)
            mask = torch.zeros_like(gate)
            mask.scatter_(-1, topk_idx, 1.0)
            gate = gate * mask
        return self.w2(gate)


# ============================================================================
# Inter-block attention residuals (AttnRes)
# ============================================================================

def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: nn.RMSNorm,
) -> torch.Tensor:
    """Inter-block attention over completed block reps + current partial block.

    Args:
        blocks: completed block representations, each [B, T, D]
        partial_block: current in-progress block state [B, T, D]
    """
    v = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]
    k = norm(v)
    logits = torch.einsum("d,nbtd->nbt", proj.weight.squeeze(0), k)
    h = torch.einsum("nbt,nbtd->btd", logits.softmax(0), v)
    return h


class AttnResidualTransformerBlock(nn.Module):
    """Transformer block augmented with inter-block attention residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float,
        layer_number: int,
        block_size: int = 8,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.block_size = block_size
        self.boundary_interval = max(1, block_size // 2)  # 2 sublayers per transformer layer

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = SwiGLUFF(d_model, d_ff)

        self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
        self.attn_res_norm = nn.RMSNorm(d_model)
        self.mlp_res_proj = nn.Linear(d_model, 1, bias=False)
        self.mlp_res_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        blocks,
        hidden_states,
        causal_mask,
        attention_mask,
        rope,
        positions,
        execution_layer_number: int | None = None,
    ):
        partial_block = hidden_states
        if partial_block is None:
            partial_block = blocks[-1]

        # Apply inter-block attention residual before self-attention.
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        # If crossing a block boundary, close the previous partial block.
        # blocks already includes token embedding, so skip layer 0 boundary.
        layer_num = self.layer_number if execution_layer_number is None else execution_layer_number
        if layer_num > 0 and (layer_num % self.boundary_interval == 0):
            blocks.append(partial_block)
            partial_block = None

        attn_out = self.attn(self.ln1(h), causal_mask, attention_mask, rope, positions)
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        # Apply inter-block attention residual before MLP.
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        mlp_out = self.ff(self.ln2(h))
        partial_block = partial_block + mlp_out
        return blocks, partial_block


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, dropout: float, ffn_top_k: int | None = None,
                 attention_type: str = "mha", mla_kwargs: dict | None = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        if attention_type == "mla":
            if mla_kwargs is None:
                raise ValueError("mla_kwargs must be provided when attention_type='mla'")
            self.attn = MultiLatentAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                **mla_kwargs,
            )
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        if ffn_top_k is not None:
            self.ff = TopKSwiGLUFF(d_model, d_ff, top_k=ffn_top_k)
        else:
            self.ff = SwiGLUFF(d_model, d_ff)
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.ff(self.ln2(x))
        return x


# ============================================================================
# Baseline+ : GQA + top-k FFN
# ============================================================================

class BaselinePlusTransformer(SimpleTransformer):
    """Baseline with two clear optimizations that reduce bytes_per_token_infer:

    1. Grouped Query Attention (GQA):  n_kv_heads < n_heads
       -> fewer KV projection weights, smaller KV cache
    2. Top-k FFN activation sparsity:  only top-k neurons of w1/w3 gate
       -> only k rows of w2 read during inference

    These produce a visually obvious improvement in the metric breakdown.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_layers: int = 8,
        d_ff: int = 2048,
        ffn_top_k: int | None = None,  # defaults to d_ff // 4
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
    ):
        # Bypass SimpleTransformer.__init__ -- we rebuild with GQA + top-k
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.ffn_top_k = ffn_top_k if ffn_top_k is not None else d_ff // 4
        self.weight_tied = True

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, dropout,
                             ffn_top_k=self.ffn_top_k)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self._init_weights(n_layers)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        """Compute bytes_per_token_infer for baseline+ (GQA + top-k FFN)."""
        d = self.d_model
        h = self.n_heads
        kv_h = self.n_kv_heads
        hd = self.head_dim
        L = self.n_layers
        V = self.vocab_size
        wb = weight_dtype_bytes
        kb = kv_dtype_bytes
        k = self.ffn_top_k

        # LM head always reads the full vocab x d_model matrix
        lm_head_bytes = V * d * wb

        # Embedding: 1 row, subsumed by lm_head when tied
        if self.weight_tied and not count_reuse:
            embedding_bytes = 0
        else:
            embedding_bytes = d * wb

        attn_q_bytes = L * d * (h * hd) * wb
        attn_k_bytes = L * d * (kv_h * hd) * wb
        attn_v_bytes = L * d * (kv_h * hd) * wb
        attn_o_bytes = L * (h * hd) * d * wb

        # FFN with top-k: w1 and w3 are read fully, w2 only top-k rows
        ffn_bytes = L * (
            d * self.d_ff  # w1 (full)
            + k * d        # w2 (only k rows read)
            + d * self.d_ff  # w3 (full)
        ) * wb

        norm_bytes = (2 * L + 1) * 2 * d * wb

        kv_cache_read_bytes = 2 * kv_h * hd * seq_len * L * kb
        kv_cache_write_bytes = 2 * kv_h * hd * L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="baseline_plus",
            d_model=d,
            n_layers=L,
            n_heads=h,
            n_kv_heads=kv_h,
            head_dim=hd,
            d_ff=self.d_ff,
            vocab_size=V,
            seq_len=seq_len,
            weight_dtype_bytes=wb,
            kv_dtype_bytes=kb,
            count_reuse=count_reuse,
            embedding_bytes=embedding_bytes,
            attn_q_bytes=attn_q_bytes,
            attn_k_bytes=attn_k_bytes,
            attn_v_bytes=attn_v_bytes,
            attn_o_bytes=attn_o_bytes,
            ffn_bytes=ffn_bytes,
            norm_bytes=norm_bytes,
            lm_head_bytes=lm_head_bytes,
            kv_cache_read_bytes=kv_cache_read_bytes,
            kv_cache_write_bytes=kv_cache_write_bytes,
            unique_param_bytes=unique_numel * wb,
            unique_opt_state_bytes=unique_numel * 12,
            notes=f"GQA n_kv_heads={kv_h}, top-k FFN k={k}/{self.d_ff}",
        )


class GQAOnlyTransformer(SimpleTransformer):
    """Ablation: Grouped Query Attention only (dense FFN)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
    ):
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.weight_tied = True

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, dropout, ffn_top_k=None)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self._init_weights(n_layers)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        d = self.d_model
        h = self.n_heads
        kv_h = self.n_kv_heads
        hd = self.head_dim
        L = self.n_layers
        V = self.vocab_size
        wb = weight_dtype_bytes
        kb = kv_dtype_bytes

        lm_head_bytes = V * d * wb
        if self.weight_tied and not count_reuse:
            embedding_bytes = 0
        else:
            embedding_bytes = d * wb

        attn_q_bytes = L * d * (h * hd) * wb
        attn_k_bytes = L * d * (kv_h * hd) * wb
        attn_v_bytes = L * d * (kv_h * hd) * wb
        attn_o_bytes = L * (h * hd) * d * wb

        # Dense FFN (no top-k)
        ffn_bytes = L * (
            d * self.d_ff
            + self.d_ff * d
            + d * self.d_ff
        ) * wb

        norm_bytes = (2 * L + 1) * 2 * d * wb
        kv_cache_read_bytes = 2 * kv_h * hd * seq_len * L * kb
        kv_cache_write_bytes = 2 * kv_h * hd * L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="gqa_only",
            d_model=d,
            n_layers=L,
            n_heads=h,
            n_kv_heads=kv_h,
            head_dim=hd,
            d_ff=self.d_ff,
            vocab_size=V,
            seq_len=seq_len,
            weight_dtype_bytes=wb,
            kv_dtype_bytes=kb,
            count_reuse=count_reuse,
            embedding_bytes=embedding_bytes,
            attn_q_bytes=attn_q_bytes,
            attn_k_bytes=attn_k_bytes,
            attn_v_bytes=attn_v_bytes,
            attn_o_bytes=attn_o_bytes,
            ffn_bytes=ffn_bytes,
            norm_bytes=norm_bytes,
            lm_head_bytes=lm_head_bytes,
            kv_cache_read_bytes=kv_cache_read_bytes,
            kv_cache_write_bytes=kv_cache_write_bytes,
            unique_param_bytes=unique_numel * wb,
            unique_opt_state_bytes=unique_numel * 12,
            notes=f"GQA only n_kv_heads={kv_h}, dense FFN",
        )


class TopKOnlyTransformer(SimpleTransformer):
    """Ablation: top-k FFN activation sparsity only (full MHA KV heads)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        ffn_top_k: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
    ):
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads  # Full MHA K/V heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.ffn_top_k = ffn_top_k if ffn_top_k is not None else d_ff // 4
        self.weight_tied = True

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                n_heads,
                d_ff,
                dropout,
                ffn_top_k=self.ffn_top_k,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self._init_weights(n_layers)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        d = self.d_model
        h = self.n_heads
        kv_h = self.n_kv_heads
        hd = self.head_dim
        L = self.n_layers
        V = self.vocab_size
        wb = weight_dtype_bytes
        kb = kv_dtype_bytes
        k = self.ffn_top_k

        lm_head_bytes = V * d * wb
        if self.weight_tied and not count_reuse:
            embedding_bytes = 0
        else:
            embedding_bytes = d * wb

        attn_q_bytes = L * d * (h * hd) * wb
        attn_k_bytes = L * d * (kv_h * hd) * wb
        attn_v_bytes = L * d * (kv_h * hd) * wb
        attn_o_bytes = L * (h * hd) * d * wb

        # Top-k FFN with full MHA attention.
        ffn_bytes = L * (
            d * self.d_ff
            + k * d
            + d * self.d_ff
        ) * wb

        norm_bytes = (2 * L + 1) * 2 * d * wb
        kv_cache_read_bytes = 2 * kv_h * hd * seq_len * L * kb
        kv_cache_write_bytes = 2 * kv_h * hd * L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="topk_only",
            d_model=d,
            n_layers=L,
            n_heads=h,
            n_kv_heads=kv_h,
            head_dim=hd,
            d_ff=self.d_ff,
            vocab_size=V,
            seq_len=seq_len,
            weight_dtype_bytes=wb,
            kv_dtype_bytes=kb,
            count_reuse=count_reuse,
            embedding_bytes=embedding_bytes,
            attn_q_bytes=attn_q_bytes,
            attn_k_bytes=attn_k_bytes,
            attn_v_bytes=attn_v_bytes,
            attn_o_bytes=attn_o_bytes,
            ffn_bytes=ffn_bytes,
            norm_bytes=norm_bytes,
            lm_head_bytes=lm_head_bytes,
            kv_cache_read_bytes=kv_cache_read_bytes,
            kv_cache_write_bytes=kv_cache_write_bytes,
            unique_param_bytes=unique_numel * wb,
            unique_opt_state_bytes=unique_numel * 12,
            notes=f"Top-k FFN only k={k}/{self.d_ff}, full MHA n_kv_heads={kv_h}",
        )


class MLATransformer(SimpleTransformer):
    """Transformer variant with DeepSeek-style Multi-Head Latent Attention."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        kv_lora_rank: int | None = None,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
    ):
        nn.Module.__init__(self)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads

        # Practical defaults for this starter codebase while keeping MLA configurable.
        default_rope_dim = max(2, (head_dim // 2) * 2)
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else max(16, d_model // 8)
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else max(16, d_model // 8)
        self.qk_nope_head_dim = qk_nope_head_dim if qk_nope_head_dim is not None else head_dim
        self.qk_rope_head_dim = qk_rope_head_dim if qk_rope_head_dim is not None else default_rope_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        assert self.qk_rope_head_dim % 2 == 0, "qk_rope_head_dim must be even"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = 1  # MLA caches latent KV plus shared RoPE key component
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.weight_tied = True

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE is applied only on the decoupled q_r / k_r dimensions.
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.qk_rope_head_dim,
            max_seq_len=max_seq_len,
        )

        mla_kwargs = {
            "kv_lora_rank": self.kv_lora_rank,
            "q_lora_rank": self.q_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
        }
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                n_heads,
                d_ff,
                dropout,
                attention_type="mla",
                mla_kwargs=mla_kwargs,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self._init_weights(n_layers)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        """Compute bytes_per_token_infer for MLA variant."""
        d = self.d_model
        h = self.n_heads
        L = self.n_layers
        V = self.vocab_size
        wb = weight_dtype_bytes
        kb = kv_dtype_bytes

        dc = self.kv_lora_rank
        dcp = self.q_lora_rank
        d_nope = self.qk_nope_head_dim
        d_rope = self.qk_rope_head_dim
        d_v = self.v_head_dim

        lm_head_bytes = V * d * wb
        if self.weight_tied and not count_reuse:
            embedding_bytes = 0
        else:
            embedding_bytes = d * wb

        # MLA attention projections per layer:
        # Q path: d->dcp, dcp->h*d_nope, dcp->h*d_rope
        attn_q_bytes = L * (
            d * dcp
            + dcp * (h * d_nope)
            + dcp * (h * d_rope)
        ) * wb
        # K/V path: d->dc, dc->h*d_nope, dc->h*d_v, and decoupled k_rope d->d_rope
        attn_k_bytes = L * (
            d * dc
            + dc * (h * d_nope)
            + d * d_rope
        ) * wb
        attn_v_bytes = L * (dc * (h * d_v)) * wb
        attn_o_bytes = L * (h * d_v) * d * wb

        ffn_bytes = L * (
            d * self.d_ff
            + self.d_ff * d
            + d * self.d_ff
        ) * wb
        norm_bytes = (2 * L + 1) * 2 * d * wb

        # MLA cache stores compressed latent c_kv plus shared RoPE key component.
        kv_cache_token_width = dc + d_rope
        kv_cache_read_bytes = kv_cache_token_width * seq_len * L * kb
        kv_cache_write_bytes = kv_cache_token_width * L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="mla",
            d_model=d,
            n_layers=L,
            n_heads=h,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            d_ff=self.d_ff,
            vocab_size=V,
            seq_len=seq_len,
            weight_dtype_bytes=wb,
            kv_dtype_bytes=kb,
            count_reuse=count_reuse,
            embedding_bytes=embedding_bytes,
            attn_q_bytes=attn_q_bytes,
            attn_k_bytes=attn_k_bytes,
            attn_v_bytes=attn_v_bytes,
            attn_o_bytes=attn_o_bytes,
            ffn_bytes=ffn_bytes,
            norm_bytes=norm_bytes,
            lm_head_bytes=lm_head_bytes,
            kv_cache_read_bytes=kv_cache_read_bytes,
            kv_cache_write_bytes=kv_cache_write_bytes,
            unique_param_bytes=unique_numel * wb,
            unique_opt_state_bytes=unique_numel * 12,
            notes=(
                f"MLA dc={dc}, dcp={dcp}, qk_nope={d_nope}, "
                f"qk_rope={d_rope}, v_head_dim={d_v}"
            ),
        )


class LoopTop4x3AttnResTransformer(SimpleTransformer):
    """Loop top 4 layers for 3 passes + inter-block attention residuals."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        loop_block_size: int = 4,
        loop_repeats: int = 3,
        attn_res_block_size: int = 4,
    ):
        nn.Module.__init__(self)
        if n_layers < loop_block_size:
            raise ValueError(
                f"loop_block_size={loop_block_size} requires n_layers >= {loop_block_size}, got {n_layers}"
            )
        if loop_repeats < 1:
            raise ValueError(f"loop_repeats must be >= 1, got {loop_repeats}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.weight_tied = True

        self.loop_block_size = loop_block_size
        self.loop_repeats = loop_repeats
        self.loop_start = self.n_layers - self.loop_block_size
        self.attn_res_block_size = attn_res_block_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )
        self.layers = nn.ModuleList([
            AttnResidualTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                layer_number=i,
                block_size=self.attn_res_block_size,
            )
            for i in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self._init_weights(n_layers)

    @property
    def effective_layer_executions(self) -> int:
        return self.loop_start + self.loop_block_size * self.loop_repeats

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        x = self.dropout(x)
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # Inter-block residual memory starts with token embedding representation.
        blocks = [x]
        partial = x
        exec_layer_number = 0

        # Prefix layers run once.
        for i in range(self.loop_start):
            blocks, partial = self.layers[i](
                blocks,
                partial,
                causal_mask,
                attention_mask,
                self.rope,
                positions,
                execution_layer_number=exec_layer_number,
            )
            exec_layer_number += 1

        # Reuse top block for multiple passes.
        for _ in range(self.loop_repeats):
            for i in range(self.loop_start, self.n_layers):
                blocks, partial = self.layers[i](
                    blocks,
                    partial,
                    causal_mask,
                    attention_mask,
                    self.rope,
                    positions,
                    execution_layer_number=exec_layer_number,
                )
                exec_layer_number += 1

        x = self.ln_f(partial)
        logits = self.head(x)
        return logits

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        """Compute bytes_per_token_infer for looped + AttnRes variant."""
        d = self.d_model
        h = self.n_heads
        kv_h = self.n_kv_heads
        hd = self.head_dim
        physical_L = self.n_layers
        exec_L = self.effective_layer_executions
        V = self.vocab_size
        wb = weight_dtype_bytes
        kb = kv_dtype_bytes

        lm_head_bytes = V * d * wb
        if self.weight_tied and not count_reuse:
            embedding_bytes = 0
        else:
            embedding_bytes = d * wb

        # Projections are read each execution pass.
        attn_q_bytes = exec_L * d * (h * hd) * wb
        attn_k_bytes = exec_L * d * (kv_h * hd) * wb
        attn_v_bytes = exec_L * d * (kv_h * hd) * wb
        attn_o_bytes = exec_L * (h * hd) * d * wb
        ffn_bytes = exec_L * (
            d * self.d_ff
            + self.d_ff * d
            + d * self.d_ff
        ) * wb

        # LayerNorm/LN + AttnRes RMSNorm + AttnRes projections.
        norm_bytes = (
            (2 * exec_L + 1) * 2 * d * wb      # ln1, ln2, ln_f
            + (2 * exec_L) * d * wb            # attn_res_norm + mlp_res_norm (RMSNorm weights)
            + (2 * exec_L) * d * wb            # attn_res_proj + mlp_res_proj (1xd each)
        )

        kv_cache_read_bytes = 2 * kv_h * hd * seq_len * exec_L * kb
        kv_cache_write_bytes = 2 * kv_h * hd * exec_L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="loop_top4x3_attnres",
            d_model=d,
            n_layers=physical_L,
            n_heads=h,
            n_kv_heads=kv_h,
            head_dim=hd,
            d_ff=self.d_ff,
            vocab_size=V,
            seq_len=seq_len,
            weight_dtype_bytes=wb,
            kv_dtype_bytes=kb,
            count_reuse=count_reuse,
            embedding_bytes=embedding_bytes,
            attn_q_bytes=attn_q_bytes,
            attn_k_bytes=attn_k_bytes,
            attn_v_bytes=attn_v_bytes,
            attn_o_bytes=attn_o_bytes,
            ffn_bytes=ffn_bytes,
            norm_bytes=norm_bytes,
            lm_head_bytes=lm_head_bytes,
            kv_cache_read_bytes=kv_cache_read_bytes,
            kv_cache_write_bytes=kv_cache_write_bytes,
            unique_param_bytes=unique_numel * wb,
            unique_opt_state_bytes=unique_numel * 12,
            notes=(
                f"Top {self.loop_block_size} layers looped x{self.loop_repeats}; "
                f"AttnRes block_size={self.attn_res_block_size}"
            ),
        )


# ============================================================================
# Factory
# ============================================================================

def create_model(variant: str = "baseline", **kwargs):
    """Factory function to create a model.

    Args:
        variant: "baseline", "gqa_only", "topk_only", "baseline_plus", "mla", or "loop_top4x3_attnres"
        **kwargs: passed to the model constructor
    """
    mla_only_keys = {
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    }
    if variant != "mla":
        kwargs = {k: v for k, v in kwargs.items() if k not in mla_only_keys}

    if variant == "gqa_only":
        return GQAOnlyTransformer(**kwargs)
    elif variant == "topk_only":
        return TopKOnlyTransformer(**kwargs)
    elif variant == "baseline_plus":
        return BaselinePlusTransformer(**kwargs)
    elif variant == "mla":
        return MLATransformer(**kwargs)
    elif variant == "loop_top4x3_attnres":
        return LoopTop4x3AttnResTransformer(**kwargs)
    else:
        return SimpleTransformer(**kwargs)


if __name__ == "__main__":
    from metric import print_profile

    for variant in ["baseline", "gqa_only", "topk_only", "baseline_plus", "mla", "loop_top4x3_attnres"]:
        print(f"\n{'='*60}")
        print(f"  {variant}")
        print(f"{'='*60}")
        model = create_model(variant=variant)
        total_params = model.count_parameters(count_zeros=True)
        nonzero_params = model.count_parameters(count_zeros=False)
        print(f"  Total parameters:    {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")
        profile = model.get_inference_profile()
        print_profile(profile)
