"""Starter model for the FineWeb challenge.

Your goal: achieve val loss < 3.3 with the lowest bytes_per_token_infer score.
Modify this model architecture to be as sparse/efficient as possible.

Six variants are provided:
  - baseline:       dense transformer (the starting point)
  - gqa_only:       grouped-query attention only (dense FFN)
  - topk_only:      top-k FFN activation sparsity only (full MHA)
  - baseline_plus:  GQA + top-k FFN activation sparsity (shows clear improvement)
  - mla:            DeepSeek-style Multi-Head Latent Attention (MLA)
  - mla_twostage_svd_mem12_binarydp: 12-layer MLA + binary-DP memory retrieval
  - dp_shared_memory: standard attention + binary-DP shared memory layers
  - loop_top4x3_attnres: loop top 4 layers 3x with inter-block attention residuals
  - mla_hybrid_loop12: 12-layer MLA hybrid with memory/SVD/Monarch + top4 looping
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from rope import RotaryPositionalEmbedding
from metric import InferenceProfile

# GPT-2 tokenizer vocab size
VOCAB_SIZE = 50257
SEQ_LEN = 512  # 513 - 1 for causal LM
DEFAULT_HOT_TOKEN_CACHE_PATH = "cache/hot_tokens_train1p3b_top2000.pt"


def _load_hot_token_ids(
    cache_path: str,
    vocab_size: int,
    hot_token_k: int,
) -> torch.Tensor:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Hot-token cache not found at '{cache_path}'. "
            f"Build it first with: python build_hot_token_cache.py --cache_path {cache_path}"
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "hot_token_ids" not in payload:
        raise ValueError(
            f"Invalid hot-token cache at '{cache_path}': missing 'hot_token_ids'."
        )
    token_ids = torch.as_tensor(payload["hot_token_ids"], dtype=torch.long).flatten()
    split = payload.get("split")
    subset = payload.get("subset")
    if split is not None and split != "train":
        raise ValueError(
            f"Hot-token cache split must be 'train' (1.3B train split in this repo), got '{split}'."
        )
    if subset is not None and subset != "sample-10BT_max_length_513":
        raise ValueError(
            f"Hot-token cache subset must be 'sample-10BT_max_length_513', got '{subset}'."
        )
    if token_ids.numel() < hot_token_k:
        raise ValueError(
            f"Hot-token cache has only {token_ids.numel()} ids but requested {hot_token_k}."
        )
    token_ids = token_ids[:hot_token_k]
    if token_ids.min().item() < 0 or token_ids.max().item() >= vocab_size:
        raise ValueError(
            f"Hot-token cache contains out-of-range ids for vocab_size={vocab_size}."
        )
    if torch.unique(token_ids).numel() != token_ids.numel():
        raise ValueError("Hot-token cache contains duplicate token ids.")
    return token_ids


class HotColdTiedEmbedding(nn.Module):
    """Tied token embedding + output projection with hot/cold factorization.

    Hot tokens are stored as full d_model vectors.
    Cold tokens are stored as rank-r factors: U[token, r] and V[r, d_model].
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        hot_token_ids: torch.Tensor,
        cold_latent_dim: int = 128,
    ):
        super().__init__()
        if cold_latent_dim < 1:
            raise ValueError(f"cold_latent_dim must be >= 1, got {cold_latent_dim}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.cold_latent_dim = cold_latent_dim

        hot_token_ids = hot_token_ids.to(torch.long).flatten()
        hot_mask = torch.zeros(vocab_size, dtype=torch.bool)
        hot_mask[hot_token_ids] = True
        cold_token_ids = torch.nonzero(~hot_mask, as_tuple=False).squeeze(-1)

        token_to_hot_idx = torch.full((vocab_size,), -1, dtype=torch.long)
        token_to_hot_idx[hot_token_ids] = torch.arange(hot_token_ids.numel(), dtype=torch.long)
        token_to_cold_idx = torch.full((vocab_size,), -1, dtype=torch.long)
        token_to_cold_idx[cold_token_ids] = torch.arange(cold_token_ids.numel(), dtype=torch.long)

        self.register_buffer("hot_token_ids", hot_token_ids, persistent=True)
        self.register_buffer("cold_token_ids", cold_token_ids, persistent=True)
        self.register_buffer("hot_token_mask", hot_mask, persistent=True)
        self.register_buffer("token_to_hot_idx", token_to_hot_idx, persistent=False)
        self.register_buffer("token_to_cold_idx", token_to_cold_idx, persistent=False)

        self.num_hot_tokens = hot_token_ids.numel()
        self.num_cold_tokens = cold_token_ids.numel()

        self.hot_emb = nn.Embedding(self.num_hot_tokens, d_model)
        self.cold_emb_u = nn.Embedding(self.num_cold_tokens, cold_latent_dim)
        self.cold_latent_to_model = nn.Linear(cold_latent_dim, d_model, bias=False)

    def token_is_hot(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.hot_token_mask[token_ids]

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1)
        hot_pos = self.hot_token_mask[flat_ids]
        out = torch.empty(
            flat_ids.numel(),
            self.d_model,
            device=flat_ids.device,
            dtype=self.hot_emb.weight.dtype,
        )

        if hot_pos.any():
            hot_ids = flat_ids[hot_pos]
            out[hot_pos] = self.hot_emb(self.token_to_hot_idx[hot_ids]).to(out.dtype)
        if (~hot_pos).any():
            cold_ids = flat_ids[~hot_pos]
            cold_latent = self.cold_emb_u(self.token_to_cold_idx[cold_ids])
            out[~hot_pos] = self.cold_latent_to_model(cold_latent).to(out.dtype)
        return out.view(*input_ids.shape, self.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def _cold_full_weight(self) -> torch.Tensor:
        # U @ V where U is [num_cold, r], V is [r, d_model]
        return self.cold_emb_u.weight @ self.cold_latent_to_model.weight.T

    def full_weight(self) -> torch.Tensor:
        full = torch.empty(
            self.vocab_size,
            self.d_model,
            device=self.hot_emb.weight.device,
            dtype=self.hot_emb.weight.dtype,
        )
        full[self.hot_token_ids] = self.hot_emb.weight
        full[self.cold_token_ids] = self._cold_full_weight()
        return full

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out_shape = (*hidden_states.shape[:-1], self.vocab_size)
        logits = torch.empty(
            out_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        hot_logits = torch.matmul(hidden_states, self.hot_emb.weight.T)
        logits[..., self.hot_token_ids] = hot_logits

        hidden_latent = torch.matmul(hidden_states, self.cold_latent_to_model.weight)
        cold_logits = torch.matmul(hidden_latent, self.cold_emb_u.weight.T)
        logits[..., self.cold_token_ids] = cold_logits
        return logits


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

    def token_partition_masks(self, token_ids: torch.Tensor):
        """Optional token partitioning hook for loss reporting."""
        return None, None


class HotColdSVDTransformer(SimpleTransformer):
    """Transformer with hot-token dense embeddings and cold-token SVD factors."""

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
        hot_token_k: int = 2000,
        cold_latent_dim: int = 128,
        hot_token_cache_path: str = DEFAULT_HOT_TOKEN_CACHE_PATH,
        svd_switch_fraction: float = 0.5,
    ):
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.weight_tied = True
        self.hot_token_k = hot_token_k
        self.cold_latent_dim = cold_latent_dim
        self.hot_token_cache_path = hot_token_cache_path

        hot_token_ids = _load_hot_token_ids(
            cache_path=hot_token_cache_path,
            vocab_size=vocab_size,
            hot_token_k=hot_token_k,
        )
        self.token_emb = HotColdTiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            hot_token_ids=hot_token_ids,
            cold_latent_dim=cold_latent_dim,
        )
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        self._init_weights(n_layers)

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
        logits = self.token_emb.logits(x)
        return logits

    def token_partition_masks(self, token_ids: torch.Tensor):
        hot_mask = self.token_emb.token_is_hot(token_ids)
        cold_mask = ~hot_mask
        return hot_mask, cold_mask

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
        r = self.cold_latent_dim
        n_hot = self.token_emb.num_hot_tokens
        n_cold = self.token_emb.num_cold_tokens

        # Split LM head read:
        # - hot logits: hidden @ W_hot^T
        # - cold logits: (hidden @ V^T) @ U^T
        lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
        if self.weight_tied and not count_reuse:
            embedding_bytes = 0
        else:
            # One token lookup: either dense hot row or latent cold row + projector.
            embedding_bytes = max(d, r + d * r) * wb

        attn_q_bytes = L * d * (h * hd) * wb
        attn_k_bytes = L * d * (kv_h * hd) * wb
        attn_v_bytes = L * d * (kv_h * hd) * wb
        attn_o_bytes = L * (h * hd) * d * wb
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
            model_name="hotcold_svd",
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
            notes=(
                f"hot={n_hot}, cold={n_cold}, cold_rank={r}, "
                f"cache={self.hot_token_cache_path}"
            ),
        )


class TwoStageSVDTransformer(SimpleTransformer):
    """Two-stage vocab training: dense first, then hot/cold SVD compression."""

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
        hot_token_k: int = 2000,
        cold_latent_dim: int = 128,
        hot_token_cache_path: str = DEFAULT_HOT_TOKEN_CACHE_PATH,
        svd_switch_fraction: float = 0.5,
    ):
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.weight_tied = True
        self.hot_token_k = hot_token_k
        self.cold_latent_dim = cold_latent_dim
        self.hot_token_cache_path = hot_token_cache_path
        if not (0.0 <= svd_switch_fraction <= 1.0):
            raise ValueError(f"svd_switch_fraction must be in [0, 1], got {svd_switch_fraction}")
        self.svd_switch_fraction = svd_switch_fraction
        self.register_buffer("uses_hotcold_flag", torch.tensor(False, dtype=torch.bool), persistent=True)

        hot_token_ids = _load_hot_token_ids(
            cache_path=hot_token_cache_path,
            vocab_size=vocab_size,
            hot_token_k=hot_token_k,
        )
        self.full_token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb = self.full_token_emb
        self.hotcold_emb = HotColdTiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            hot_token_ids=hot_token_ids,
            cold_latent_dim=cold_latent_dim,
        )
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self._init_weights(n_layers)

    def convert_full_to_hotcold_svd(self):
        """Initialize hot/cold factors from the current dense embedding matrix."""
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            full_weight = self.full_token_emb.weight.detach()
            hot_ids = self.hotcold_emb.hot_token_ids
            cold_ids = self.hotcold_emb.cold_token_ids

            self.hotcold_emb.hot_emb.weight.copy_(full_weight[hot_ids])

            cold_full = full_weight[cold_ids].float()
            U, S, Vh = torch.linalg.svd(cold_full, full_matrices=False)
            rank = min(self.cold_latent_dim, S.numel())
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            cold_u = U_r * S_r.unsqueeze(0)

            self.hotcold_emb.cold_emb_u.weight.zero_()
            self.hotcold_emb.cold_emb_u.weight[:, :rank].copy_(cold_u.to(self.hotcold_emb.cold_emb_u.weight.dtype))

            self.hotcold_emb.cold_latent_to_model.weight.zero_()
            self.hotcold_emb.cold_latent_to_model.weight[:, :rank].copy_(
                Vh_r.T.to(self.hotcold_emb.cold_latent_to_model.weight.dtype)
            )

            self.full_token_emb.weight.requires_grad_(False)
            self.uses_hotcold_flag.fill_(True)

    def token_partition_masks(self, token_ids: torch.Tensor):
        hot_mask = self.hotcold_emb.token_is_hot(token_ids)
        cold_mask = ~hot_mask
        return hot_mask, cold_mask

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        if bool(self.uses_hotcold_flag.item()):
            x = self.hotcold_emb(input_ids)
        else:
            x = self.full_token_emb(input_ids)
        x = self.dropout(x)
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        x = self.ln_f(x)
        if bool(self.uses_hotcold_flag.item()):
            return self.hotcold_emb.logits(x)
        return F.linear(x, self.full_token_emb.weight)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        if not bool(self.uses_hotcold_flag.item()):
            d = self.d_model
            h = self.n_heads
            kv_h = self.n_kv_heads
            hd = self.head_dim
            L = self.n_layers
            V = self.vocab_size
            wb = weight_dtype_bytes
            kb = kv_dtype_bytes
            lm_head_bytes = V * d * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else d * wb
            attn_q_bytes = L * d * (h * hd) * wb
            attn_k_bytes = L * d * (kv_h * hd) * wb
            attn_v_bytes = L * d * (kv_h * hd) * wb
            attn_o_bytes = L * (h * hd) * d * wb
            ffn_bytes = L * (d * self.d_ff + self.d_ff * d + d * self.d_ff) * wb
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
                model_name="twostage_svd_dense_phase",
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
                notes="dense phase (pre-switch)",
            )

        d = self.d_model
        h = self.n_heads
        kv_h = self.n_kv_heads
        hd = self.head_dim
        L = self.n_layers
        V = self.vocab_size
        wb = weight_dtype_bytes
        kb = kv_dtype_bytes
        r = self.cold_latent_dim
        n_hot = self.hotcold_emb.num_hot_tokens
        n_cold = self.hotcold_emb.num_cold_tokens
        lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
        embedding_bytes = 0 if (self.weight_tied and not count_reuse) else max(d, r + d * r) * wb
        attn_q_bytes = L * d * (h * hd) * wb
        attn_k_bytes = L * d * (kv_h * hd) * wb
        attn_v_bytes = L * d * (kv_h * hd) * wb
        attn_o_bytes = L * (h * hd) * d * wb
        ffn_bytes = L * (d * self.d_ff + self.d_ff * d + d * self.d_ff) * wb
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
            model_name="twostage_svd_hotcold_phase",
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
            notes="compressed hot/cold phase (post-switch)",
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


class MonarchLinear(nn.Module):
    """Monarch-style structured linear layer with block-wise factors."""

    def __init__(self, in_features: int, out_features: int, block_size: int = 32):
        super().__init__()
        if in_features != out_features:
            raise ValueError("MonarchLinear currently requires in_features == out_features")
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by block_size={block_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.n_blocks = in_features // block_size
        self.left = nn.Parameter(torch.empty(self.n_blocks, block_size, block_size))
        self.right = nn.Parameter(torch.empty(self.n_blocks, block_size, block_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = self.block_size ** -0.5
        nn.init.normal_(self.left, mean=0.0, std=std)
        nn.init.normal_(self.right, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [..., D]
        x_blocks = x.view(*x.shape[:-1], self.n_blocks, self.block_size)
        y = torch.einsum("...nb,nab->...na", x_blocks, self.right)
        y = torch.einsum("...na,nab->...nb", y, self.left)
        return y.reshape(*x.shape[:-1], self.out_features)


class MultiLatentAttentionMonarch(MultiLatentAttention):
    """MLA with Monarch-structured output projection."""

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
        monarch_block_size: int = 32,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        )
        proj_in = n_heads * v_head_dim
        self.proj = MonarchLinear(proj_in, d_model, block_size=monarch_block_size)
        self.monarch_block_size = monarch_block_size


class TwoStageSVDLinear(nn.Module):
    """Dense-to-SVD linear that converts once during training."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.full = nn.Linear(in_features, out_features, bias=bias)
        self.left = nn.Linear(self.rank, out_features, bias=False)
        self.right = nn.Linear(in_features, self.rank, bias=False)
        self.register_buffer("uses_svd_flag", torch.tensor(False, dtype=torch.bool), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.uses_svd_flag.item()):
            return self.left(self.right(x))
        return self.full(x)

    def convert_full_to_svd(self):
        if bool(self.uses_svd_flag.item()):
            return
        with torch.no_grad():
            w = self.full.weight.detach().float()
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            r = min(self.rank, S.numel())
            left = U[:, :r] * S[:r].unsqueeze(0)
            right = Vh[:r, :]

            self.left.weight.zero_()
            self.right.weight.zero_()
            self.left.weight[:, :r].copy_(left.to(self.left.weight.dtype))
            self.right.weight[:r, :].copy_(right.to(self.right.weight.dtype))

            self.full.weight.requires_grad_(False)
            if self.full.bias is not None:
                self.full.bias.requires_grad_(False)
            self.uses_svd_flag.fill_(True)

    def active_weight_numel(self) -> int:
        if bool(self.uses_svd_flag.item()):
            return self.left.weight.numel() + self.right.weight.numel()
        n = self.full.weight.numel()
        if self.full.bias is not None:
            n += self.full.bias.numel()
        return n


class MultiLatentAttentionSVD(MultiLatentAttention):
    """MLA with two-stage SVD output projection."""

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
        svd_rank: int,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        )
        proj_in = n_heads * v_head_dim
        self.proj = TwoStageSVDLinear(proj_in, d_model, rank=svd_rank, bias=False)

    def convert_full_to_svd(self):
        self.proj.convert_full_to_svd()


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


class SVDSwiGLUFF(nn.Module):
    """SwiGLU FFN with dense->SVD switchable linear projections."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        svd_rank: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.svd_rank = svd_rank
        self.w1 = TwoStageSVDLinear(d_model, d_ff, rank=svd_rank, bias=False)
        self.w2 = TwoStageSVDLinear(d_ff, d_model, rank=svd_rank, bias=False)
        self.w3 = TwoStageSVDLinear(d_model, d_ff, rank=svd_rank, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def convert_full_to_svd(self):
        self.w1.convert_full_to_svd()
        self.w2.convert_full_to_svd()
        self.w3.convert_full_to_svd()

    def active_weight_numel(self) -> int:
        return (
            self.w1.active_weight_numel()
            + self.w2.active_weight_numel()
            + self.w3.active_weight_numel()
        )


class LoRALinear(nn.Module):
    """Dense linear with switchable LoRA residual."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False)
        self.register_buffer("uses_lora_flag", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if bool(self.uses_lora_flag.item()):
            out = out + self.scaling * self.lora_up(self.lora_down(x))
        return out

    def enable_lora(self):
        self.uses_lora_flag.fill_(True)

    def active_weight_numel(self) -> int:
        n = self.base.weight.numel()
        if self.base.bias is not None:
            n += self.base.bias.numel()
        if bool(self.uses_lora_flag.item()):
            n += self.lora_down.weight.numel() + self.lora_up.weight.numel()
        return n

    # Kept so existing layer.convert_full_to_svd() call-sites can drive phase switch.
    def convert_full_to_svd(self):
        self.enable_lora()


class LoRASwiGLUFF(nn.Module):
    """SwiGLU FFN with switchable LoRA residuals on all three projections."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        lora_rank: int,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.lora_rank = lora_rank
        self.w1 = LoRALinear(d_model, d_ff, rank=lora_rank, alpha=lora_alpha, bias=False)
        self.w2 = LoRALinear(d_ff, d_model, rank=lora_rank, alpha=lora_alpha, bias=False)
        self.w3 = LoRALinear(d_model, d_ff, rank=lora_rank, alpha=lora_alpha, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def convert_full_to_svd(self):
        self.w1.enable_lora()
        self.w2.enable_lora()
        self.w3.enable_lora()

    def active_weight_numel(self) -> int:
        return (
            self.w1.active_weight_numel()
            + self.w2.active_weight_numel()
            + self.w3.active_weight_numel()
        )


class MonarchSwiGLUFF(nn.Module):
    """SwiGLU FFN with Monarch-structured projections (requires d_ff == d_model)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        monarch_block_size: int = 32,
    ):
        super().__init__()
        if d_ff != d_model:
            raise ValueError(
                f"MonarchSwiGLUFF requires d_ff == d_model, got d_ff={d_ff}, d_model={d_model}"
            )
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = MonarchLinear(d_model, d_ff, block_size=monarch_block_size)
        self.w2 = MonarchLinear(d_ff, d_model, block_size=monarch_block_size)
        self.w3 = MonarchLinear(d_model, d_ff, block_size=monarch_block_size)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# Memory layers (product-key memory from "Memory Layers at Scale")
# ============================================================================

def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class ProductKeyValueStore(nn.Module):
    """Trainable value table of size mem_n_keys^2."""

    def __init__(self, mem_n_keys: int, value_dim: int):
        super().__init__()
        self.mem_n_keys = mem_n_keys
        self.size = mem_n_keys * mem_n_keys
        self.value_dim = value_dim
        self.values = nn.Embedding(self.size, value_dim)
        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        return self.values.weight

    def reset_parameters(self):
        nn.init.normal_(self.values.weight, mean=0.0, std=self.value_dim ** -0.5)


class ProductKeyMemoryLayer(nn.Module):
    """Product-key memory lookup + weighted value retrieval."""

    def __init__(
        self,
        d_model: int,
        mem_n_keys: int,
        mem_heads: int,
        mem_knn: int,
        key_dim: int,
        value_dim: int = -1,
        mem_q_rank: int | None = None,
        value_store: ProductKeyValueStore | None = None,
        memory_plus: bool = False,
        query_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert key_dim % 2 == 0, "key_dim must be even for product quantization"
        assert mem_heads >= 1
        self.d_model = d_model
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.mem_knn = min(mem_knn, mem_n_keys)
        self.key_dim = key_dim
        self.value_dim = d_model if value_dim < 0 else value_dim
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else key_dim
        if self.mem_q_rank < 1:
            raise ValueError(f"mem_q_rank must be >= 1, got {self.mem_q_rank}")
        self.memory_plus = memory_plus
        self.qk_norm = qk_norm

        half = key_dim // 2
        self.keys = nn.Parameter(torch.empty(2, mem_heads, mem_n_keys, half))
        self.value_store = (
            value_store if value_store is not None
            else ProductKeyValueStore(mem_n_keys, self.value_dim)
        )
        self.query_down = nn.Linear(d_model, self.mem_q_rank, bias=query_bias)
        self.query_up = nn.Linear(self.mem_q_rank, mem_heads * key_dim, bias=False)

        if memory_plus:
            self.swilu_projection = nn.Linear(d_model, self.value_dim, bias=False)
            self.value_proj = nn.Linear(self.value_dim, d_model, bias=False)
        elif self.value_dim != d_model:
            self.swilu_projection = None
            self.value_proj = nn.Linear(self.value_dim, d_model, bias=False)
        else:
            self.swilu_projection = None
            self.value_proj = None

        self.reset_product_key_parameters(reset_values=value_store is None)

    def reset_product_key_parameters(self, reset_values: bool = False):
        bound = 1.0 / math.sqrt(self.key_dim)
        nn.init.uniform_(self.keys, -bound, bound)
        nn.init.xavier_uniform_(self.query_down.weight)
        if self.query_down.bias is not None:
            nn.init.zeros_(self.query_down.bias)
        nn.init.xavier_uniform_(self.query_up.weight)
        if self.value_proj is not None:
            nn.init.normal_(self.value_proj.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.swilu_projection is not None:
            nn.init.normal_(self.swilu_projection.weight, mean=0.0, std=self.d_model ** -0.5)
        if reset_values:
            self.value_store.reset_parameters()

    def get_indices(self, query: torch.Tensor, knn: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert query.dim() == 2 and query.size(1) == self.key_dim
        bs = query.size(0) // self.mem_heads
        query = query.view(bs, self.mem_heads, self.key_dim)
        half = self.key_dim // 2
        q1 = query[..., :half]
        q2 = query[..., half:]
        k1 = self.keys[0]
        k2 = self.keys[1]
        if self.qk_norm:
            q1 = _rms_norm(q1)
            q2 = _rms_norm(q2)
            k1 = _rms_norm(k1)
            k2 = _rms_norm(k2)
        scores1 = torch.einsum("bhd,hnd->bhn", q1.float(), k1.float())
        scores2 = torch.einsum("bhd,hnd->bhn", q2.float(), k2.float())
        scores1, indices1 = torch.topk(scores1, k=knn, dim=-1, largest=True, sorted=True)
        scores2, indices2 = torch.topk(scores2, k=knn, dim=-1, largest=True, sorted=True)

        all_scores = (
            scores1.unsqueeze(-1).expand(bs, self.mem_heads, knn, knn)
            + scores2.unsqueeze(-2).expand(bs, self.mem_heads, knn, knn)
        ).reshape(bs, self.mem_heads, knn * knn)
        all_indices = (
            indices1.unsqueeze(-1).expand(bs, self.mem_heads, knn, knn) * self.mem_n_keys
            + indices2.unsqueeze(-2).expand(bs, self.mem_heads, knn, knn)
        ).reshape(bs, self.mem_heads, knn * knn)
        scores, best = torch.topk(all_scores, k=knn, dim=-1, largest=True, sorted=True)
        indices = all_indices.gather(-1, best)
        return scores.reshape(bs * self.mem_heads, knn), indices.reshape(bs * self.mem_heads, knn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x_flat = x.reshape(b * t, self.d_model)
        query_latent = self.query_down(x_flat)
        query = self.query_up(query_latent).view(b * t, self.mem_heads, self.key_dim)
        scores, indices = self.get_indices(
            query.reshape(b * t * self.mem_heads, self.key_dim),
            self.mem_knn,
        )
        scores = F.softmax(scores.float(), dim=-1).to(self.value_store.weight.dtype)
        indices = indices.view(b * t, self.mem_heads * self.mem_knn)
        scores = scores.view(b * t, self.mem_heads * self.mem_knn)
        y = F.embedding_bag(
            indices,
            self.value_store.weight,
            per_sample_weights=scores,
            mode="sum",
        )
        if self.memory_plus:
            y = self.value_proj(y * F.silu(self.swilu_projection(x_flat)))
        elif self.value_proj is not None:
            y = self.value_proj(y)
        return y.view(b, t, -1)


class BinaryCodeValueStore(nn.Module):
    """Trainable value table indexed by binary product codes."""

    def __init__(self, total_keys: int, value_dim: int):
        super().__init__()
        if total_keys < 2:
            raise ValueError(f"total_keys must be >= 2, got {total_keys}")
        self.total_keys = total_keys
        self.value_dim = value_dim
        self.values = nn.Embedding(total_keys, value_dim)
        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        return self.values.weight

    def reset_parameters(self):
        nn.init.normal_(self.values.weight, mean=0.0, std=self.value_dim ** -0.5)


class BinaryProductCodeMemoryLayer(nn.Module):
    """m-way binary product-code memory with exact top-k DP retrieval."""

    def __init__(
        self,
        d_model: int,
        mem_n_keys: int,
        mem_heads: int,
        mem_knn: int,
        key_dim: int,
        value_dim: int = -1,
        mem_q_rank: int | None = None,
        value_store: BinaryCodeValueStore | None = None,
        memory_plus: bool = False,
        query_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        if mem_heads < 1:
            raise ValueError(f"mem_heads must be >= 1, got {mem_heads}")
        if mem_n_keys < 2:
            raise ValueError(f"mem_n_keys must be >= 2, got {mem_n_keys}")
        self.d_model = d_model
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.key_dim = key_dim
        self.value_dim = d_model if value_dim < 0 else value_dim
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else key_dim
        if self.mem_q_rank < 1:
            raise ValueError(f"mem_q_rank must be >= 1, got {self.mem_q_rank}")
        self.memory_plus = memory_plus
        self.qk_norm = qk_norm

        self.total_keys = mem_n_keys * mem_n_keys
        self.num_buckets = int(math.log2(self.total_keys))
        if (1 << self.num_buckets) != self.total_keys:
            raise ValueError(
                "Binary product-code memory requires mem_n_keys^2 to be a power of 2; "
                f"got mem_n_keys={mem_n_keys} => total_keys={self.total_keys}"
            )
        if self.key_dim % self.num_buckets != 0:
            raise ValueError(
                "Binary product-code memory requires key_dim divisible by num_buckets; "
                f"got key_dim={self.key_dim}, num_buckets={self.num_buckets}"
            )
        self.bucket_dim = self.key_dim // self.num_buckets
        self.mem_knn = min(mem_knn, self.total_keys)

        # [heads, buckets, 2, bucket_dim]
        self.keys = nn.Parameter(torch.empty(mem_heads, self.num_buckets, 2, self.bucket_dim))
        self.value_store = (
            value_store if value_store is not None
            else BinaryCodeValueStore(self.total_keys, self.value_dim)
        )
        self.query_down = nn.Linear(d_model, self.mem_q_rank, bias=query_bias)
        self.query_up = nn.Linear(self.mem_q_rank, mem_heads * key_dim, bias=False)

        if memory_plus:
            self.swilu_projection = nn.Linear(d_model, self.value_dim, bias=False)
            self.value_proj = nn.Linear(self.value_dim, d_model, bias=False)
        elif self.value_dim != d_model:
            self.swilu_projection = None
            self.value_proj = nn.Linear(self.value_dim, d_model, bias=False)
        else:
            self.swilu_projection = None
            self.value_proj = None

        self.reset_binary_key_parameters(reset_values=value_store is None)

    def reset_binary_key_parameters(self, reset_values: bool = False):
        bound = 1.0 / math.sqrt(self.bucket_dim)
        nn.init.uniform_(self.keys, -bound, bound)
        nn.init.xavier_uniform_(self.query_down.weight)
        if self.query_down.bias is not None:
            nn.init.zeros_(self.query_down.bias)
        nn.init.xavier_uniform_(self.query_up.weight)
        if self.value_proj is not None:
            nn.init.normal_(self.value_proj.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.swilu_projection is not None:
            nn.init.normal_(self.swilu_projection.weight, mean=0.0, std=self.d_model ** -0.5)
        if reset_values:
            self.value_store.reset_parameters()

    def get_indices(self, query: torch.Tensor, knn: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert query.dim() == 2 and query.size(1) == self.key_dim
        bs = query.size(0) // self.mem_heads
        query = query.view(bs, self.mem_heads, self.num_buckets, self.bucket_dim)
        keys = self.keys
        if self.qk_norm:
            query = _rms_norm(query)
            keys = _rms_norm(keys)

        local_scores = torch.einsum("bhmd,hmcd->bhmc", query.float(), keys.float())
        score0 = local_scores[..., 0]
        score1 = local_scores[..., 1]
        best_bits = score1 > score0
        best_scores = torch.maximum(score0, score1).sum(dim=-1)
        deltas = (score0 - score1).abs()

        bit_weights = (1 << torch.arange(self.num_buckets, device=query.device, dtype=torch.int64))
        best_codes = (best_bits.to(torch.int64) * bit_weights.view(1, 1, -1)).sum(dim=-1)

        rows = bs * self.mem_heads
        deltas = deltas.reshape(rows, self.num_buckets)
        best_scores = best_scores.reshape(rows, 1)
        best_codes = best_codes.reshape(rows, 1)
        knn_eff = min(knn, 1 << self.num_buckets)

        frontier_penalties = torch.zeros(rows, 1, device=query.device, dtype=torch.float32)
        frontier_masks = torch.zeros(rows, 1, device=query.device, dtype=torch.int64)
        for t in range(self.num_buckets):
            shifted_penalties = frontier_penalties + deltas[:, t:t + 1]
            shifted_masks = frontier_masks ^ bit_weights[t]
            candidate_penalties = torch.cat([frontier_penalties, shifted_penalties], dim=1)
            candidate_masks = torch.cat([frontier_masks, shifted_masks], dim=1)
            step_k = min(knn_eff, candidate_penalties.size(1))
            frontier_penalties, keep_idx = torch.topk(
                candidate_penalties,
                k=step_k,
                dim=1,
                largest=False,
                sorted=True,
            )
            frontier_masks = candidate_masks.gather(1, keep_idx)

        scores = best_scores - frontier_penalties
        indices = best_codes ^ frontier_masks
        return scores, indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x_flat = x.reshape(b * t, self.d_model)
        query_latent = self.query_down(x_flat)
        query = self.query_up(query_latent).view(b * t, self.mem_heads, self.key_dim)
        scores, indices = self.get_indices(
            query.reshape(b * t * self.mem_heads, self.key_dim),
            self.mem_knn,
        )
        scores = F.softmax(scores.float(), dim=-1).to(self.value_store.weight.dtype)
        indices = indices.view(b * t, self.mem_heads * self.mem_knn)
        scores = scores.view(b * t, self.mem_heads * self.mem_knn)
        y = F.embedding_bag(
            indices,
            self.value_store.weight,
            per_sample_weights=scores,
            mode="sum",
        )
        if self.memory_plus:
            y = self.value_proj(y * F.silu(self.swilu_projection(x_flat)))
        elif self.value_proj is not None:
            y = self.value_proj(y)
        return y.view(b, t, -1)


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


class HybridAttnResidualTransformerBlock(nn.Module):
    """AttnRes block with configurable attention and FF/memory branch."""

    def __init__(
        self,
        d_model: int,
        layer_number: int,
        attn: nn.Module,
        ff: nn.Module | None,
        memory_layer: nn.Module | None,
        block_size: int = 8,
    ):
        super().__init__()
        if ff is None and memory_layer is None:
            raise ValueError("Either ff or memory_layer must be provided")
        self.layer_number = layer_number
        self.block_size = block_size
        self.boundary_interval = max(1, block_size // 2)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = ff
        self.memory_layer = memory_layer
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

        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
        layer_num = self.layer_number if execution_layer_number is None else execution_layer_number
        if layer_num > 0 and (layer_num % self.boundary_interval == 0):
            blocks.append(partial_block)
            partial_block = None

        attn_out = self.attn(self.ln1(h), causal_mask, attention_mask, rope, positions)
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        ff_in = self.ln2(h)
        ff_out = self.memory_layer(ff_in) if self.memory_layer is not None else self.ff(ff_in)
        partial_block = partial_block + ff_out
        return blocks, partial_block

    def convert_full_to_svd(self):
        if hasattr(self.attn, "convert_full_to_svd"):
            self.attn.convert_full_to_svd()
        if self.ff is not None and hasattr(self.ff, "convert_full_to_svd"):
            self.ff.convert_full_to_svd()


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, dropout: float, ffn_top_k: int | None = None,
                 attention_type: str = "mha", mla_kwargs: dict | None = None,
                 monarch_kwargs: dict | None = None, svd_kwargs: dict | None = None,
                 memory_layer: nn.Module | None = None):
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
        elif attention_type == "mla_monarch":
            if mla_kwargs is None:
                raise ValueError("mla_kwargs must be provided when attention_type='mla_monarch'")
            if monarch_kwargs is None:
                monarch_kwargs = {}
            self.attn = MultiLatentAttentionMonarch(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                **mla_kwargs,
                **monarch_kwargs,
            )
        elif attention_type == "mla_svd":
            if mla_kwargs is None:
                raise ValueError("mla_kwargs must be provided when attention_type='mla_svd'")
            if svd_kwargs is None:
                svd_kwargs = {}
            self.attn = MultiLatentAttentionSVD(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                **mla_kwargs,
                **svd_kwargs,
            )
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.memory_layer = memory_layer
        if memory_layer is not None:
            self.ff = None
        elif ffn_top_k is not None:
            self.ff = TopKSwiGLUFF(d_model, d_ff, top_k=ffn_top_k)
        else:
            self.ff = SwiGLUFF(d_model, d_ff)
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        ff_in = self.ln2(x)
        x = x + (self.memory_layer(ff_in) if self.memory_layer is not None else self.ff(ff_in))
        return x

    def convert_full_to_svd(self):
        if hasattr(self.attn, "convert_full_to_svd"):
            self.attn.convert_full_to_svd()
        if self.ff is not None and hasattr(self.ff, "convert_full_to_svd"):
            self.ff.convert_full_to_svd()


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


class HotColdMLATransformer(SimpleTransformer):
    """MLA attention with dense->hot/cold mid-training switch."""

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
        hot_token_k: int = 2000,
        cold_latent_dim: int = 128,
        hot_token_cache_path: str = DEFAULT_HOT_TOKEN_CACHE_PATH,
        svd_switch_fraction: float = 0.5,
    ):
        nn.Module.__init__(self)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads

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
        self.n_kv_heads = 1
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.weight_tied = True
        self.hot_token_k = hot_token_k
        self.cold_latent_dim = cold_latent_dim
        self.hot_token_cache_path = hot_token_cache_path
        if not (0.0 <= svd_switch_fraction <= 1.0):
            raise ValueError(f"svd_switch_fraction must be in [0, 1], got {svd_switch_fraction}")
        self.svd_switch_fraction = svd_switch_fraction
        self.register_buffer("uses_hotcold_flag", torch.tensor(False, dtype=torch.bool), persistent=True)

        hot_token_ids = _load_hot_token_ids(
            cache_path=hot_token_cache_path,
            vocab_size=vocab_size,
            hot_token_k=hot_token_k,
        )
        self.full_token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb = HotColdTiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            hot_token_ids=hot_token_ids,
            cold_latent_dim=cold_latent_dim,
        )
        self.dropout = nn.Dropout(dropout)

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
        self._init_weights(n_layers)

    def convert_full_to_hotcold_svd(self):
        """Initialize hot/cold factors from dense embedding and enable compressed path."""
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            full_weight = self.full_token_emb.weight.detach()
            hot_ids = self.token_emb.hot_token_ids
            cold_ids = self.token_emb.cold_token_ids

            self.token_emb.hot_emb.weight.copy_(full_weight[hot_ids])

            cold_full = full_weight[cold_ids].float()
            U, S, Vh = torch.linalg.svd(cold_full, full_matrices=False)
            rank = min(self.cold_latent_dim, S.numel())
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            cold_u = U_r * S_r.unsqueeze(0)

            self.token_emb.cold_emb_u.weight.zero_()
            self.token_emb.cold_emb_u.weight[:, :rank].copy_(
                cold_u.to(self.token_emb.cold_emb_u.weight.dtype)
            )

            self.token_emb.cold_latent_to_model.weight.zero_()
            self.token_emb.cold_latent_to_model.weight[:, :rank].copy_(
                Vh_r.T.to(self.token_emb.cold_latent_to_model.weight.dtype)
            )

            self.full_token_emb.weight.requires_grad_(False)
            self.uses_hotcold_flag.fill_(True)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        if bool(self.uses_hotcold_flag.item()):
            x = self.token_emb(input_ids)
        else:
            x = self.full_token_emb(input_ids)
        x = self.dropout(x)

        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)

        x = self.ln_f(x)
        if bool(self.uses_hotcold_flag.item()):
            return self.token_emb.logits(x)
        return F.linear(x, self.full_token_emb.weight)

    def token_partition_masks(self, token_ids: torch.Tensor):
        hot_mask = self.token_emb.token_is_hot(token_ids)
        cold_mask = ~hot_mask
        return hot_mask, cold_mask

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
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
        if not bool(self.uses_hotcold_flag.item()):
            lm_head_bytes = V * d * wb
            if self.weight_tied and not count_reuse:
                embedding_bytes = 0
            else:
                embedding_bytes = d * wb
        else:
            r = self.cold_latent_dim
            n_hot = self.token_emb.num_hot_tokens
            n_cold = self.token_emb.num_cold_tokens
            lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
            if self.weight_tied and not count_reuse:
                embedding_bytes = 0
            else:
                embedding_bytes = max(d, r + d * r) * wb

        attn_q_bytes = L * (
            d * dcp
            + dcp * (h * d_nope)
            + dcp * (h * d_rope)
        ) * wb
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
            model_name="hotcold_mla",
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
                f"phase={'hotcold' if bool(self.uses_hotcold_flag.item()) else 'dense'}; "
                f"MLA dc={dc}, dcp={dcp}, qk_nope={d_nope}, qk_rope={d_rope}, "
                f"v_head_dim={d_v}; hot={self.token_emb.num_hot_tokens}, "
                f"cold={self.token_emb.num_cold_tokens}, cold_rank={self.cold_latent_dim}"
            ),
        )


class MLATwoStageSVDMemoryMonarchTransformer(SimpleTransformer):
    """MLA + two-stage dense->hot/cold SVD + 12 all-memory layers + SVD O-proj."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        kv_lora_rank: int | None = None,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        hot_token_k: int = 2000,
        cold_latent_dim: int = 128,
        hot_token_cache_path: str = DEFAULT_HOT_TOKEN_CACHE_PATH,
        svd_switch_fraction: float = 0.5,
        monarch_block_size: int = 32,
        svd_attn_rank: int | None = None,
        memory_layers: int | list[int] | tuple[int, ...] = 12,
        mem_n_keys: int = 256,
        mem_heads: int = 4,
        mem_knn: int = 32,
        mem_k_dim: int | None = None,
        mem_v_dim: int | None = None,
        mem_q_rank: int | None = None,
        mem_share_values: bool = True,
        qk_norm: bool = False,
    ):
        nn.Module.__init__(self)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads
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
        self.n_kv_heads = 1
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.weight_tied = True

        self.hot_token_k = hot_token_k
        self.cold_latent_dim = cold_latent_dim
        self.hot_token_cache_path = hot_token_cache_path
        if not (0.0 <= svd_switch_fraction <= 1.0):
            raise ValueError(f"svd_switch_fraction must be in [0, 1], got {svd_switch_fraction}")
        self.svd_switch_fraction = svd_switch_fraction
        self.register_buffer("uses_hotcold_flag", torch.tensor(False, dtype=torch.bool), persistent=True)

        self.monarch_block_size = monarch_block_size  # kept for CLI compatibility; unused
        self.svd_attn_rank = svd_attn_rank if svd_attn_rank is not None else max(16, d_model // 2)
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.mem_knn = mem_knn
        self.mem_k_dim = mem_k_dim if mem_k_dim is not None else d_model
        self.mem_v_dim = mem_v_dim if mem_v_dim is not None else d_model
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else max(16, d_model // 4)
        self.mem_share_values = mem_share_values
        self.qk_norm = qk_norm
        if self.n_layers != 12:
            raise ValueError(f"This architecture is fixed to n_layers=12, got {self.n_layers}")
        if isinstance(memory_layers, int):
            if memory_layers != 12:
                raise ValueError(
                    f"This architecture requires all 12 layers to be memory layers, got memory_layers={memory_layers}"
                )
        else:
            requested = sorted(set(int(i) for i in memory_layers))
            expected = list(range(self.n_layers))
            if requested != expected:
                raise ValueError(
                    "This architecture requires all layers to be memory layers; "
                    f"expected {expected}, got {requested}"
                )
        self.memory_layer_indices = list(range(self.n_layers))
        if len(self.memory_layer_indices) != 12:
            raise ValueError(
                f"This architecture requires exactly 12 memory layers, got {len(self.memory_layer_indices)}"
            )

        hot_token_ids = _load_hot_token_ids(
            cache_path=hot_token_cache_path,
            vocab_size=vocab_size,
            hot_token_k=hot_token_k,
        )
        self.full_token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb = HotColdTiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            hot_token_ids=hot_token_ids,
            cold_latent_dim=cold_latent_dim,
        )
        self.dropout = nn.Dropout(dropout)
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
        self.shared_value_store = (
            ProductKeyValueStore(self.mem_n_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            memory_layer = None
            if i in self.memory_layer_indices:
                memory_layer = ProductKeyMemoryLayer(
                    d_model=d_model,
                    mem_n_keys=self.mem_n_keys,
                    mem_heads=self.mem_heads,
                    mem_knn=self.mem_knn,
                    key_dim=self.mem_k_dim,
                    value_dim=self.mem_v_dim,
                    mem_q_rank=self.mem_q_rank,
                    value_store=self.shared_value_store,
                    memory_plus=False,
                    qk_norm=self.qk_norm,
                )
            self.layers.append(
                TransformerBlock(
                    d_model,
                    n_heads,
                    n_heads,
                    d_ff,
                    dropout,
                    attention_type="mla_svd",
                    mla_kwargs=mla_kwargs,
                    svd_kwargs={"svd_rank": self.svd_attn_rank},
                    memory_layer=memory_layer,
                )
            )
        self.ln_f = nn.LayerNorm(d_model)
        self._init_weights(n_layers)

    def convert_full_to_hotcold_svd(self):
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            full_weight = self.full_token_emb.weight.detach()
            hot_ids = self.token_emb.hot_token_ids
            cold_ids = self.token_emb.cold_token_ids
            self.token_emb.hot_emb.weight.copy_(full_weight[hot_ids])

            cold_full = full_weight[cold_ids].float()
            U, S, Vh = torch.linalg.svd(cold_full, full_matrices=False)
            rank = min(self.cold_latent_dim, S.numel())
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            cold_u = U_r * S_r.unsqueeze(0)

            self.token_emb.cold_emb_u.weight.zero_()
            self.token_emb.cold_emb_u.weight[:, :rank].copy_(
                cold_u.to(self.token_emb.cold_emb_u.weight.dtype)
            )
            self.token_emb.cold_latent_to_model.weight.zero_()
            self.token_emb.cold_latent_to_model.weight[:, :rank].copy_(
                Vh_r.T.to(self.token_emb.cold_latent_to_model.weight.dtype)
            )
            self.full_token_emb.weight.requires_grad_(False)
            for layer in self.layers:
                layer.convert_full_to_svd()
            self.uses_hotcold_flag.fill_(True)

    def token_partition_masks(self, token_ids: torch.Tensor):
        hot_mask = self.token_emb.token_is_hot(token_ids)
        cold_mask = ~hot_mask
        return hot_mask, cold_mask

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        if bool(self.uses_hotcold_flag.item()):
            x = self.token_emb(input_ids)
        else:
            x = self.full_token_emb(input_ids)
        x = self.dropout(x)
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        x = self.ln_f(x)
        if bool(self.uses_hotcold_flag.item()):
            return self.token_emb.logits(x)
        return F.linear(x, self.full_token_emb.weight)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
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
        M = len(self.memory_layer_indices)
        dense_L = L - M

        if bool(self.uses_hotcold_flag.item()):
            r = self.cold_latent_dim
            n_hot = self.token_emb.num_hot_tokens
            n_cold = self.token_emb.num_cold_tokens
            lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else max(d, r + d * r) * wb
        else:
            lm_head_bytes = V * d * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else d * wb

        attn_q_bytes = L * (d * dcp + dcp * (h * d_nope) + dcp * (h * d_rope)) * wb
        attn_k_bytes = L * (d * dc + dc * (h * d_nope) + d * d_rope) * wb
        attn_v_bytes = L * (dc * (h * d_v)) * wb

        attn_proj_numel = 0
        for layer in self.layers:
            attn = getattr(layer, "attn", None)
            proj = getattr(attn, "proj", None)
            if proj is not None:
                if hasattr(proj, "active_weight_numel"):
                    attn_proj_numel += int(proj.active_weight_numel())
                else:
                    attn_proj_numel += sum(p.numel() for p in proj.parameters())
        attn_o_bytes = attn_proj_numel * wb

        dense_ffn_bytes = dense_L * (d * self.d_ff + self.d_ff * d + d * self.d_ff) * wb
        memory_query_bytes = M * (
            d * self.mem_q_rank + self.mem_q_rank * (self.mem_heads * self.mem_k_dim)
        ) * wb
        memory_key_bytes = M * (self.mem_heads * self.mem_n_keys * self.mem_k_dim) * wb
        memory_value_bytes = M * (self.mem_heads * self.mem_knn * self.mem_v_dim) * wb
        memory_proj_bytes = M * (self.mem_v_dim * d) * wb if self.mem_v_dim != d else 0
        ffn_bytes = (
            dense_ffn_bytes
            + memory_query_bytes
            + memory_key_bytes
            + memory_value_bytes
            + memory_proj_bytes
        )

        norm_bytes = (2 * L + 1) * 2 * d * wb
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
            model_name="mla_twostage_svd_mem12_monarch",
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
                f"phase={'hotcold' if bool(self.uses_hotcold_flag.item()) else 'dense'}; "
                f"mem_layers={self.memory_layer_indices}; svd_attn_rank={self.svd_attn_rank}; "
                f"mem_q_rank={self.mem_q_rank}"
            ),
        )


class MLATwoStageSVDBinaryMemoryMonarchTransformer(SimpleTransformer):
    """MLA + dense->hot/cold SVD + 12 binary-DP memory layers + SVD O-proj."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        kv_lora_rank: int | None = None,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        hot_token_k: int = 2000,
        cold_latent_dim: int = 128,
        hot_token_cache_path: str = DEFAULT_HOT_TOKEN_CACHE_PATH,
        svd_switch_fraction: float = 0.5,
        monarch_block_size: int = 32,
        svd_attn_rank: int | None = None,
        memory_layers: int | list[int] | tuple[int, ...] = 12,
        mem_n_keys: int = 256,
        mem_heads: int = 4,
        mem_knn: int = 32,
        mem_k_dim: int | None = None,
        mem_v_dim: int | None = None,
        mem_q_rank: int | None = None,
        mem_share_values: bool = True,
        qk_norm: bool = False,
    ):
        nn.Module.__init__(self)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads
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
        self.n_kv_heads = 1
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.weight_tied = True

        self.hot_token_k = hot_token_k
        self.cold_latent_dim = cold_latent_dim
        self.hot_token_cache_path = hot_token_cache_path
        if not (0.0 <= svd_switch_fraction <= 1.0):
            raise ValueError(f"svd_switch_fraction must be in [0, 1], got {svd_switch_fraction}")
        self.svd_switch_fraction = svd_switch_fraction
        self.register_buffer("uses_hotcold_flag", torch.tensor(False, dtype=torch.bool), persistent=True)

        self.monarch_block_size = monarch_block_size  # kept for CLI compatibility; unused
        self.svd_attn_rank = svd_attn_rank if svd_attn_rank is not None else max(16, d_model // 2)
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.mem_knn = mem_knn
        self.mem_k_dim = mem_k_dim if mem_k_dim is not None else d_model
        self.mem_v_dim = mem_v_dim if mem_v_dim is not None else d_model
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else max(16, d_model // 4)
        self.mem_share_values = mem_share_values
        self.qk_norm = qk_norm
        self.mem_binary_total_keys = self.mem_n_keys * self.mem_n_keys
        self.mem_binary_buckets = int(math.log2(self.mem_binary_total_keys))
        if (1 << self.mem_binary_buckets) != self.mem_binary_total_keys:
            raise ValueError(
                "Binary memory requires mem_n_keys^2 to be power-of-two. "
                f"Got mem_n_keys={self.mem_n_keys} => total_keys={self.mem_binary_total_keys}."
            )
        if self.mem_k_dim % self.mem_binary_buckets != 0:
            raise ValueError(
                f"mem_k_dim must be divisible by num_binary_buckets={self.mem_binary_buckets}; "
                f"got mem_k_dim={self.mem_k_dim}."
            )
        if self.n_layers != 12:
            raise ValueError(f"This architecture is fixed to n_layers=12, got {self.n_layers}")
        if isinstance(memory_layers, int):
            if memory_layers != 12:
                raise ValueError(
                    f"This architecture requires all 12 layers to be memory layers, got memory_layers={memory_layers}"
                )
        else:
            requested = sorted(set(int(i) for i in memory_layers))
            expected = list(range(self.n_layers))
            if requested != expected:
                raise ValueError(
                    "This architecture requires all layers to be memory layers; "
                    f"expected {expected}, got {requested}"
                )
        self.memory_layer_indices = list(range(self.n_layers))
        if len(self.memory_layer_indices) != 12:
            raise ValueError(
                f"This architecture requires exactly 12 memory layers, got {len(self.memory_layer_indices)}"
            )

        hot_token_ids = _load_hot_token_ids(
            cache_path=hot_token_cache_path,
            vocab_size=vocab_size,
            hot_token_k=hot_token_k,
        )
        self.full_token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb = HotColdTiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            hot_token_ids=hot_token_ids,
            cold_latent_dim=cold_latent_dim,
        )
        self.dropout = nn.Dropout(dropout)
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
        self.shared_value_store = (
            BinaryCodeValueStore(self.mem_binary_total_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            memory_layer = None
            if i in self.memory_layer_indices:
                memory_layer = BinaryProductCodeMemoryLayer(
                    d_model=d_model,
                    mem_n_keys=self.mem_n_keys,
                    mem_heads=self.mem_heads,
                    mem_knn=self.mem_knn,
                    key_dim=self.mem_k_dim,
                    value_dim=self.mem_v_dim,
                    mem_q_rank=self.mem_q_rank,
                    value_store=self.shared_value_store,
                    memory_plus=False,
                    qk_norm=self.qk_norm,
                )
            self.layers.append(
                TransformerBlock(
                    d_model,
                    n_heads,
                    n_heads,
                    d_ff,
                    dropout,
                    attention_type="mla_svd",
                    mla_kwargs=mla_kwargs,
                    svd_kwargs={"svd_rank": self.svd_attn_rank},
                    memory_layer=memory_layer,
                )
            )
        self.ln_f = nn.LayerNorm(d_model)
        self._init_weights(n_layers)

    def convert_full_to_hotcold_svd(self):
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            full_weight = self.full_token_emb.weight.detach()
            hot_ids = self.token_emb.hot_token_ids
            cold_ids = self.token_emb.cold_token_ids
            self.token_emb.hot_emb.weight.copy_(full_weight[hot_ids])

            cold_full = full_weight[cold_ids].float()
            U, S, Vh = torch.linalg.svd(cold_full, full_matrices=False)
            rank = min(self.cold_latent_dim, S.numel())
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            cold_u = U_r * S_r.unsqueeze(0)

            self.token_emb.cold_emb_u.weight.zero_()
            self.token_emb.cold_emb_u.weight[:, :rank].copy_(
                cold_u.to(self.token_emb.cold_emb_u.weight.dtype)
            )
            self.token_emb.cold_latent_to_model.weight.zero_()
            self.token_emb.cold_latent_to_model.weight[:, :rank].copy_(
                Vh_r.T.to(self.token_emb.cold_latent_to_model.weight.dtype)
            )
            self.full_token_emb.weight.requires_grad_(False)
            for layer in self.layers:
                layer.convert_full_to_svd()
            self.uses_hotcold_flag.fill_(True)

    def token_partition_masks(self, token_ids: torch.Tensor):
        hot_mask = self.token_emb.token_is_hot(token_ids)
        cold_mask = ~hot_mask
        return hot_mask, cold_mask

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        if bool(self.uses_hotcold_flag.item()):
            x = self.token_emb(input_ids)
        else:
            x = self.full_token_emb(input_ids)
        x = self.dropout(x)
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        x = self.ln_f(x)
        if bool(self.uses_hotcold_flag.item()):
            return self.token_emb.logits(x)
        return F.linear(x, self.full_token_emb.weight)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
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
        M = len(self.memory_layer_indices)
        dense_L = L - M

        if bool(self.uses_hotcold_flag.item()):
            r = self.cold_latent_dim
            n_hot = self.token_emb.num_hot_tokens
            n_cold = self.token_emb.num_cold_tokens
            lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else max(d, r + d * r) * wb
        else:
            lm_head_bytes = V * d * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else d * wb

        attn_q_bytes = L * (d * dcp + dcp * (h * d_nope) + dcp * (h * d_rope)) * wb
        attn_k_bytes = L * (d * dc + dc * (h * d_nope) + d * d_rope) * wb
        attn_v_bytes = L * (dc * (h * d_v)) * wb

        attn_proj_numel = 0
        for layer in self.layers:
            attn = getattr(layer, "attn", None)
            proj = getattr(attn, "proj", None)
            if proj is not None:
                if hasattr(proj, "active_weight_numel"):
                    attn_proj_numel += int(proj.active_weight_numel())
                else:
                    attn_proj_numel += sum(p.numel() for p in proj.parameters())
        attn_o_bytes = attn_proj_numel * wb

        dense_ffn_bytes = dense_L * (d * self.d_ff + self.d_ff * d + d * self.d_ff) * wb
        memory_query_bytes = M * (
            d * self.mem_q_rank + self.mem_q_rank * (self.mem_heads * self.mem_k_dim)
        ) * wb
        # Binary buckets store exactly two choices per factor.
        memory_key_bytes = M * (self.mem_heads * 2 * self.mem_k_dim) * wb
        memory_value_bytes = M * (self.mem_heads * self.mem_knn * self.mem_v_dim) * wb
        memory_proj_bytes = M * (self.mem_v_dim * d) * wb if self.mem_v_dim != d else 0
        ffn_bytes = (
            dense_ffn_bytes
            + memory_query_bytes
            + memory_key_bytes
            + memory_value_bytes
            + memory_proj_bytes
        )

        norm_bytes = (2 * L + 1) * 2 * d * wb
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
            model_name="mla_twostage_svd_mem12_binarydp",
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
                f"phase={'hotcold' if bool(self.uses_hotcold_flag.item()) else 'dense'}; "
                f"mem_layers={self.memory_layer_indices}; svd_attn_rank={self.svd_attn_rank}; "
                f"binary_total_keys={self.mem_binary_total_keys}; binary_buckets={self.mem_binary_buckets}; "
                f"mem_q_rank={self.mem_q_rank}"
            ),
        )


class DPSharedMemoryTransformer(SimpleTransformer):
    """Standard MHA transformer with binary-DP memory layers and shared values."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        monarch_block_size: int = 32,  # kept for CLI compatibility; unused
        memory_layers: int | list[int] | tuple[int, ...] = 12,
        mem_n_keys: int = 256,
        mem_heads: int = 4,
        mem_knn: int = 32,
        mem_k_dim: int | None = None,
        mem_v_dim: int | None = None,
        mem_q_rank: int | None = None,
        mem_share_values: bool = True,
        qk_norm: bool = False,
    ):
        nn.Module.__init__(self)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.weight_tied = True

        self.monarch_block_size = monarch_block_size
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.mem_knn = mem_knn
        self.mem_k_dim = mem_k_dim if mem_k_dim is not None else d_model
        self.mem_v_dim = mem_v_dim if mem_v_dim is not None else d_model
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else max(16, d_model // 4)
        self.mem_share_values = mem_share_values
        self.qk_norm = qk_norm
        self.mem_binary_total_keys = self.mem_n_keys * self.mem_n_keys
        self.mem_binary_buckets = int(math.log2(self.mem_binary_total_keys))
        if (1 << self.mem_binary_buckets) != self.mem_binary_total_keys:
            raise ValueError(
                "Binary memory requires mem_n_keys^2 to be power-of-two. "
                f"Got mem_n_keys={self.mem_n_keys} => total_keys={self.mem_binary_total_keys}."
            )
        if self.mem_k_dim % self.mem_binary_buckets != 0:
            raise ValueError(
                f"mem_k_dim must be divisible by num_binary_buckets={self.mem_binary_buckets}; "
                f"got mem_k_dim={self.mem_k_dim}."
            )

        if isinstance(memory_layers, int):
            if memory_layers < 1 or memory_layers > self.n_layers:
                raise ValueError(
                    f"memory_layers as int must be in [1, n_layers], got {memory_layers} with n_layers={self.n_layers}"
                )
            self.memory_layer_indices = list(range(self.n_layers - memory_layers, self.n_layers))
        else:
            requested = sorted(set(int(i) for i in memory_layers))
            if not requested:
                raise ValueError("memory_layers list cannot be empty")
            if requested[0] < 0 or requested[-1] >= self.n_layers:
                raise ValueError(
                    f"memory_layers indices must be within [0, {self.n_layers - 1}], got {requested}"
                )
            self.memory_layer_indices = requested

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.head_dim,
            max_seq_len=max_seq_len,
        )
        self.shared_value_store = (
            BinaryCodeValueStore(self.mem_binary_total_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            memory_layer = None
            if i in self.memory_layer_indices:
                memory_layer = BinaryProductCodeMemoryLayer(
                    d_model=d_model,
                    mem_n_keys=self.mem_n_keys,
                    mem_heads=self.mem_heads,
                    mem_knn=self.mem_knn,
                    key_dim=self.mem_k_dim,
                    value_dim=self.mem_v_dim,
                    mem_q_rank=self.mem_q_rank,
                    value_store=self.shared_value_store,
                    memory_plus=False,
                    qk_norm=self.qk_norm,
                )
            self.layers.append(
                TransformerBlock(
                    d_model,
                    n_heads,
                    n_heads,
                    d_ff,
                    dropout,
                    memory_layer=memory_layer,
                )
            )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self._init_weights(n_layers)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        x = self.dropout(x)
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        x = self.ln_f(x)
        return self.head(x)

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
        M = len(self.memory_layer_indices)
        dense_L = L - M

        lm_head_bytes = V * d * wb
        if self.weight_tied and not count_reuse:
            embedding_bytes = 0
        else:
            embedding_bytes = d * wb

        attn_q_bytes = L * d * (h * hd) * wb
        attn_k_bytes = L * d * (kv_h * hd) * wb
        attn_v_bytes = L * d * (kv_h * hd) * wb
        attn_o_bytes = L * (h * hd) * d * wb

        dense_ffn_bytes = dense_L * (d * self.d_ff + self.d_ff * d + d * self.d_ff) * wb
        memory_query_bytes = M * (
            d * self.mem_q_rank + self.mem_q_rank * (self.mem_heads * self.mem_k_dim)
        ) * wb
        memory_key_bytes = M * (self.mem_heads * 2 * self.mem_k_dim) * wb
        memory_value_bytes = M * (self.mem_heads * self.mem_knn * self.mem_v_dim) * wb
        memory_proj_bytes = M * (self.mem_v_dim * d) * wb if self.mem_v_dim != d else 0
        ffn_bytes = (
            dense_ffn_bytes
            + memory_query_bytes
            + memory_key_bytes
            + memory_value_bytes
            + memory_proj_bytes
        )

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
            model_name="dp_shared_memory",
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
            notes=(
                f"mem_layers={self.memory_layer_indices}; "
                f"binary_total_keys={self.mem_binary_total_keys}; "
                f"binary_buckets={self.mem_binary_buckets}; "
                f"shared_values={self.mem_share_values}; "
                f"mem_q_rank={self.mem_q_rank}"
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


class MLAHybridLoop12Transformer(SimpleTransformer):
    """12-layer MLA hybrid with layerwise memory/SVD/Monarch schedule + top-loop."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        kv_lora_rank: int | None = None,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        hot_token_k: int = 2000,
        cold_latent_dim: int = 128,
        hot_token_cache_path: str = DEFAULT_HOT_TOKEN_CACHE_PATH,
        svd_switch_fraction: float = 1.0 / 3.0,
        monarch_block_size: int = 32,
        memory_layers: int | list[int] | tuple[int, ...] = 12,
        mem_n_keys: int = 256,
        mem_heads: int = 4,
        mem_knn: int = 32,
        mem_k_dim: int | None = None,
        mem_v_dim: int | None = None,
        mem_q_rank: int | None = None,
        mem_share_values: bool = True,
        qk_norm: bool = False,
        svd_ffn_rank: int | None = None,
        svd_attn_rank: int | None = None,
        loop_block_size: int = 4,
        loop_repeats: int = 3,
        attn_res_block_size: int = 4,
    ):
        nn.Module.__init__(self)
        if n_layers != 12:
            raise ValueError(f"mla_hybrid_loop12 requires n_layers=12, got {n_layers}")
        if not (0.0 <= svd_switch_fraction <= 1.0):
            raise ValueError(f"svd_switch_fraction must be in [0, 1], got {svd_switch_fraction}")
        if loop_block_size != 4:
            raise ValueError(f"mla_hybrid_loop12 requires loop_block_size=4, got {loop_block_size}")
        if loop_repeats != 3:
            raise ValueError(f"mla_hybrid_loop12 requires loop_repeats=3, got {loop_repeats}")
        if isinstance(memory_layers, int):
            if memory_layers != 12:
                raise ValueError(
                    "mla_hybrid_loop12 uses a fixed 12-layer schedule; "
                    f"expected memory_layers=12, got {memory_layers}"
                )
        else:
            requested = sorted(set(int(i) for i in memory_layers))
            expected = list(range(12))
            if requested != expected:
                raise ValueError(
                    "mla_hybrid_loop12 expects memory_layers to cover all 12 indices "
                    f"for compatibility with shared CLI args; expected {expected}, got {requested}"
                )

        head_dim = d_model // n_heads
        default_rope_dim = max(2, (head_dim // 2) * 2)
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else max(16, d_model // 8)
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else max(16, d_model // 8)
        self.qk_nope_head_dim = qk_nope_head_dim if qk_nope_head_dim is not None else head_dim
        self.qk_rope_head_dim = qk_rope_head_dim if qk_rope_head_dim is not None else default_rope_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = 1
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.weight_tied = True

        self.hot_token_k = hot_token_k
        self.cold_latent_dim = cold_latent_dim
        self.hot_token_cache_path = hot_token_cache_path
        self.svd_switch_fraction = svd_switch_fraction
        self.monarch_block_size = monarch_block_size
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.mem_knn = mem_knn
        self.mem_k_dim = mem_k_dim if mem_k_dim is not None else d_model
        self.mem_v_dim = mem_v_dim if mem_v_dim is not None else d_model
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else max(16, d_model // 4)
        self.mem_share_values = mem_share_values
        self.qk_norm = qk_norm
        self.svd_ffn_rank = svd_ffn_rank if svd_ffn_rank is not None else max(16, min(d_model, d_ff) // 2)
        self.svd_attn_rank = svd_attn_rank if svd_attn_rank is not None else max(16, d_model // 2)
        self.loop_block_size = loop_block_size
        self.loop_repeats = loop_repeats
        self.loop_start = self.n_layers - self.loop_block_size
        self.attn_res_block_size = attn_res_block_size

        self.bottom_even_memory_layers = [1, 3, 5, 7]
        self.bottom_odd_svd_ffn_layers = [2, 4, 6]
        self.top_even_monarch_layers = [8, 10]
        self.top_odd_svd_attn_layers = [9, 11]

        self.register_buffer("uses_hotcold_flag", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.register_buffer("uses_structured_svd_flag", torch.tensor(False, dtype=torch.bool), persistent=True)

        hot_token_ids = _load_hot_token_ids(
            cache_path=hot_token_cache_path,
            vocab_size=vocab_size,
            hot_token_k=hot_token_k,
        )
        self.full_token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb = HotColdTiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            hot_token_ids=hot_token_ids,
            cold_latent_dim=cold_latent_dim,
        )
        self.dropout = nn.Dropout(dropout)
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
        shared_value_store = (
            ProductKeyValueStore(self.mem_n_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i in self.top_even_monarch_layers:
                attn = MultiLatentAttentionMonarch(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    **mla_kwargs,
                    monarch_block_size=self.monarch_block_size,
                )
            elif i in self.top_odd_svd_attn_layers:
                attn = MultiLatentAttentionSVD(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    **mla_kwargs,
                    svd_rank=self.svd_attn_rank,
                )
            else:
                attn = MultiLatentAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    **mla_kwargs,
                )

            memory_layer = None
            ff = None
            if i in self.bottom_even_memory_layers:
                memory_layer = ProductKeyMemoryLayer(
                    d_model=d_model,
                    mem_n_keys=self.mem_n_keys,
                    mem_heads=self.mem_heads,
                    mem_knn=self.mem_knn,
                    key_dim=self.mem_k_dim,
                    value_dim=self.mem_v_dim,
                    mem_q_rank=self.mem_q_rank,
                    value_store=shared_value_store,
                    memory_plus=False,
                    qk_norm=self.qk_norm,
                )
            elif i in self.bottom_odd_svd_ffn_layers:
                ff = SVDSwiGLUFF(d_model=d_model, d_ff=d_ff, svd_rank=self.svd_ffn_rank)
            else:
                ff = SwiGLUFF(d_model=d_model, d_ff=d_ff)

            self.layers.append(
                HybridAttnResidualTransformerBlock(
                    d_model=d_model,
                    layer_number=i,
                    attn=attn,
                    ff=ff,
                    memory_layer=memory_layer,
                    block_size=self.attn_res_block_size,
                )
            )
        self.ln_f = nn.LayerNorm(d_model)
        self._init_weights(self.n_layers)

    @property
    def effective_layer_executions(self) -> int:
        return self.loop_start + self.loop_block_size * self.loop_repeats

    def _layer_exec_counts(self) -> list[int]:
        return [1 if i < self.loop_start else self.loop_repeats for i in range(self.n_layers)]

    def _convert_embedding_to_hotcold(self):
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            full_weight = self.full_token_emb.weight.detach()
            hot_ids = self.token_emb.hot_token_ids
            cold_ids = self.token_emb.cold_token_ids
            self.token_emb.hot_emb.weight.copy_(full_weight[hot_ids])

            cold_full = full_weight[cold_ids].float()
            U, S, Vh = torch.linalg.svd(cold_full, full_matrices=False)
            rank = min(self.cold_latent_dim, S.numel())
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            cold_u = U_r * S_r.unsqueeze(0)

            self.token_emb.cold_emb_u.weight.zero_()
            self.token_emb.cold_emb_u.weight[:, :rank].copy_(
                cold_u.to(self.token_emb.cold_emb_u.weight.dtype)
            )
            self.token_emb.cold_latent_to_model.weight.zero_()
            self.token_emb.cold_latent_to_model.weight[:, :rank].copy_(
                Vh_r.T.to(self.token_emb.cold_latent_to_model.weight.dtype)
            )
            self.full_token_emb.weight.requires_grad_(False)
            self.uses_hotcold_flag.fill_(True)

    def _convert_structured_layers_to_svd(self):
        if bool(self.uses_structured_svd_flag.item()):
            return
        for layer in self.layers:
            layer.convert_full_to_svd()
        self.uses_structured_svd_flag.fill_(True)

    def convert_full_to_hotcold_svd(self):
        # Entry point used by training loop for one-shot phase transition.
        self._convert_embedding_to_hotcold()
        self._convert_structured_layers_to_svd()

    def token_partition_masks(self, token_ids: torch.Tensor):
        hot_mask = self.token_emb.token_is_hot(token_ids)
        cold_mask = ~hot_mask
        return hot_mask, cold_mask

    def forward(self, input_ids, attention_mask=None):
        _, T = input_ids.shape
        device = input_ids.device
        if bool(self.uses_hotcold_flag.item()):
            x = self.token_emb(input_ids)
        else:
            x = self.full_token_emb(input_ids)
        x = self.dropout(x)
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        blocks = [x]
        partial = x
        exec_layer_number = 0

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
        if bool(self.uses_hotcold_flag.item()):
            return self.token_emb.logits(x)
        return F.linear(x, self.full_token_emb.weight)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
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
        exec_counts = self._layer_exec_counts()
        exec_L = sum(exec_counts)
        M = len(self.bottom_even_memory_layers)

        if bool(self.uses_hotcold_flag.item()):
            r = self.cold_latent_dim
            n_hot = self.token_emb.num_hot_tokens
            n_cold = self.token_emb.num_cold_tokens
            lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else max(d, r + d * r) * wb
        else:
            lm_head_bytes = V * d * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else d * wb

        # Count parameter reads once per physical layer for looped top-4 blocks.
        # (Looping repeats compute, but should not triple-count model weights here.)
        attn_q_bytes = L * (d * dcp + dcp * (h * d_nope) + dcp * (h * d_rope)) * wb
        attn_k_bytes = L * (d * dc + dc * (h * d_nope) + d * d_rope) * wb
        attn_v_bytes = L * (dc * (h * d_v)) * wb

        attn_o_numel = 0
        ffn_numel = 0
        for i, layer in enumerate(self.layers):
            rep = 1
            attn_proj = getattr(layer.attn, "proj", None)
            if attn_proj is not None:
                if isinstance(attn_proj, TwoStageSVDLinear):
                    attn_o_numel += rep * attn_proj.active_weight_numel()
                else:
                    attn_o_numel += rep * sum(p.numel() for p in attn_proj.parameters())
            if layer.memory_layer is not None:
                continue
            if isinstance(layer.ff, SVDSwiGLUFF):
                ffn_numel += rep * layer.ff.active_weight_numel()
            else:
                ffn_numel += rep * sum(p.numel() for p in layer.ff.parameters())
        attn_o_bytes = attn_o_numel * wb

        dense_memory_query_bytes = M * (
            d * self.mem_q_rank + self.mem_q_rank * (self.mem_heads * self.mem_k_dim)
        ) * wb
        dense_memory_key_bytes = M * (self.mem_heads * self.mem_n_keys * self.mem_k_dim) * wb
        dense_memory_value_bytes = M * (self.mem_heads * self.mem_knn * self.mem_v_dim) * wb
        dense_memory_proj_bytes = M * (self.mem_v_dim * d) * wb if self.mem_v_dim != d else 0
        ffn_bytes = (
            ffn_numel * wb
            + dense_memory_query_bytes
            + dense_memory_key_bytes
            + dense_memory_value_bytes
            + dense_memory_proj_bytes
        )

        norm_bytes = (
            (2 * L + 1) * 2 * d * wb
            + (2 * L) * d * wb
            + (2 * L) * d * wb
        )
        kv_cache_token_width = dc + d_rope
        kv_cache_read_bytes = kv_cache_token_width * seq_len * exec_L * kb
        kv_cache_write_bytes = kv_cache_token_width * exec_L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="mla_hybrid_loop12",
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
                f"phase={'hotcold+svd' if bool(self.uses_structured_svd_flag.item()) else 'dense'}; "
                f"mem_layers={self.bottom_even_memory_layers}; "
                f"svd_ffn_layers={self.bottom_odd_svd_ffn_layers}; "
                f"top_monarch={self.top_even_monarch_layers}; "
                f"top_svd={self.top_odd_svd_attn_layers}; "
                f"top4_loopx{self.loop_repeats}; weight_reads_counted_once_for_looped_layers"
            ),
        )


class MLAHybridLoop12MonarchTransformer(SimpleTransformer):
    """12-layer MLA hybrid loop variant with Monarch replacing all prior SVD paths."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 1024,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        kv_lora_rank: int | None = None,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        hot_token_k: int = 2000,
        cold_latent_dim: int = 128,
        hot_token_cache_path: str = DEFAULT_HOT_TOKEN_CACHE_PATH,
        svd_switch_fraction: float = 1.0 / 3.0,
        monarch_block_size: int = 32,
        memory_layers: int | list[int] | tuple[int, ...] = 12,
        mem_n_keys: int = 256,
        mem_heads: int = 4,
        mem_knn: int = 32,
        mem_k_dim: int | None = None,
        mem_v_dim: int | None = None,
        mem_q_rank: int | None = None,
        mem_share_values: bool = True,
        qk_norm: bool = False,
        loop_block_size: int = 4,
        loop_repeats: int = 3,
        attn_res_block_size: int = 4,
    ):
        nn.Module.__init__(self)
        if n_layers != 12:
            raise ValueError(f"mla_hybrid_loop12_monarch requires n_layers=12, got {n_layers}")
        if not (0.0 <= svd_switch_fraction <= 1.0):
            raise ValueError(f"svd_switch_fraction must be in [0, 1], got {svd_switch_fraction}")
        if loop_block_size != 4:
            raise ValueError(
                f"mla_hybrid_loop12_monarch requires loop_block_size=4, got {loop_block_size}"
            )
        if loop_repeats != 3:
            raise ValueError(f"mla_hybrid_loop12_monarch requires loop_repeats=3, got {loop_repeats}")
        if isinstance(memory_layers, int):
            if memory_layers != 12:
                raise ValueError(
                    "mla_hybrid_loop12_monarch uses a fixed 12-layer schedule; "
                    f"expected memory_layers=12, got {memory_layers}"
                )
        else:
            requested = sorted(set(int(i) for i in memory_layers))
            expected = list(range(12))
            if requested != expected:
                raise ValueError(
                    "mla_hybrid_loop12_monarch expects memory_layers to cover all 12 indices "
                    f"for compatibility with shared CLI args; expected {expected}, got {requested}"
                )
        if d_ff != d_model:
            raise ValueError(
                f"mla_hybrid_loop12_monarch requires d_ff == d_model for Monarch FFN; got d_ff={d_ff}, d_model={d_model}"
            )

        head_dim = d_model // n_heads
        default_rope_dim = max(2, (head_dim // 2) * 2)
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else max(16, d_model // 8)
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else max(16, d_model // 8)
        self.qk_nope_head_dim = qk_nope_head_dim if qk_nope_head_dim is not None else head_dim
        self.qk_rope_head_dim = qk_rope_head_dim if qk_rope_head_dim is not None else default_rope_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = 1
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.weight_tied = True
        self.hot_token_k = hot_token_k
        self.cold_latent_dim = cold_latent_dim
        self.hot_token_cache_path = hot_token_cache_path
        self.svd_switch_fraction = svd_switch_fraction
        self.monarch_block_size = monarch_block_size
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.mem_knn = mem_knn
        self.mem_k_dim = mem_k_dim if mem_k_dim is not None else d_model
        self.mem_v_dim = mem_v_dim if mem_v_dim is not None else d_model
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else max(16, d_model // 4)
        self.mem_share_values = mem_share_values
        self.qk_norm = qk_norm
        self.loop_block_size = loop_block_size
        self.loop_repeats = loop_repeats
        self.loop_start = self.n_layers - self.loop_block_size
        self.attn_res_block_size = attn_res_block_size

        self.bottom_even_memory_layers = [0, 2, 4, 6]
        self.bottom_odd_monarch_ffn_layers = [1, 3, 5, 7]
        self.top_even_monarch_layers = [8, 10]
        self.top_odd_monarch_layers = [9, 11]
        self.monarch_ffn_layers = (
            self.bottom_odd_monarch_ffn_layers
            + self.top_even_monarch_layers
            + self.top_odd_monarch_layers
        )

        self.register_buffer("uses_hotcold_flag", torch.tensor(False, dtype=torch.bool), persistent=True)

        hot_token_ids = _load_hot_token_ids(
            cache_path=hot_token_cache_path,
            vocab_size=vocab_size,
            hot_token_k=hot_token_k,
        )
        self.full_token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb = HotColdTiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            hot_token_ids=hot_token_ids,
            cold_latent_dim=cold_latent_dim,
        )
        self.dropout = nn.Dropout(dropout)
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
        shared_value_store = (
            ProductKeyValueStore(self.mem_n_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            attn = MultiLatentAttentionMonarch(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                **mla_kwargs,
                monarch_block_size=self.monarch_block_size,
            )

            memory_layer = None
            ff = None
            if i in self.bottom_even_memory_layers:
                memory_layer = ProductKeyMemoryLayer(
                    d_model=d_model,
                    mem_n_keys=self.mem_n_keys,
                    mem_heads=self.mem_heads,
                    mem_knn=self.mem_knn,
                    key_dim=self.mem_k_dim,
                    value_dim=self.mem_v_dim,
                    mem_q_rank=self.mem_q_rank,
                    value_store=shared_value_store,
                    memory_plus=False,
                    qk_norm=self.qk_norm,
                )
            elif i in self.monarch_ffn_layers:
                ff = MonarchSwiGLUFF(
                    d_model=d_model,
                    d_ff=d_ff,
                    monarch_block_size=self.monarch_block_size,
                )
            else:
                ff = SwiGLUFF(d_model=d_model, d_ff=d_ff)

            self.layers.append(
                HybridAttnResidualTransformerBlock(
                    d_model=d_model,
                    layer_number=i,
                    attn=attn,
                    ff=ff,
                    memory_layer=memory_layer,
                    block_size=self.attn_res_block_size,
                )
            )
        self.ln_f = nn.LayerNorm(d_model)
        self._init_weights(self.n_layers)

    @property
    def effective_layer_executions(self) -> int:
        return self.loop_start + self.loop_block_size * self.loop_repeats

    def _layer_exec_counts(self) -> list[int]:
        return [1 if i < self.loop_start else self.loop_repeats for i in range(self.n_layers)]

    def _convert_embedding_to_hotcold(self):
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            full_weight = self.full_token_emb.weight.detach()
            hot_ids = self.token_emb.hot_token_ids
            cold_ids = self.token_emb.cold_token_ids
            self.token_emb.hot_emb.weight.copy_(full_weight[hot_ids])

            cold_full = full_weight[cold_ids].float()
            U, S, Vh = torch.linalg.svd(cold_full, full_matrices=False)
            rank = min(self.cold_latent_dim, S.numel())
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            cold_u = U_r * S_r.unsqueeze(0)

            self.token_emb.cold_emb_u.weight.zero_()
            self.token_emb.cold_emb_u.weight[:, :rank].copy_(
                cold_u.to(self.token_emb.cold_emb_u.weight.dtype)
            )
            self.token_emb.cold_latent_to_model.weight.zero_()
            self.token_emb.cold_latent_to_model.weight[:, :rank].copy_(
                Vh_r.T.to(self.token_emb.cold_latent_to_model.weight.dtype)
            )
            self.full_token_emb.weight.requires_grad_(False)
            self.uses_hotcold_flag.fill_(True)

    def convert_full_to_hotcold_svd(self):
        # Monarch variant only swaps dense embedding/unembedding to hot/cold.
        self._convert_embedding_to_hotcold()

    def forward(self, input_ids, attention_mask=None):
        _, T = input_ids.shape
        device = input_ids.device
        if bool(self.uses_hotcold_flag.item()):
            x = self.token_emb(input_ids)
        else:
            x = self.full_token_emb(input_ids)
        x = self.dropout(x)
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        blocks = [x]
        partial = x
        exec_layer_number = 0
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
        if bool(self.uses_hotcold_flag.item()):
            return self.token_emb.logits(x)
        return F.linear(x, self.full_token_emb.weight)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
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
        exec_counts = self._layer_exec_counts()
        exec_L = sum(exec_counts)
        M = len(self.bottom_even_memory_layers)

        if bool(self.uses_hotcold_flag.item()):
            r = self.cold_latent_dim
            n_hot = self.token_emb.num_hot_tokens
            n_cold = self.token_emb.num_cold_tokens
            lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else max(d, r + d * r) * wb
        else:
            lm_head_bytes = V * d * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else d * wb

        attn_q_bytes = L * (d * dcp + dcp * (h * d_nope) + dcp * (h * d_rope)) * wb
        attn_k_bytes = L * (d * dc + dc * (h * d_nope) + d * d_rope) * wb
        attn_v_bytes = L * (dc * (h * d_v)) * wb

        attn_o_numel = 0
        ffn_numel = 0
        for layer in self.layers:
            attn = getattr(layer, "attn", None)
            proj = getattr(attn, "proj", None)
            if proj is not None:
                attn_o_numel += sum(p.numel() for p in proj.parameters())
            if layer.memory_layer is not None:
                continue
            ffn_numel += sum(p.numel() for p in layer.ff.parameters())
        attn_o_bytes = attn_o_numel * wb

        dense_memory_query_bytes = M * (
            d * self.mem_q_rank + self.mem_q_rank * (self.mem_heads * self.mem_k_dim)
        ) * wb
        dense_memory_key_bytes = M * (self.mem_heads * self.mem_n_keys * self.mem_k_dim) * wb
        dense_memory_value_bytes = M * (self.mem_heads * self.mem_knn * self.mem_v_dim) * wb
        dense_memory_proj_bytes = M * (self.mem_v_dim * d) * wb if self.mem_v_dim != d else 0
        ffn_bytes = (
            ffn_numel * wb
            + dense_memory_query_bytes
            + dense_memory_key_bytes
            + dense_memory_value_bytes
            + dense_memory_proj_bytes
        )

        norm_bytes = (
            (2 * L + 1) * 2 * d * wb
            + (2 * L) * d * wb
            + (2 * L) * d * wb
        )
        kv_cache_token_width = dc + d_rope
        kv_cache_read_bytes = kv_cache_token_width * seq_len * exec_L * kb
        kv_cache_write_bytes = kv_cache_token_width * exec_L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="mla_hybrid_loop12_monarch",
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
                f"phase={'hotcold_embed' if bool(self.uses_hotcold_flag.item()) else 'dense_embed'}; "
                f"mem_layers={self.bottom_even_memory_layers}; "
                f"monarch_ffn={self.monarch_ffn_layers}; "
                f"top_monarch_attn={self.top_even_monarch_layers + self.top_odd_monarch_layers}; "
                f"top4_loopx{self.loop_repeats}"
            ),
        )


class MLAHybridLoop12MonarchAttnSVDFfnTransformer(SimpleTransformer):
    """12-layer MLA hybrid loop with Monarch attention and switchable SVD FFNs."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 1024,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        kv_lora_rank: int | None = None,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        hot_token_k: int = 2000,
        cold_latent_dim: int = 128,
        hot_token_cache_path: str = DEFAULT_HOT_TOKEN_CACHE_PATH,
        svd_switch_fraction: float = 1.0 / 3.0,
        monarch_block_size: int = 32,
        memory_layers: int | list[int] | tuple[int, ...] = 12,
        mem_n_keys: int = 256,
        mem_heads: int = 4,
        mem_knn: int = 32,
        mem_k_dim: int | None = None,
        mem_v_dim: int | None = None,
        mem_q_rank: int | None = None,
        mem_share_values: bool = True,
        qk_norm: bool = False,
        svd_ffn_rank: int | None = None,
        loop_block_size: int = 4,
        loop_repeats: int = 3,
        attn_res_block_size: int = 4,
    ):
        nn.Module.__init__(self)
        if n_layers != 12:
            raise ValueError(
                f"mla_hybrid_loop12_monarch_attn_svd_ffn requires n_layers=12, got {n_layers}"
            )
        if not (0.0 <= svd_switch_fraction <= 1.0):
            raise ValueError(f"svd_switch_fraction must be in [0, 1], got {svd_switch_fraction}")
        if loop_block_size != 4:
            raise ValueError(
                f"mla_hybrid_loop12_monarch_attn_svd_ffn requires loop_block_size=4, got {loop_block_size}"
            )
        if loop_repeats != 3:
            raise ValueError(
                f"mla_hybrid_loop12_monarch_attn_svd_ffn requires loop_repeats=3, got {loop_repeats}"
            )
        if isinstance(memory_layers, int):
            if memory_layers != 12:
                raise ValueError(
                    "mla_hybrid_loop12_monarch_attn_svd_ffn uses a fixed 12-layer schedule; "
                    f"expected memory_layers=12, got {memory_layers}"
                )
        else:
            requested = sorted(set(int(i) for i in memory_layers))
            expected = list(range(12))
            if requested != expected:
                raise ValueError(
                    "mla_hybrid_loop12_monarch_attn_svd_ffn expects memory_layers to cover all 12 indices "
                    f"for compatibility with shared CLI args; expected {expected}, got {requested}"
                )

        head_dim = d_model // n_heads
        default_rope_dim = max(2, (head_dim // 2) * 2)
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else max(16, d_model // 8)
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else max(16, d_model // 8)
        self.qk_nope_head_dim = qk_nope_head_dim if qk_nope_head_dim is not None else head_dim
        self.qk_rope_head_dim = qk_rope_head_dim if qk_rope_head_dim is not None else default_rope_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = 1
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.weight_tied = True
        self.hot_token_k = hot_token_k
        self.cold_latent_dim = cold_latent_dim
        self.hot_token_cache_path = hot_token_cache_path
        self.svd_switch_fraction = svd_switch_fraction
        self.monarch_block_size = monarch_block_size
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.mem_knn = mem_knn
        self.mem_k_dim = mem_k_dim if mem_k_dim is not None else d_model
        self.mem_v_dim = mem_v_dim if mem_v_dim is not None else d_model
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else max(16, d_model // 4)
        self.mem_share_values = mem_share_values
        self.qk_norm = qk_norm
        self.svd_ffn_rank = svd_ffn_rank if svd_ffn_rank is not None else 96
        self.loop_block_size = loop_block_size
        self.loop_repeats = loop_repeats
        self.loop_start = self.n_layers - self.loop_block_size
        self.attn_res_block_size = attn_res_block_size

        self.bottom_even_memory_layers = [0, 2, 4, 6]
        self.bottom_odd_svd_ffn_layers = [1, 3, 5, 7]
        self.top_even_monarch_layers = [8, 10]
        self.top_odd_monarch_layers = [9, 11]
        # Keep one looped SwiGLU dense (layer 11) for a mixed loop FFN stack.
        self.loop_dense_ffn_layers = [11]
        self.svd_ffn_layers = (
            self.bottom_odd_svd_ffn_layers
            + self.top_even_monarch_layers
            + [i for i in self.top_odd_monarch_layers if i not in self.loop_dense_ffn_layers]
        )

        self.register_buffer("uses_hotcold_flag", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.register_buffer("uses_structured_svd_flag", torch.tensor(False, dtype=torch.bool), persistent=True)

        hot_token_ids = _load_hot_token_ids(
            cache_path=hot_token_cache_path,
            vocab_size=vocab_size,
            hot_token_k=hot_token_k,
        )
        self.full_token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb = HotColdTiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            hot_token_ids=hot_token_ids,
            cold_latent_dim=cold_latent_dim,
        )
        self.dropout = nn.Dropout(dropout)
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
        shared_value_store = (
            ProductKeyValueStore(self.mem_n_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            attn = MultiLatentAttentionMonarch(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                **mla_kwargs,
                monarch_block_size=self.monarch_block_size,
            )

            memory_layer = None
            ff = None
            if i in self.bottom_even_memory_layers:
                memory_layer = ProductKeyMemoryLayer(
                    d_model=d_model,
                    mem_n_keys=self.mem_n_keys,
                    mem_heads=self.mem_heads,
                    mem_knn=self.mem_knn,
                    key_dim=self.mem_k_dim,
                    value_dim=self.mem_v_dim,
                    mem_q_rank=self.mem_q_rank,
                    value_store=shared_value_store,
                    memory_plus=False,
                    qk_norm=self.qk_norm,
                )
            elif i in self.svd_ffn_layers:
                ff = SVDSwiGLUFF(
                    d_model=d_model,
                    d_ff=d_ff,
                    svd_rank=self.svd_ffn_rank,
                )
            else:
                ff = SwiGLUFF(d_model=d_model, d_ff=d_ff)

            self.layers.append(
                HybridAttnResidualTransformerBlock(
                    d_model=d_model,
                    layer_number=i,
                    attn=attn,
                    ff=ff,
                    memory_layer=memory_layer,
                    block_size=self.attn_res_block_size,
                )
            )
        self.ln_f = nn.LayerNorm(d_model)
        self._init_weights(self.n_layers)

    @property
    def effective_layer_executions(self) -> int:
        return self.loop_start + self.loop_block_size * self.loop_repeats

    def _layer_exec_counts(self) -> list[int]:
        return [1 if i < self.loop_start else self.loop_repeats for i in range(self.n_layers)]

    def _convert_embedding_to_hotcold(self):
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            full_weight = self.full_token_emb.weight.detach()
            hot_ids = self.token_emb.hot_token_ids
            cold_ids = self.token_emb.cold_token_ids
            self.token_emb.hot_emb.weight.copy_(full_weight[hot_ids])

            cold_full = full_weight[cold_ids].float()
            U, S, Vh = torch.linalg.svd(cold_full, full_matrices=False)
            rank = min(self.cold_latent_dim, S.numel())
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            cold_u = U_r * S_r.unsqueeze(0)

            self.token_emb.cold_emb_u.weight.zero_()
            self.token_emb.cold_emb_u.weight[:, :rank].copy_(
                cold_u.to(self.token_emb.cold_emb_u.weight.dtype)
            )
            self.token_emb.cold_latent_to_model.weight.zero_()
            self.token_emb.cold_latent_to_model.weight[:, :rank].copy_(
                Vh_r.T.to(self.token_emb.cold_latent_to_model.weight.dtype)
            )
            self.full_token_emb.weight.requires_grad_(False)
            self.uses_hotcold_flag.fill_(True)

    def _convert_structured_layers_to_svd(self):
        if bool(self.uses_structured_svd_flag.item()):
            return
        for layer in self.layers:
            layer.convert_full_to_svd()
        self.uses_structured_svd_flag.fill_(True)

    def convert_full_to_hotcold_svd(self):
        # One-shot phase transition: hot/cold embedding + FFN SVD factorization.
        self._convert_embedding_to_hotcold()
        self._convert_structured_layers_to_svd()

    def forward(self, input_ids, attention_mask=None):
        _, T = input_ids.shape
        device = input_ids.device
        if bool(self.uses_hotcold_flag.item()):
            x = self.token_emb(input_ids)
        else:
            x = self.full_token_emb(input_ids)
        x = self.dropout(x)
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        blocks = [x]
        partial = x
        exec_layer_number = 0
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
        if bool(self.uses_hotcold_flag.item()):
            return self.token_emb.logits(x)
        return F.linear(x, self.full_token_emb.weight)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
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
        exec_counts = self._layer_exec_counts()
        exec_L = sum(exec_counts)
        M = len(self.bottom_even_memory_layers)

        if bool(self.uses_hotcold_flag.item()):
            r = self.cold_latent_dim
            n_hot = self.token_emb.num_hot_tokens
            n_cold = self.token_emb.num_cold_tokens
            lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else max(d, r + d * r) * wb
        else:
            lm_head_bytes = V * d * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else d * wb

        attn_q_bytes = L * (d * dcp + dcp * (h * d_nope) + dcp * (h * d_rope)) * wb
        attn_k_bytes = L * (d * dc + dc * (h * d_nope) + d * d_rope) * wb
        attn_v_bytes = L * (dc * (h * d_v)) * wb

        attn_o_numel = 0
        ffn_numel = 0
        for layer in self.layers:
            attn = getattr(layer, "attn", None)
            proj = getattr(attn, "proj", None)
            if proj is not None:
                attn_o_numel += sum(p.numel() for p in proj.parameters())
            if layer.memory_layer is not None:
                continue
            if isinstance(layer.ff, SVDSwiGLUFF):
                ffn_numel += layer.ff.active_weight_numel()
            else:
                ffn_numel += sum(p.numel() for p in layer.ff.parameters())
        attn_o_bytes = attn_o_numel * wb

        dense_memory_query_bytes = M * (
            d * self.mem_q_rank + self.mem_q_rank * (self.mem_heads * self.mem_k_dim)
        ) * wb
        dense_memory_key_bytes = M * (self.mem_heads * self.mem_n_keys * self.mem_k_dim) * wb
        dense_memory_value_bytes = M * (self.mem_heads * self.mem_knn * self.mem_v_dim) * wb
        dense_memory_proj_bytes = M * (self.mem_v_dim * d) * wb if self.mem_v_dim != d else 0
        ffn_bytes = (
            ffn_numel * wb
            + dense_memory_query_bytes
            + dense_memory_key_bytes
            + dense_memory_value_bytes
            + dense_memory_proj_bytes
        )

        norm_bytes = (
            (2 * L + 1) * 2 * d * wb
            + (2 * L) * d * wb
            + (2 * L) * d * wb
        )
        kv_cache_token_width = dc + d_rope
        kv_cache_read_bytes = kv_cache_token_width * seq_len * exec_L * kb
        kv_cache_write_bytes = kv_cache_token_width * exec_L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="mla_hybrid_loop12_monarch_attn_svd_ffn",
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
                f"phase={'hotcold+svd' if bool(self.uses_structured_svd_flag.item()) else 'dense'}; "
                f"mem_layers={self.bottom_even_memory_layers}; "
                f"svd_ffn={self.svd_ffn_layers}; "
                f"dense_loop_ffn={self.loop_dense_ffn_layers}; "
                f"top_monarch_attn={self.top_even_monarch_layers + self.top_odd_monarch_layers}; "
                f"top4_loopx{self.loop_repeats}"
            ),
        )


class MLAHybridLoop12MonarchAttnSVDFfnBinaryDPTransformer(MLAHybridLoop12MonarchAttnSVDFfnTransformer):
    """Hybrid loop12 + Monarch attn + SVD FFN, with binary-DP memory layers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem_binary_total_keys = self.mem_n_keys * self.mem_n_keys
        self.mem_binary_buckets = int(math.log2(self.mem_binary_total_keys))
        if (1 << self.mem_binary_buckets) != self.mem_binary_total_keys:
            raise ValueError(
                "Binary memory requires mem_n_keys^2 to be power-of-two. "
                f"Got mem_n_keys={self.mem_n_keys} => total_keys={self.mem_binary_total_keys}."
            )
        if self.mem_k_dim % self.mem_binary_buckets != 0:
            raise ValueError(
                f"mem_k_dim must be divisible by num_binary_buckets={self.mem_binary_buckets}; "
                f"got mem_k_dim={self.mem_k_dim}."
            )

        shared_value_store = (
            BinaryCodeValueStore(self.mem_binary_total_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        for i in self.bottom_even_memory_layers:
            self.layers[i].memory_layer = BinaryProductCodeMemoryLayer(
                d_model=self.d_model,
                mem_n_keys=self.mem_n_keys,
                mem_heads=self.mem_heads,
                mem_knn=self.mem_knn,
                key_dim=self.mem_k_dim,
                value_dim=self.mem_v_dim,
                mem_q_rank=self.mem_q_rank,
                value_store=shared_value_store,
                memory_plus=False,
                qk_norm=self.qk_norm,
            )

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        profile = super().get_inference_profile(
            seq_len=seq_len,
            weight_dtype_bytes=weight_dtype_bytes,
            kv_dtype_bytes=kv_dtype_bytes,
            count_reuse=count_reuse,
        )
        wb = weight_dtype_bytes
        M = len(self.bottom_even_memory_layers)
        dense_memory_key_bytes = M * (self.mem_heads * self.mem_n_keys * self.mem_k_dim) * wb
        binary_memory_key_bytes = M * (self.mem_heads * 2 * self.mem_k_dim) * wb
        profile.ffn_bytes += (binary_memory_key_bytes - dense_memory_key_bytes)
        profile.model_name = "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp"
        profile.notes = (
            f"phase={'hotcold+svd' if bool(self.uses_structured_svd_flag.item()) else 'dense'}; "
            f"mem_layers={self.bottom_even_memory_layers}; "
            f"svd_ffn={self.svd_ffn_layers}; "
            f"top_monarch_attn={self.top_even_monarch_layers + self.top_odd_monarch_layers}; "
            f"binary_total_keys={self.mem_binary_total_keys}; "
            f"binary_buckets={self.mem_binary_buckets}; "
            f"top4_loopx{self.loop_repeats}"
        )
        return profile


class MLAHybridLoop12MonarchAttnLoRAFfnTransformer(MLAHybridLoop12MonarchAttnSVDFfnTransformer):
    """12-layer MLA hybrid loop with Monarch attention and switchable LoRA FFNs."""

    def __init__(
        self,
        *args,
        lora_ffn_rank: int | None = None,
        lora_ffn_alpha: float = 1.0,
        **kwargs,
    ):
        # LoRA schedule: activate adapters at 30% by default.
        if "svd_switch_fraction" not in kwargs:
            kwargs["svd_switch_fraction"] = 0.3
        if lora_ffn_rank is None:
            lora_ffn_rank = kwargs.pop("svd_ffn_rank", None)
        kwargs.pop("svd_ffn_rank", None)
        super().__init__(*args, svd_ffn_rank=lora_ffn_rank, **kwargs)
        self.lora_ffn_rank = lora_ffn_rank if lora_ffn_rank is not None else 96
        self.lora_ffn_alpha = float(lora_ffn_alpha)
        # Architecture requested by user:
        # - memory layers: 1,3,5,7
        # - shared dense FFN bases for pairs: (0,2), (4,6), (8,9), (10,11)
        # - per-layer LoRA on every non-memory layer
        self.bottom_even_memory_layers = [1, 3, 5, 7]
        self.lora_ffn_layers = [0, 2, 4, 6, 8, 9, 10, 11]
        shared_pairs = [(0, 2), (4, 6), (8, 9), (10, 11)]

        shared_value_store = (
            ProductKeyValueStore(self.mem_n_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        for i in range(self.n_layers):
            if i in self.bottom_even_memory_layers:
                self.layers[i].memory_layer = ProductKeyMemoryLayer(
                    d_model=self.d_model,
                    mem_n_keys=self.mem_n_keys,
                    mem_heads=self.mem_heads,
                    mem_knn=self.mem_knn,
                    key_dim=self.mem_k_dim,
                    value_dim=self.mem_v_dim,
                    mem_q_rank=self.mem_q_rank,
                    value_store=shared_value_store,
                    memory_plus=False,
                    qk_norm=self.qk_norm,
                )
            else:
                self.layers[i].memory_layer = None

        for a, b in shared_pairs:
            shared_dense = SwiGLUFF(d_model=self.d_model, d_ff=self.d_ff)
            ff_a = LoRASwiGLUFF(
                d_model=self.d_model,
                d_ff=self.d_ff,
                lora_rank=self.lora_ffn_rank,
                lora_alpha=self.lora_ffn_alpha,
            )
            ff_b = LoRASwiGLUFF(
                d_model=self.d_model,
                d_ff=self.d_ff,
                lora_rank=self.lora_ffn_rank,
                lora_alpha=self.lora_ffn_alpha,
            )
            # Pair-shared dense base; per-layer LoRA adapters remain independent.
            ff_a.w1.base = shared_dense.w1
            ff_a.w2.base = shared_dense.w2
            ff_a.w3.base = shared_dense.w3
            ff_b.w1.base = shared_dense.w1
            ff_b.w2.base = shared_dense.w2
            ff_b.w3.base = shared_dense.w3
            self.layers[a].ff = ff_a
            self.layers[b].ff = ff_b

    def _convert_embedding_to_hotcold(self):
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            full_weight = self.full_token_emb.weight.detach()
            hot_ids = self.token_emb.hot_token_ids
            cold_ids = self.token_emb.cold_token_ids
            self.token_emb.hot_emb.weight.copy_(full_weight[hot_ids])

            # LoRA path: initialize as cold low-dim embedding -> single linear to d_model.
            # Hot tokens stay full-dim with no extra projection.
            cold_full = full_weight[cold_ids].float()
            r = self.token_emb.cold_emb_u.weight.shape[1]
            cold_latent = cold_full[:, :r].contiguous()
            self.token_emb.cold_emb_u.weight.copy_(
                cold_latent.to(self.token_emb.cold_emb_u.weight.dtype)
            )

            # Fit projection W (Linear(r -> d_model)) by least squares:
            # cold_latent @ W^T ≈ cold_full.
            # This keeps init simple and avoids SVD-based decomposition.
            ls = torch.linalg.lstsq(cold_latent, cold_full).solution  # [r, d_model]
            self.token_emb.cold_latent_to_model.weight.copy_(
                ls.T.to(self.token_emb.cold_latent_to_model.weight.dtype)
            )

            self.full_token_emb.weight.requires_grad_(False)
            self.uses_hotcold_flag.fill_(True)

    def _convert_structured_layers_to_svd(self):
        if bool(self.uses_structured_svd_flag.item()):
            return
        for layer in self.layers:
            layer.convert_full_to_svd()
        self.uses_structured_svd_flag.fill_(True)

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
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
        exec_counts = self._layer_exec_counts()
        exec_L = sum(exec_counts)
        M = len(self.bottom_even_memory_layers)

        if bool(self.uses_hotcold_flag.item()):
            r = self.cold_latent_dim
            n_hot = self.token_emb.num_hot_tokens
            n_cold = self.token_emb.num_cold_tokens
            lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else max(d, r + d * r) * wb
        else:
            lm_head_bytes = V * d * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else d * wb

        attn_q_bytes = L * (d * dcp + dcp * (h * d_nope) + dcp * (h * d_rope)) * wb
        attn_k_bytes = L * (d * dc + dc * (h * d_nope) + d * d_rope) * wb
        attn_v_bytes = L * (dc * (h * d_v)) * wb

        attn_o_numel = 0
        ffn_numel = 0
        for layer in self.layers:
            attn = getattr(layer, "attn", None)
            proj = getattr(attn, "proj", None)
            if proj is not None:
                attn_o_numel += sum(p.numel() for p in proj.parameters())
            if layer.memory_layer is not None:
                continue
            if isinstance(layer.ff, (SVDSwiGLUFF, LoRASwiGLUFF)):
                ffn_numel += layer.ff.active_weight_numel()
            else:
                ffn_numel += sum(p.numel() for p in layer.ff.parameters())
        attn_o_bytes = attn_o_numel * wb

        dense_memory_query_bytes = M * (
            d * self.mem_q_rank + self.mem_q_rank * (self.mem_heads * self.mem_k_dim)
        ) * wb
        dense_memory_key_bytes = M * (self.mem_heads * self.mem_n_keys * self.mem_k_dim) * wb
        dense_memory_value_bytes = M * (self.mem_heads * self.mem_knn * self.mem_v_dim) * wb
        dense_memory_proj_bytes = M * (self.mem_v_dim * d) * wb if self.mem_v_dim != d else 0
        ffn_bytes = (
            ffn_numel * wb
            + dense_memory_query_bytes
            + dense_memory_key_bytes
            + dense_memory_value_bytes
            + dense_memory_proj_bytes
        )

        norm_bytes = (
            (2 * L + 1) * 2 * d * wb
            + (2 * L) * d * wb
            + (2 * L) * d * wb
        )
        kv_cache_token_width = dc + d_rope
        kv_cache_read_bytes = kv_cache_token_width * seq_len * exec_L * kb
        kv_cache_write_bytes = kv_cache_token_width * exec_L * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name="mla_hybrid_loop12_monarch_attn_lora_ffn",
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
                f"phase={'hotcold+lora' if bool(self.uses_structured_svd_flag.item()) else 'dense'}; "
                f"mem_layers={self.bottom_even_memory_layers}; "
                f"lora_ffn={self.lora_ffn_layers}; rank={self.lora_ffn_rank}; alpha={self.lora_ffn_alpha}; "
                f"dense_loop_ffn={self.loop_dense_ffn_layers}; "
                f"top_monarch_attn={self.top_even_monarch_layers + self.top_odd_monarch_layers}; "
                f"top4_loopx{self.loop_repeats}"
            ),
        )


class MLAHybridLoop12MonarchAttnLoRAFfnBinaryDPTransformer(MLAHybridLoop12MonarchAttnLoRAFfnTransformer):
    """Hybrid loop12 + Monarch attn + LoRA FFN, with binary-DP memory layers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem_binary_total_keys = self.mem_n_keys * self.mem_n_keys
        self.mem_binary_buckets = int(math.log2(self.mem_binary_total_keys))
        if (1 << self.mem_binary_buckets) != self.mem_binary_total_keys:
            raise ValueError(
                "Binary memory requires mem_n_keys^2 to be power-of-two. "
                f"Got mem_n_keys={self.mem_n_keys} => total_keys={self.mem_binary_total_keys}."
            )
        if self.mem_k_dim % self.mem_binary_buckets != 0:
            raise ValueError(
                f"mem_k_dim must be divisible by num_binary_buckets={self.mem_binary_buckets}; "
                f"got mem_k_dim={self.mem_k_dim}."
            )

        shared_value_store = (
            BinaryCodeValueStore(self.mem_binary_total_keys, self.mem_v_dim)
            if self.mem_share_values else None
        )
        for i in self.bottom_even_memory_layers:
            self.layers[i].memory_layer = BinaryProductCodeMemoryLayer(
                d_model=self.d_model,
                mem_n_keys=self.mem_n_keys,
                mem_heads=self.mem_heads,
                mem_knn=self.mem_knn,
                key_dim=self.mem_k_dim,
                value_dim=self.mem_v_dim,
                mem_q_rank=self.mem_q_rank,
                value_store=shared_value_store,
                memory_plus=False,
                qk_norm=self.qk_norm,
            )

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        profile = super().get_inference_profile(
            seq_len=seq_len,
            weight_dtype_bytes=weight_dtype_bytes,
            kv_dtype_bytes=kv_dtype_bytes,
            count_reuse=count_reuse,
        )
        wb = weight_dtype_bytes
        M = len(self.bottom_even_memory_layers)
        dense_memory_key_bytes = M * (self.mem_heads * self.mem_n_keys * self.mem_k_dim) * wb
        binary_memory_key_bytes = M * (self.mem_heads * 2 * self.mem_k_dim) * wb
        profile.ffn_bytes += (binary_memory_key_bytes - dense_memory_key_bytes)
        profile.model_name = "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp"
        profile.notes = (
            f"phase={'hotcold+lora' if bool(self.uses_structured_svd_flag.item()) else 'dense'}; "
            f"mem_layers={self.bottom_even_memory_layers}; "
            f"lora_ffn={self.lora_ffn_layers}; rank={self.lora_ffn_rank}; alpha={self.lora_ffn_alpha}; "
            f"top_monarch_attn={self.top_even_monarch_layers + self.top_odd_monarch_layers}; "
            f"binary_total_keys={self.mem_binary_total_keys}; "
            f"binary_buckets={self.mem_binary_buckets}; "
            f"top4_loopx{self.loop_repeats}"
        )
        return profile


# ============================================================================
# Factory
# ============================================================================

def create_model(variant: str = "baseline", **kwargs):
    """Factory function to create a model.

    Args:
        variant: "baseline", "gqa_only", "topk_only", "baseline_plus", "mla", "loop_top4x3_attnres", or "mla_hybrid_loop12"
        **kwargs: passed to the model constructor
    """
    mla_only_keys = {
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    }
    hotcold_only_keys = {
        "hot_token_k",
        "cold_latent_dim",
        "hot_token_cache_path",
    }
    twostage_only_keys = {
        "svd_switch_fraction",
    }
    mla_mem_monarch_only_keys = {
        "monarch_block_size",
        "memory_layers",
        "mem_n_keys",
        "mem_heads",
        "mem_knn",
        "mem_k_dim",
        "mem_v_dim",
        "mem_q_rank",
        "mem_share_values",
        "qk_norm",
    }
    if variant not in {
        "mla",
        "hotcold_mla",
        "mla_twostage_svd_mem12_monarch",
        "mla_twostage_svd_mem12_binarydp",
        "mla_hybrid_loop12",
        "mla_hybrid_loop12_monarch",
        "mla_hybrid_loop12_monarch_attn_svd_ffn",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
        "mla_hybrid_loop12_monarch_attn_lora_ffn",
        "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
    }:
        kwargs = {k: v for k, v in kwargs.items() if k not in mla_only_keys}
    if variant not in {
        "hotcold_svd",
        "twostage_svd",
        "hotcold_mla",
        "mla_twostage_svd_mem12_monarch",
        "mla_twostage_svd_mem12_binarydp",
        "mla_hybrid_loop12",
        "mla_hybrid_loop12_monarch",
        "mla_hybrid_loop12_monarch_attn_svd_ffn",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
        "mla_hybrid_loop12_monarch_attn_lora_ffn",
        "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
    }:
        kwargs = {k: v for k, v in kwargs.items() if k not in hotcold_only_keys}
    if variant not in {
        "twostage_svd",
        "hotcold_mla",
        "mla_twostage_svd_mem12_monarch",
        "mla_twostage_svd_mem12_binarydp",
        "mla_hybrid_loop12",
        "mla_hybrid_loop12_monarch",
        "mla_hybrid_loop12_monarch_attn_svd_ffn",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
        "mla_hybrid_loop12_monarch_attn_lora_ffn",
        "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
    }:
        kwargs = {k: v for k, v in kwargs.items() if k not in twostage_only_keys}
    if variant not in {
        "mla_twostage_svd_mem12_monarch",
        "mla_twostage_svd_mem12_binarydp",
        "dp_shared_memory",
        "mla_hybrid_loop12",
        "mla_hybrid_loop12_monarch",
        "mla_hybrid_loop12_monarch_attn_svd_ffn",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
        "mla_hybrid_loop12_monarch_attn_lora_ffn",
        "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
    }:
        kwargs = {k: v for k, v in kwargs.items() if k not in mla_mem_monarch_only_keys}

    if variant == "gqa_only":
        return GQAOnlyTransformer(**kwargs)
    elif variant == "topk_only":
        return TopKOnlyTransformer(**kwargs)
    elif variant == "baseline_plus":
        return BaselinePlusTransformer(**kwargs)
    elif variant == "mla":
        return MLATransformer(**kwargs)
    elif variant == "hotcold_mla":
        return HotColdMLATransformer(**kwargs)
    elif variant == "mla_twostage_svd_mem12_monarch":
        return MLATwoStageSVDMemoryMonarchTransformer(**kwargs)
    elif variant == "mla_twostage_svd_mem12_binarydp":
        return MLATwoStageSVDBinaryMemoryMonarchTransformer(**kwargs)
    elif variant == "dp_shared_memory":
        return DPSharedMemoryTransformer(**kwargs)
    elif variant == "mla_hybrid_loop12":
        return MLAHybridLoop12Transformer(**kwargs)
    elif variant == "mla_hybrid_loop12_monarch":
        return MLAHybridLoop12MonarchTransformer(**kwargs)
    elif variant == "mla_hybrid_loop12_monarch_attn_svd_ffn":
        return MLAHybridLoop12MonarchAttnSVDFfnTransformer(**kwargs)
    elif variant == "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp":
        return MLAHybridLoop12MonarchAttnSVDFfnBinaryDPTransformer(**kwargs)
    elif variant == "mla_hybrid_loop12_monarch_attn_lora_ffn":
        return MLAHybridLoop12MonarchAttnLoRAFfnTransformer(**kwargs)
    elif variant == "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp":
        return MLAHybridLoop12MonarchAttnLoRAFfnBinaryDPTransformer(**kwargs)
    elif variant == "hotcold_svd":
        return HotColdSVDTransformer(**kwargs)
    elif variant == "twostage_svd":
        return TwoStageSVDTransformer(**kwargs)
    elif variant == "loop_top4x3_attnres":
        return LoopTop4x3AttnResTransformer(**kwargs)
    else:
        return SimpleTransformer(**kwargs)


if __name__ == "__main__":
    from metric import print_profile

    for variant in [
        "baseline",
        "gqa_only",
        "topk_only",
        "baseline_plus",
        "mla",
        "hotcold_mla",
        "mla_twostage_svd_mem12_monarch",
        "mla_twostage_svd_mem12_binarydp",
        "dp_shared_memory",
        "mla_hybrid_loop12",
        "mla_hybrid_loop12_monarch",
        "mla_hybrid_loop12_monarch_attn_svd_ffn",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
        "mla_hybrid_loop12_monarch_attn_lora_ffn",
        "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
        "hotcold_svd",
        "twostage_svd",
        "loop_top4x3_attnres",
    ]:
        print(f"\n{'='*60}")
        print(f"  {variant}")
        print(f"{'='*60}")
        try:
            model = create_model(variant=variant)
        except FileNotFoundError as e:
            print(f"  Skipping {variant}: {e}")
            continue
        total_params = model.count_parameters(count_zeros=True)
        nonzero_params = model.count_parameters(count_zeros=False)
        print(f"  Total parameters:    {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")
        profile = model.get_inference_profile()
        print_profile(profile)
