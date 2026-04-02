import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from metric import InferenceProfile
from rope import RotaryPositionalEmbedding


VOCAB_SIZE = 50257
SEQ_LEN = 512
DEFAULT_HOT_TOKEN_CACHE_PATH = "cache/hot_tokens_train1p3b_top2000.pt"
SUPPORTED_VARIANT = "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp"


def _load_hot_token_ids(cache_path: str, vocab_size: int, hot_token_k: int) -> torch.Tensor:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Hot-token cache not found at '{cache_path}'. "
            f"Build it first with: python build_hot_token_cache.py --cache_path {cache_path}"
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    token_ids = torch.as_tensor(payload["hot_token_ids"], dtype=torch.long).flatten()[:hot_token_k]
    if token_ids.numel() < hot_token_k:
        raise ValueError(f"Hot-token cache has only {token_ids.numel()} ids but requested {hot_token_k}.")
    if token_ids.min().item() < 0 or token_ids.max().item() >= vocab_size:
        raise ValueError(f"Hot-token cache contains out-of-range ids for vocab_size={vocab_size}.")
    return token_ids


class HotColdTiedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hot_token_ids: torch.Tensor, cold_latent_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        hot = self.hot_token_mask[flat]
        out = torch.empty(flat.numel(), self.d_model, device=flat.device, dtype=self.hot_emb.weight.dtype)
        if hot.any():
            out[hot] = self.hot_emb(self.token_to_hot_idx[flat[hot]]).to(out.dtype)
        if (~hot).any():
            cold_latent = self.cold_emb_u(self.token_to_cold_idx[flat[~hot]])
            out[~hot] = self.cold_latent_to_model(cold_latent).to(out.dtype)
        return out.view(*input_ids.shape, self.d_model)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.empty(
            (*hidden_states.shape[:-1], self.vocab_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        logits[..., self.hot_token_ids] = hidden_states @ self.hot_emb.weight.T
        hidden_latent = hidden_states @ self.cold_latent_to_model.weight
        logits[..., self.cold_token_ids] = hidden_latent @ self.cold_emb_u.weight.T
        return logits


class MultiLatentAttention(nn.Module):
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
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.kv_down = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.k_up = nn.Linear(kv_lora_rank, n_heads * qk_nope_head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)
        self.k_rope_proj = nn.Linear(d_model, qk_rope_head_dim, bias=False)
        self.q_down = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_up = nn.Linear(q_lora_rank, n_heads * qk_nope_head_dim, bias=False)
        self.q_rope_proj = nn.Linear(q_lora_rank, n_heads * qk_rope_head_dim, bias=False)
        self.proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def forward(self, x, causal_mask, attention_mask, rope, positions):
        del causal_mask, attention_mask
        bsz, seq_len, _ = x.shape
        c_q = self.q_down(x)
        q_c = self.q_up(c_q).reshape(bsz, seq_len, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        q_r = self.q_rope_proj(c_q).reshape(bsz, seq_len, self.n_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_r = rope(q_r, positions)
        q = torch.cat([q_c, q_r], dim=-1)

        c_kv = self.kv_down(x)
        k_c = self.k_up(c_kv).reshape(bsz, seq_len, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        v_c = self.v_up(c_kv).reshape(bsz, seq_len, self.n_heads, self.v_head_dim).transpose(1, 2)
        k_r = self.k_rope_proj(x).reshape(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_r = rope(k_r, positions).expand(bsz, self.n_heads, seq_len, self.qk_rope_head_dim)
        k = torch.cat([k_c, k_r], dim=-1)
        out = F.scaled_dot_product_attention(
            q, k, v_c, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.v_head_dim)
        return self.proj(out)


class MonarchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, block_size: int = 32):
        super().__init__()
        if in_features != out_features or in_features % block_size != 0:
            raise ValueError("MonarchLinear requires square dims divisible by block_size.")
        self.n_blocks = in_features // block_size
        self.block_size = block_size
        self.left = nn.Parameter(torch.empty(self.n_blocks, block_size, block_size))
        self.right = nn.Parameter(torch.empty(self.n_blocks, block_size, block_size))
        nn.init.normal_(self.left, mean=0.0, std=block_size ** -0.5)
        nn.init.normal_(self.right, mean=0.0, std=block_size ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(*x.shape[:-1], self.n_blocks, self.block_size)
        y = torch.einsum("...nb,nab->...na", x, self.right)
        y = torch.einsum("...na,nab->...nb", y, self.left)
        return y.reshape(*y.shape[:-2], self.n_blocks * self.block_size)


class MultiLatentAttentionMonarch(MultiLatentAttention):
    def __init__(self, monarch_block_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        proj_in = kwargs["n_heads"] * kwargs["v_head_dim"]
        self.proj = MonarchLinear(proj_in, kwargs["d_model"], block_size=monarch_block_size)


class SwiGLUFF(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TopKSwiGLUFF(SwiGLUFF):
    def __init__(self, d_model: int, d_ff: int, top_k: int):
        super().__init__(d_model, d_ff)
        self.top_k = top_k
        self.d_ff = d_ff

    def forward(self, x):
        gate = F.silu(self.w1(x)) * self.w3(x)
        if self.top_k < self.d_ff:
            _, idx = torch.topk(gate.abs(), self.top_k, dim=-1)
            mask = torch.zeros_like(gate)
            mask.scatter_(-1, idx, 1.0)
            gate = gate * mask
        return self.w2(gate)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class BinaryCodeValueStore(nn.Module):
    def __init__(self, total_keys: int, value_dim: int):
        super().__init__()
        self.values = nn.Embedding(total_keys, value_dim)
        nn.init.normal_(self.values.weight, mean=0.0, std=value_dim ** -0.5)

    @property
    def weight(self) -> torch.Tensor:
        return self.values.weight


class BinaryProductCodeMemoryLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        mem_n_keys: int,
        mem_heads: int,
        mem_knn: int,
        key_dim: int,
        value_dim: int,
        mem_q_rank: int,
        value_store: BinaryCodeValueStore | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.mem_heads = mem_heads
        self.mem_knn = mem_knn
        self.key_dim = key_dim
        self.mem_q_rank = mem_q_rank
        self.value_dim = value_dim
        self.qk_norm = qk_norm
        self.total_keys = mem_n_keys * mem_n_keys
        self.num_buckets = int(math.log2(self.total_keys))
        self.bucket_dim = key_dim // self.num_buckets
        self.keys = nn.Parameter(torch.empty(mem_heads, self.num_buckets, 2, self.bucket_dim))
        self.value_store = value_store if value_store is not None else BinaryCodeValueStore(self.total_keys, value_dim)
        self.query_down = nn.Linear(d_model, mem_q_rank, bias=True)
        self.query_up = nn.Linear(mem_q_rank, mem_heads * key_dim, bias=False)
        self.value_proj = nn.Linear(value_dim, d_model, bias=False) if value_dim != d_model else None
        self.reset_binary_key_parameters()

    def reset_binary_key_parameters(self):
        bound = 1.0 / math.sqrt(self.bucket_dim)
        nn.init.uniform_(self.keys, -bound, bound)
        nn.init.xavier_uniform_(self.query_down.weight)
        nn.init.zeros_(self.query_down.bias)
        nn.init.xavier_uniform_(self.query_up.weight)

    def get_indices(self, query: torch.Tensor, knn: int) -> tuple[torch.Tensor, torch.Tensor]:
        bs = query.size(0) // self.mem_heads
        query = query.view(bs, self.mem_heads, self.num_buckets, self.bucket_dim)
        keys = self.keys
        if self.qk_norm:
            query = _rms_norm(query)
            keys = _rms_norm(keys)
        local_scores = torch.einsum("bhmd,hmcd->bhmc", query.float(), keys.float())
        score0, score1 = local_scores[..., 0], local_scores[..., 1]
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
        penalties = torch.zeros(rows, 1, device=query.device, dtype=torch.float32)
        masks = torch.zeros(rows, 1, device=query.device, dtype=torch.int64)
        for t in range(self.num_buckets):
            cand_pen = torch.cat([penalties, penalties + deltas[:, t:t + 1]], dim=1)
            cand_msk = torch.cat([masks, masks ^ bit_weights[t]], dim=1)
            step_k = min(knn_eff, cand_pen.size(1))
            penalties, keep = torch.topk(cand_pen, k=step_k, dim=1, largest=False, sorted=True)
            masks = cand_msk.gather(1, keep)
        return best_scores - penalties, best_codes ^ masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x_flat = x.reshape(bsz * seq_len, self.d_model)
        query = self.query_up(self.query_down(x_flat)).view(bsz * seq_len, self.mem_heads, self.key_dim)
        scores, indices = self.get_indices(query.reshape(bsz * seq_len * self.mem_heads, self.key_dim), self.mem_knn)
        scores = F.softmax(scores.float(), dim=-1).to(self.value_store.weight.dtype)
        indices = indices.view(bsz * seq_len, self.mem_heads * self.mem_knn)
        scores = scores.view(bsz * seq_len, self.mem_heads * self.mem_knn)
        y = F.embedding_bag(indices, self.value_store.weight, per_sample_weights=scores, mode="sum")
        if self.value_proj is not None:
            y = self.value_proj(y)
        return y.view(bsz, seq_len, -1)


def block_attn_res(
    blocks: list[torch.Tensor],
    partial: torch.Tensor,
    proj: nn.Linear,
    norm: nn.RMSNorm,
) -> torch.Tensor:
    v = torch.stack(blocks + [partial], dim=0)
    k = norm(v)
    logits = torch.einsum("d,nbtd->nbt", proj.weight.squeeze(0), k)
    return torch.einsum("nbt,nbtd->btd", logits.softmax(0), v)


class HybridAttnResidualTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        layer_number: int,
        attn: nn.Module,
        ff: nn.Module | None,
        memory: nn.Module | None,
        block_size: int,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.boundary_interval = max(1, block_size // 2)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = ff
        self.memory_layer = memory
        self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
        self.attn_res_norm = nn.RMSNorm(d_model)
        self.mlp_res_proj = nn.Linear(d_model, 1, bias=False)
        self.mlp_res_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        blocks,
        hidden,
        causal_mask,
        attention_mask,
        rope,
        positions,
        execution_layer_number=None,
    ):
        partial = hidden if hidden is not None else blocks[-1]
        h = block_attn_res(blocks, partial, self.attn_res_proj, self.attn_res_norm)
        layer_num = self.layer_number if execution_layer_number is None else execution_layer_number
        if layer_num > 0 and (layer_num % self.boundary_interval == 0):
            blocks.append(partial)
            partial = None
        attn_out = self.attn(self.ln1(h), causal_mask, attention_mask, rope, positions)
        partial = attn_out if partial is None else partial + attn_out
        h = block_attn_res(blocks, partial, self.mlp_res_proj, self.mlp_res_norm)
        ff_out = (
            self.memory_layer(self.ln2(h))
            if self.memory_layer is not None
            else self.ff(self.ln2(h))
        )
        return blocks, partial + ff_out


class MLAHybridLoop12MonarchAttnLoRAFfnBinaryDPTransformer(nn.Module):
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
        svd_switch_fraction: float = 0.3,
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
        loop_ffn_top_k: int | None = None,
        lora_ffn_rank: int | None = None,
        lora_ffn_alpha: float = 1.0,
    ):
        super().__init__()
        if n_layers != 12 or loop_block_size != 4 or loop_repeats != 3:
            raise ValueError(
                f"{SUPPORTED_VARIANT} requires n_layers=12, "
                "loop_block_size=4, loop_repeats=3."
            )
        del memory_layers, svd_switch_fraction, lora_ffn_rank, lora_ffn_alpha

        head_dim = d_model // n_heads
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else max(
            16, d_model // 8
        )
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else max(
            16, d_model // 8
        )
        self.qk_nope_head_dim = (
            qk_nope_head_dim if qk_nope_head_dim is not None else head_dim
        )
        self.qk_rope_head_dim = (
            qk_rope_head_dim
            if qk_rope_head_dim is not None
            else max(2, (head_dim // 2) * 2)
        )
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

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
        self.mem_n_keys = mem_n_keys
        self.mem_heads = mem_heads
        self.mem_knn = mem_knn
        self.mem_k_dim = mem_k_dim if mem_k_dim is not None else d_model
        self.mem_v_dim = mem_v_dim if mem_v_dim is not None else d_model
        self.mem_q_rank = mem_q_rank if mem_q_rank is not None else max(
            16, d_model // 4
        )
        self.loop_block_size = loop_block_size
        self.loop_repeats = loop_repeats
        self.loop_start = self.n_layers - self.loop_block_size
        self.loop_ffn_top_k = (
            loop_ffn_top_k if loop_ffn_top_k is not None else self.d_ff // 4
        )

        self.bottom_even_memory_layers = [1, 3, 5, 7]
        self.bottom_tied_ffn_layers = [0, 2, 4, 6]
        self.loop_topk_ffn_layers = [8, 9, 10]
        self.loop_dense_ffn_layers = [11]
        self.top_even_monarch_layers = [8, 10]
        self.top_odd_monarch_layers = [9, 11]

        self.register_buffer(
            "uses_hotcold_flag",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "uses_structured_svd_flag",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

        hot_ids = _load_hot_token_ids(hot_token_cache_path, vocab_size, hot_token_k)
        self.full_token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb = HotColdTiedEmbedding(
            vocab_size, d_model, hot_ids, cold_latent_dim
        )
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=self.qk_rope_head_dim,
            max_seq_len=max_seq_len,
        )

        self.mem_binary_total_keys = self.mem_n_keys * self.mem_n_keys
        self.mem_binary_buckets = int(math.log2(self.mem_binary_total_keys))
        shared_store = (
            BinaryCodeValueStore(self.mem_binary_total_keys, self.mem_v_dim)
            if mem_share_values
            else None
        )
        shared_dense = SwiGLUFF(d_model, d_ff)

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            attn = MultiLatentAttentionMonarch(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                monarch_block_size=monarch_block_size,
            )
            memory = None
            ff = None
            if i in self.bottom_even_memory_layers:
                memory = BinaryProductCodeMemoryLayer(
                    d_model=d_model,
                    mem_n_keys=self.mem_n_keys,
                    mem_heads=self.mem_heads,
                    mem_knn=self.mem_knn,
                    key_dim=self.mem_k_dim,
                    value_dim=self.mem_v_dim,
                    mem_q_rank=self.mem_q_rank,
                    value_store=shared_store,
                    qk_norm=qk_norm,
                )
            elif i in self.bottom_tied_ffn_layers:
                ff_i = SwiGLUFF(d_model, d_ff)
                ff_i.w1, ff_i.w2, ff_i.w3 = (
                    shared_dense.w1,
                    shared_dense.w2,
                    shared_dense.w3,
                )
                ff = ff_i
            elif i in self.loop_topk_ffn_layers:
                ff = TopKSwiGLUFF(d_model, d_ff, self.loop_ffn_top_k)
            else:
                ff = SwiGLUFF(d_model, d_ff)
            self.layers.append(
                HybridAttnResidualTransformerBlock(
                    d_model,
                    i,
                    attn,
                    ff,
                    memory,
                    attn_res_block_size,
                )
            )

        self.ln_f = nn.LayerNorm(d_model)

    def _convert_embedding_to_hotcold(self):
        if bool(self.uses_hotcold_flag.item()):
            return
        with torch.no_grad():
            w = self.full_token_emb.weight.detach()
            hot_ids = self.token_emb.hot_token_ids
            cold_ids = self.token_emb.cold_token_ids
            self.token_emb.hot_emb.weight.copy_(w[hot_ids])
            cold_full = w[cold_ids].float()
            r = self.token_emb.cold_emb_u.weight.shape[1]
            cold_latent = cold_full[:, :r].contiguous()
            self.token_emb.cold_emb_u.weight.copy_(
                cold_latent.to(self.token_emb.cold_emb_u.weight.dtype)
            )
            ls = torch.linalg.lstsq(cold_latent, cold_full).solution
            self.token_emb.cold_latent_to_model.weight.copy_(
                ls.T.to(self.token_emb.cold_latent_to_model.weight.dtype)
            )
            self.full_token_emb.weight.requires_grad_(False)
            self.uses_hotcold_flag.fill_(True)

    def convert_full_to_hotcold_svd(self):
        self._convert_embedding_to_hotcold()
        self.uses_structured_svd_flag.fill_(True)

    def token_partition_masks(self, token_ids: torch.Tensor):
        if not bool(self.uses_hotcold_flag.item()):
            return None, None
        hot = self.token_emb.token_is_hot(token_ids)
        return hot, ~hot

    def count_parameters(self, count_zeros: bool = False):
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        return sum((p != 0).sum().item() for p in self.parameters())

    def forward(self, input_ids, attention_mask=None):
        _, t = input_ids.shape
        x = (
            self.token_emb(input_ids)
            if bool(self.uses_hotcold_flag.item())
            else self.full_token_emb(input_ids)
        )
        x = self.dropout(x)
        positions = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        causal_mask = torch.triu(
            torch.ones(t, t, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
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
                exec_layer_number,
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
                    exec_layer_number,
                )
                exec_layer_number += 1
        x = self.ln_f(partial)
        return (
            self.token_emb.logits(x)
            if bool(self.uses_hotcold_flag.item())
            else F.linear(x, self.full_token_emb.weight)
        )

    def get_inference_profile(
        self,
        seq_len: int = 512,
        weight_dtype_bytes: float = 2,
        kv_dtype_bytes: float = 2,
        count_reuse: bool = False,
    ) -> InferenceProfile:
        d, h, lyrs, vocab = self.d_model, self.n_heads, self.n_layers, self.vocab_size
        wb, kb = weight_dtype_bytes, kv_dtype_bytes
        dc, dcp = self.kv_lora_rank, self.q_lora_rank
        d_nope, d_rope, d_v = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        exec_l = self.loop_start + self.loop_block_size * self.loop_repeats
        mem_layers = len(self.bottom_even_memory_layers)
        k = self.loop_ffn_top_k

        if bool(self.uses_hotcold_flag.item()):
            r = self.cold_latent_dim
            n_hot, n_cold = (
                self.token_emb.num_hot_tokens,
                self.token_emb.num_cold_tokens,
            )
            lm_head_bytes = (n_hot * d + d * r + n_cold * r) * wb
            embedding_bytes = (
                0 if (self.weight_tied and not count_reuse) else max(d, r + d * r) * wb
            )
            phase = "hotcold_embed"
        else:
            lm_head_bytes = vocab * d * wb
            embedding_bytes = 0 if (self.weight_tied and not count_reuse) else d * wb
            phase = "dense_embed"

        attn_q_bytes = lyrs * (d * dcp + dcp * (h * d_nope) + dcp * (h * d_rope)) * wb
        attn_k_bytes = lyrs * (d * dc + dc * (h * d_nope) + d * d_rope) * wb
        attn_v_bytes = lyrs * (dc * (h * d_v)) * wb
        attn_o_numel = sum(
            sum(p.numel() for p in layer.attn.proj.parameters())
            for layer in self.layers
        )
        attn_o_bytes = attn_o_numel * wb

        tied_dense_ffn_bytes = (d * self.d_ff + self.d_ff * d + d * self.d_ff) * wb
        topk_ffn_bytes = (
            len(self.loop_topk_ffn_layers) * (d * self.d_ff + k * d + d * self.d_ff) * wb
        )
        dense_loop_ffn_bytes = (
            len(self.loop_dense_ffn_layers)
            * (d * self.d_ff + self.d_ff * d + d * self.d_ff)
            * wb
        )
        memory_query_bytes = (
            mem_layers
            * (d * self.mem_q_rank + self.mem_q_rank * (self.mem_heads * self.mem_k_dim))
            * wb
        )
        memory_key_bytes = mem_layers * (self.mem_heads * 2 * self.mem_k_dim) * wb
        memory_value_bytes = (
            mem_layers * (self.mem_heads * self.mem_knn * self.mem_v_dim) * wb
        )
        memory_proj_bytes = mem_layers * (self.mem_v_dim * d) * wb if self.mem_v_dim != d else 0
        ffn_bytes = (
            tied_dense_ffn_bytes
            + topk_ffn_bytes
            + dense_loop_ffn_bytes
            + memory_query_bytes
            + memory_key_bytes
            + memory_value_bytes
            + memory_proj_bytes
        )

        norm_bytes = ((2 * lyrs + 1) * 2 * d + (2 * lyrs) * d + (2 * lyrs) * d) * wb
        kv_cache_token_width = dc + d_rope
        kv_cache_read_bytes = kv_cache_token_width * seq_len * exec_l * kb
        kv_cache_write_bytes = kv_cache_token_width * exec_l * kb

        seen_ptrs: set[int] = set()
        unique_numel = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_numel += p.numel()

        return InferenceProfile(
            model_name=SUPPORTED_VARIANT,
            d_model=d,
            n_layers=lyrs,
            n_heads=h,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            d_ff=self.d_ff,
            vocab_size=vocab,
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
                f"phase={phase}; mem_layers={self.bottom_even_memory_layers}; "
                f"bottom_tied_ffn={self.bottom_tied_ffn_layers}; "
                f"loop_topk_ffn={self.loop_topk_ffn_layers}; top_k={self.loop_ffn_top_k}; "
                f"dense_loop_ffn={self.loop_dense_ffn_layers}; "
                f"binary_total_keys={self.mem_binary_total_keys}; "
                f"binary_buckets={self.mem_binary_buckets}; "
                f"top_monarch_attn={self.top_even_monarch_layers + self.top_odd_monarch_layers}; "
                f"top4_loopx{self.loop_repeats}; weight_reads_counted_once_for_looped_layers"
            ),
        )


def create_model(variant: str = SUPPORTED_VARIANT, **kwargs):
    if variant != SUPPORTED_VARIANT:
        raise ValueError(
            f"Only '{SUPPORTED_VARIANT}' is supported in this simplified model.py; got '{variant}'."
        )
    return MLAHybridLoop12MonarchAttnLoRAFfnBinaryDPTransformer(**kwargs)
