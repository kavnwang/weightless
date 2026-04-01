import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_key: int,  # dimension for query/key vectors
        max_seq_len: int,
        device: torch.device | None = None,  # where to store the buffer
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_key = d_key
        assert self.d_key % 2 == 0, "d_key must be even"
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        # Create cos/sin cache
        inds = torch.arange(max_seq_len, device=device, dtype=dtype).reshape(-1, 1)
        ks = torch.arange(d_key // 2, device=device, dtype=dtype).reshape(1, -1)
        freqs = inds / (theta ** (2 * ks / d_key))

        # Register as buffers (non-trainable, saved in state_dict)
        self.register_buffer("coses", torch.cos(freqs), persistent=False)
        self.register_buffer("sins", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        *batch, seq_len, d_key = x.shape
        if d_key != self.d_key:
            raise ValueError(f"x final dim has length {d_key} expected {self.d_key}")

        # Gather cos/sin: (..., seq, d/2)
        # positions can be (batch, seq) or (batch, 1, seq)
        cos = self.coses[positions]
        sin = self.sins[positions]

        # Apply RoPE: (x0, x1) -> (x0 cos - x1 sin, x0 sin + x1 cos)
        # View as pairs: (..., seq, d/2, 2)
        x_pairs = x.view(*batch, seq_len, d_key // 2, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        # Automatic broadcasting happens here between x (batch, heads, ...) and cos (batch, 1, ...)
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos

        # Stack and flatten back to (..., seq, d_key)
        result = torch.stack((x0_rot, x1_rot), dim=-1)
        return result.view(*batch, seq_len, d_key).to(x.dtype)