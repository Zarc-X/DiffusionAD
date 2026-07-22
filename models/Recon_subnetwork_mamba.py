import math
from typing import List

import torch
from torch import nn

from models.Recon_subnetwork import (
    AttentionBlock,
    GroupNorm32,
    PositionalEmbedding,
    ResBlock,
    TimestepEmbedSequential,
    zero_module,
)

try:
    from mamba_ssm import Mamba as MambaSSM
except Exception:
    MambaSSM = None


MAMBA_AVAILABLE = MambaSSM is not None


def _parse_resolution_to_ds(img_size: int, res_text: str) -> List[int]:
    ds_values = []
    for item in str(res_text).split(","):
        item = item.strip()
        if not item:
            continue
        ds_values.append(img_size // int(item))
    return ds_values


class SpatialMambaBlock(nn.Module):
    """
    Spatial token mixer using Mamba on flattened HxW tokens.
    This block is drop-in compatible with AttentionBlock in UNet stages.
    """

    def __init__(
        self,
        in_channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is not installed. Install with: pip install mamba-ssm causal-conv1d"
            )

        self.in_channels = in_channels
        self.norm = GroupNorm32(32, in_channels)
        self.bidirectional = bidirectional

        self.mamba_fwd = MambaSSM(
            d_model=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if self.bidirectional:
            self.mamba_bwd = MambaSSM(
                d_model=in_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, time=None):
        if not x.is_cuda:
            raise RuntimeError(
                "SpatialMambaBlock requires CUDA tensors because mamba_ssm/causal_conv1d kernels are CUDA-only in this environment."
            )

        residual = x
        b, c, h, w = x.shape

        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        out = self.mamba_fwd(x)
        if self.bidirectional:
            rev_in = torch.flip(x, dims=[1])
            rev_out = self.mamba_bwd(rev_in)
            rev_out = torch.flip(rev_out, dims=[1])
            out = 0.5 * (out + rev_out)

        out = self.out_proj(out)
        out = self.dropout(out)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        return residual + out


class UNetModelMamba(nn.Module):
    """
    UNet reconstruction model with optional Mamba injection.

    mamba_mode:
        - "none": keep original attention-only UNet behavior.
        - "low": replace attention blocks at selected resolutions with Mamba.
        - "medium": low-risk replacement + additional deep-stage Mamba blocks.
    """

    def __init__(
        self,
        img_size,
        base_channels,
        conv_resample=True,
        n_heads=1,
        n_head_channels=-1,
        channel_mults="",
        num_res_blocks=2,
        dropout=0,
        attention_resolutions="32,16,8",
        biggan_updown=True,
        in_channels=1,
        mamba_mode="low",
        mamba_resolutions="32,16,8",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dropout=0.0,
        mamba_bidirectional=True,
        mamba_medium_min_ds=8,
    ):
        super().__init__()
        self.dtype = torch.float32

        if channel_mults == "":
            if img_size == 512:
                channel_mults = (0.5, 1, 1, 2, 2, 4, 4)
            elif img_size == 256:
                channel_mults = (1, 1, 2, 2, 4, 4)
            elif img_size == 128:
                channel_mults = (1, 1, 2, 3, 4)
            elif img_size in (64, 32):
                channel_mults = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {img_size}")

        if mamba_mode not in {"none", "low", "medium"}:
            raise ValueError(f"unsupported mamba_mode: {mamba_mode}")
        if mamba_mode != "none" and not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_mode is enabled but mamba_ssm is missing. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )

        self.image_size = img_size
        self.in_channels = in_channels
        self.model_channels = base_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample
        self.num_heads = n_heads
        self.num_head_channels = n_head_channels

        self.mamba_mode = mamba_mode
        self.mamba_ds = set(_parse_resolution_to_ds(img_size, mamba_resolutions))
        self.attention_ds = set(_parse_resolution_to_ds(img_size, attention_resolutions))
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dropout = mamba_dropout
        self.mamba_bidirectional = mamba_bidirectional
        self.mamba_medium_min_ds = mamba_medium_min_ds

        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(base_channels, 1),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mults[0] * base_channels)
        self.down = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(self.in_channels, base_channels, 3, padding=1))]
        )
        channels = [ch]
        ds = 1

        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        out_channels=base_channels * mult,
                        dropout=dropout,
                    )
                ]
                ch = base_channels * mult

                context = self._build_context_block(ch=ch, ds=ds)
                if context is not None:
                    layers.append(context)

                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)

            if i != len(channel_mults) - 1:
                out_channels = ch
                self.down.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim=time_embed_dim,
                            out_channels=out_channels,
                            dropout=dropout,
                            down=True,
                        )
                        if biggan_updown
                        else nn.AvgPool2d(kernel_size=2, stride=2)
                    )
                )
                ds *= 2
                ch = out_channels
                channels.append(ch)

        middle_layers = [
            ResBlock(
                ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
            )
        ]
        middle_context = self._build_context_block(ch=ch, ds=ds, force_attention_fallback=True)
        middle_layers.append(middle_context)
        middle_layers.append(
            ResBlock(
                ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
            )
        )
        if self.mamba_mode == "medium":
            middle_layers.append(
                SpatialMambaBlock(
                    ch,
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand,
                    dropout=self.mamba_dropout,
                    bidirectional=self.mamba_bidirectional,
                )
            )
        self.middle = TimestepEmbedSequential(*middle_layers)

        self.up = nn.ModuleList([])
        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [
                    ResBlock(
                        ch + inp_chs,
                        time_embed_dim=time_embed_dim,
                        out_channels=base_channels * mult,
                        dropout=dropout,
                    )
                ]
                ch = base_channels * mult

                context = self._build_context_block(ch=ch, ds=ds)
                if context is not None:
                    layers.append(context)

                if i and j == num_res_blocks:
                    out_channels = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim=time_embed_dim,
                            out_channels=out_channels,
                            dropout=dropout,
                            up=True,
                        )
                    )
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(base_channels * channel_mults[0], self.out_channels, 3, padding=1)),
        )

    def _build_context_block(self, ch, ds, force_attention_fallback=False):
        use_attention = ds in self.attention_ds
        use_mamba = False

        if self.mamba_mode == "low":
            use_mamba = ds in self.mamba_ds
        elif self.mamba_mode == "medium":
            use_mamba = (ds in self.mamba_ds) or (ds >= self.mamba_medium_min_ds)

        if use_mamba:
            return SpatialMambaBlock(
                ch,
                d_state=self.mamba_d_state,
                d_conv=self.mamba_d_conv,
                expand=self.mamba_expand,
                dropout=self.mamba_dropout,
                bidirectional=self.mamba_bidirectional,
            )

        if use_attention or force_attention_fallback:
            return AttentionBlock(
                ch,
                n_heads=self.num_heads,
                n_head_channels=self.num_head_channels,
            )

        return None

    def forward(self, x, time):
        time_embed = self.time_embedding(time)

        skips = []
        h = x.type(self.dtype)
        for module in self.down:
            h = module(h, time_embed)
            skips.append(h)

        h = self.middle(h, time_embed)

        for module in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, time_embed)

        h = h.type(x.dtype)
        h = self.out(h)
        return h


if __name__ == "__main__":
    args = {
        "img_size": 256,
        "base_channels": 128,
        "dropout": 0,
        "num_heads": 4,
        "attention_resolutions": "32,16,8",
    }

    model = UNetModelMamba(
        img_size=args["img_size"],
        base_channels=args["base_channels"],
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        attention_resolutions=args["attention_resolutions"],
        in_channels=3,
        mamba_mode="none",
    )
    x = torch.randn(1, 3, 256, 256)
    t = torch.tensor([10])
    y = model(x, t)
    print(y.shape)
