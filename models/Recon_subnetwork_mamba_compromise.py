import math
from typing import List, Tuple

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


def _normalize_profile(profile: str) -> str:
    p = str(profile).strip().lower()
    aliases = {
        "fast": "speed",
        "faster": "speed",
        "quick": "speed",
        "balance": "balanced",
        "quality": "accuracy",
        "acc": "accuracy",
        "best": "accuracy",
    }
    if p in aliases:
        p = aliases[p]
    if p not in {"speed", "balanced", "accuracy"}:
        raise ValueError(
            f"Unsupported directional_profile: {profile}. "
            "Use one of: speed, balanced, accuracy"
        )
    return p


class SpatialMambaDirectionalBlock(nn.Module):
    """
    Spatial token mixer with selectable scan directions.

    Supported direction sets:
    - Horizontal: left-to-right (and optional right-to-left)
    - Vertical: top-to-bottom (and optional bottom-to-top)

    The block averages enabled directional branches, then applies a 1x1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        use_horizontal: bool = True,
        horizontal_bidirectional: bool = False,
        use_vertical: bool = False,
        vertical_bidirectional: bool = False,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is not installed. Install with: pip install mamba-ssm causal-conv1d"
            )
        if not use_horizontal and not use_vertical:
            raise ValueError("At least one scan direction must be enabled.")

        self.in_channels = in_channels
        self.norm = GroupNorm32(32, in_channels)

        self.use_horizontal = use_horizontal
        self.horizontal_bidirectional = horizontal_bidirectional
        self.use_vertical = use_vertical
        self.vertical_bidirectional = vertical_bidirectional

        if self.use_horizontal:
            self.h_fwd = MambaSSM(
                d_model=in_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            if self.horizontal_bidirectional:
                self.h_bwd = MambaSSM(
                    d_model=in_channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )

        if self.use_vertical:
            self.v_fwd = MambaSSM(
                d_model=in_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            if self.vertical_bidirectional:
                self.v_bwd = MambaSSM(
                    d_model=in_channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )

        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _run_seq(self, seq: torch.Tensor, fwd: nn.Module, bwd: nn.Module = None) -> torch.Tensor:
        out = fwd(seq)
        if bwd is not None:
            rev_in = torch.flip(seq, dims=[1])
            rev_out = bwd(rev_in)
            rev_out = torch.flip(rev_out, dims=[1])
            out = 0.5 * (out + rev_out)
        return out

    def _scan_horizontal(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        out_seq = self._run_seq(
            seq,
            self.h_fwd,
            self.h_bwd if self.horizontal_bidirectional else None,
        )
        return out_seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _scan_vertical(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # [B,C,H,W] -> sequence ordered by (column, row)
        seq = x.permute(0, 3, 2, 1).contiguous().view(b, w * h, c)
        out_seq = self._run_seq(
            seq,
            self.v_fwd,
            self.v_bwd if self.vertical_bidirectional else None,
        )
        # inverse mapping to [B,C,H,W]
        return out_seq.view(b, w, h, c).permute(0, 3, 2, 1).contiguous()

    def forward(self, x, time=None):
        if not x.is_cuda:
            raise RuntimeError(
                "SpatialMambaDirectionalBlock requires CUDA tensors because "
                "mamba_ssm/causal_conv1d kernels are CUDA-only in this environment."
            )

        residual = x
        x = self.norm(x)

        mixed = 0.0
        branch_count = 0

        if self.use_horizontal:
            mixed = mixed + self._scan_horizontal(x)
            branch_count += 1

        if self.use_vertical:
            mixed = mixed + self._scan_vertical(x)
            branch_count += 1

        mixed = mixed / float(branch_count)
        mixed = self.out_proj(mixed)
        mixed = self.dropout(mixed)

        return residual + mixed


class UNetModelMambaCompromise(nn.Module):
    """
    UNet reconstruction model with compromise directional Mamba strategy.

    mamba_mode controls where Mamba is inserted:
    - "none": original attention-only behavior.
    - "low": replace selected attention resolutions with Mamba.
    - "medium": low-risk replacement + extra deep-stage Mamba.

    directional_profile controls how many scan directions are used:
    - "speed": horizontal single-direction only.
    - "balanced": shallow horizontal single-direction, deep horizontal bidirectional,
      optional vertical single-direction only at bottleneck.
    - "accuracy": shallow horizontal bidirectional, deep horizontal+vertical bidirectional.
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
        mamba_medium_min_ds=8,
        directional_profile="balanced",
        deep_bidirectional_min_ds=8,
        balanced_bottleneck_vertical=True,
        accuracy_vertical_min_ds=8,
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

        self.directional_profile = _normalize_profile(directional_profile)

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
        self.mamba_medium_min_ds = mamba_medium_min_ds

        self.deep_bidirectional_min_ds = deep_bidirectional_min_ds
        self.balanced_bottleneck_vertical = bool(balanced_bottleneck_vertical)
        self.accuracy_vertical_min_ds = accuracy_vertical_min_ds

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

                context = self._build_context_block(ch=ch, ds=ds, is_bottleneck=False)
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

        middle_context = self._build_context_block(
            ch=ch,
            ds=ds,
            force_attention_fallback=True,
            is_bottleneck=True,
        )
        middle_layers.append(middle_context)

        middle_layers.append(
            ResBlock(
                ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
            )
        )

        if self.mamba_mode == "medium":
            middle_layers.append(self._make_mamba_block(ch=ch, ds=ds, is_bottleneck=True))

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

                context = self._build_context_block(ch=ch, ds=ds, is_bottleneck=False)
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

    def _should_use_mamba(self, ds: int) -> bool:
        if self.mamba_mode == "none":
            return False
        if self.mamba_mode == "low":
            return ds in self.mamba_ds
        return (ds in self.mamba_ds) or (ds >= self.mamba_medium_min_ds)

    def _resolve_direction_policy(self, ds: int, is_bottleneck: bool) -> Tuple[bool, bool, bool, bool]:
        is_deep = ds >= self.deep_bidirectional_min_ds

        if self.directional_profile == "speed":
            return True, False, False, False

        if self.directional_profile == "balanced":
            use_horizontal = True
            horizontal_bidirectional = is_deep
            use_vertical = bool(is_bottleneck and self.balanced_bottleneck_vertical)
            vertical_bidirectional = False
            return use_horizontal, horizontal_bidirectional, use_vertical, vertical_bidirectional

        # accuracy profile
        use_horizontal = True
        horizontal_bidirectional = True
        use_vertical = (ds >= self.accuracy_vertical_min_ds) or (
            is_bottleneck and self.balanced_bottleneck_vertical
        )
        vertical_bidirectional = use_vertical
        return use_horizontal, horizontal_bidirectional, use_vertical, vertical_bidirectional

    def _make_mamba_block(self, ch: int, ds: int, is_bottleneck: bool) -> nn.Module:
        (
            use_horizontal,
            horizontal_bidirectional,
            use_vertical,
            vertical_bidirectional,
        ) = self._resolve_direction_policy(ds=ds, is_bottleneck=is_bottleneck)

        return SpatialMambaDirectionalBlock(
            ch,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand=self.mamba_expand,
            dropout=self.mamba_dropout,
            use_horizontal=use_horizontal,
            horizontal_bidirectional=horizontal_bidirectional,
            use_vertical=use_vertical,
            vertical_bidirectional=vertical_bidirectional,
        )

    def _build_context_block(
        self,
        ch,
        ds,
        force_attention_fallback=False,
        is_bottleneck=False,
    ):
        use_attention = ds in self.attention_ds

        if self._should_use_mamba(ds):
            return self._make_mamba_block(ch=ch, ds=ds, is_bottleneck=is_bottleneck)

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

    model = UNetModelMambaCompromise(
        img_size=args["img_size"],
        base_channels=args["base_channels"],
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        attention_resolutions=args["attention_resolutions"],
        in_channels=3,
        mamba_mode="low",
        directional_profile="balanced",
    )

    print("MAMBA_AVAILABLE:", MAMBA_AVAILABLE)
    print("Params:", sum(p.numel() for p in model.parameters()))

    if torch.cuda.is_available() and MAMBA_AVAILABLE:
        model = model.cuda().eval()
        x = torch.randn(1, 3, 256, 256, device="cuda")
        t = torch.tensor([10], device="cuda")
        with torch.no_grad():
            y = model(x, t)
        print("Output:", y.shape)
    else:
        print("Skip CUDA forward smoke test (requires CUDA + mamba_ssm kernels).")
