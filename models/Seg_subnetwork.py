
import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_dilation_list(value, default=(1, 2)):
    if value is None:
        return list(default)

    if isinstance(value, (list, tuple)):
        vals = []
        for x in value:
            try:
                xi = int(x)
            except Exception:
                continue
            if xi > 0:
                vals.append(xi)
        return vals or list(default)

    text = str(value).strip()
    if text == "":
        return list(default)

    vals = []
    for item in text.split(","):
        item = item.strip()
        if item == "":
            continue
        try:
            xi = int(item)
        except Exception:
            continue
        if xi > 0:
            vals.append(xi)
    return vals or list(default)


def _parse_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _build_group_norm(num_channels, max_groups=8):
    channels = int(num_channels)
    groups = max(1, int(max_groups))
    groups = min(groups, channels)
    while groups > 1 and (channels % groups != 0):
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _replace_norm_activation(module, norm_mode="batch", act_mode="relu", groupnorm_groups=8):
    norm_mode = str(norm_mode).strip().lower()
    act_mode = str(act_mode).strip().lower()

    for name, child in module.named_children():
        replaced = False

        if norm_mode in {"group", "groupnorm", "gn"} and isinstance(child, nn.BatchNorm2d):
            new_norm = _build_group_norm(child.num_features, max_groups=groupnorm_groups)
            if child.affine:
                with torch.no_grad():
                    new_norm.weight.copy_(child.weight)
                    new_norm.bias.copy_(child.bias)
            setattr(module, name, new_norm)
            replaced = True
        elif act_mode in {"silu", "swish"} and isinstance(child, nn.ReLU):
            setattr(module, name, nn.SiLU(inplace=child.inplace))
            replaced = True

        if not replaced:
            _replace_norm_activation(child, norm_mode=norm_mode, act_mode=act_mode, groupnorm_groups=groupnorm_groups)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LiteDualContext(nn.Module):
    """
    Lightweight multi-dilation context block for bottleneck enhancement.
    """

    def __init__(self, channels, dilations=(1, 2)):
        super().__init__()
        branch_out = max(16, channels // 4)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels, branch_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_out),
                    nn.SiLU(inplace=True),
                )
            )

        merged_channels = channels + len(self.branches) * branch_out
        self.project = nn.Sequential(
            nn.Conv2d(merged_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        feats = [x]
        for branch in self.branches:
            feats.append(branch(x))
        merged = torch.cat(feats, dim=1)
        return x + self.project(merged)


class BottleneckMultiScaleContext(nn.Module):
    """
    Multi-scale context block for bottleneck features.
    """

    def __init__(self, channels, dilations=(1, 2, 3), use_global_pool=True):
        super().__init__()
        branch_out = max(16, channels // 4)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels, branch_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_out),
                    nn.SiLU(inplace=True),
                )
            )

        self.global_pool = None
        self.use_global_pool = _parse_bool(use_global_pool, default=True)
        if self.use_global_pool:
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, branch_out, kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
            )

        merged_channels = channels + len(self.branches) * branch_out + (branch_out if self.global_pool is not None else 0)
        self.project = nn.Sequential(
            nn.Conv2d(merged_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        feats = [x]
        for branch in self.branches:
            feats.append(branch(x))

        if self.global_pool is not None:
            pooled = self.global_pool(x)
            pooled = nn.functional.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(pooled)

        merged = torch.cat(feats, dim=1)
        return x + self.project(merged)

class SegmentationSubNetwork(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        use_fp16=False,
        base_channels=64,
        out_features=False,
        arch_mode="legacy",
        decoder_mode="legacy",
        unet3p_stage_channels=0,
        boundary_head_enable=False,
        boundary_gate_alpha=0.5,
        norm_mode="batch",
        act_mode="relu",
        groupnorm_groups=8,
        lite_stem_channels=16,
        lite_fuse_channels=0,
        lite_use_context=True,
        lite_context_dilations="1,2",
        bottleneck_context_enable=False,
        bottleneck_context_dilations="1,2,3",
        bottleneck_context_use_global=True,
    ):
        super(SegmentationSubNetwork, self).__init__()
        self.in_channels = int(in_channels)
        self.arch_mode = str(arch_mode).strip().lower()
        if self.arch_mode not in {"legacy", "lite_dual"}:
            raise ValueError(f"Unsupported seg arch_mode: {arch_mode}")

        self.decoder_mode = str(decoder_mode).strip().lower()
        if self.decoder_mode not in {"legacy", "unet3p"}:
            raise ValueError(f"Unsupported seg decoder_mode: {decoder_mode}")

        self.norm_mode = str(norm_mode).strip().lower()
        self.act_mode = str(act_mode).strip().lower()
        self.groupnorm_groups = max(1, int(groupnorm_groups))

        base_width = base_channels
        self.encoder_segment = EncoderSegmentation(self.in_channels, base_width)

        stage_channels = 0
        try:
            stage_channels = int(unet3p_stage_channels)
        except Exception:
            stage_channels = 0
        if stage_channels <= 0:
            stage_channels = base_width

        self.decoder_feature_channels = stage_channels if self.decoder_mode == "unet3p" else base_width

        if self.decoder_mode == "unet3p":
            self.decoder_segment = DecoderSegmentationUNet3Plus(
                base_width,
                out_channels=out_channels,
                stage_channels=stage_channels,
            )
        else:
            self.decoder_segment = DecoderSegmentation(base_width, out_channels=out_channels)
        self.out_features = out_features

        # Optional lightweight dual-input fusion before the shared encoder.
        self.lite_pair_channels = None
        self.lite_img_stem = None
        self.lite_recon_stem = None
        self.lite_diff_stem = None
        self.lite_fuse = None
        self.lite_context = None
        self.bottleneck_context = None
        self.boundary_head = None
        try:
            self.boundary_gate_alpha = max(0.0, float(boundary_gate_alpha))
        except Exception:
            self.boundary_gate_alpha = 0.5

        if self.arch_mode == "lite_dual":
            if self.in_channels % 2 != 0:
                raise ValueError(
                    "lite_dual requires even in_channels so image/recon can be split equally. "
                    f"Got in_channels={self.in_channels}."
                )

            self.lite_pair_channels = self.in_channels // 2
            stem_channels = max(4, int(lite_stem_channels))
            diff_channels = max(4, stem_channels // 2)

            try:
                fuse_channels = int(lite_fuse_channels)
            except Exception:
                fuse_channels = 0
            if fuse_channels <= 0:
                fuse_channels = max(self.in_channels * 2, stem_channels * 2)

            self.lite_img_stem = ConvBNAct(self.lite_pair_channels, stem_channels, kernel_size=3, padding=1)
            self.lite_recon_stem = ConvBNAct(self.lite_pair_channels, stem_channels, kernel_size=3, padding=1)
            self.lite_diff_stem = ConvBNAct(self.lite_pair_channels, diff_channels, kernel_size=3, padding=1)

            fused_in = stem_channels + stem_channels + diff_channels
            self.lite_fuse = nn.Sequential(
                ConvBNAct(fused_in, fuse_channels, kernel_size=1, padding=0),
                nn.Conv2d(fuse_channels, self.in_channels, kernel_size=1, padding=0),
            )

            if _parse_bool(lite_use_context, default=True):
                dilations = _parse_dilation_list(lite_context_dilations, default=(1, 2))
                self.lite_context = LiteDualContext(base_width * 8, dilations=dilations)

        if _parse_bool(bottleneck_context_enable, default=False):
            dilations = _parse_dilation_list(bottleneck_context_dilations, default=(1, 2, 3))
            self.bottleneck_context = BottleneckMultiScaleContext(
                base_width * 8,
                dilations=dilations,
                use_global_pool=bottleneck_context_use_global,
            )

        if _parse_bool(boundary_head_enable, default=False):
            c = int(self.decoder_feature_channels)
            self.boundary_head = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, out_channels, kernel_size=1, padding=0),
            )

        if self.norm_mode in {"group", "groupnorm", "gn"} or self.act_mode in {"silu", "swish"}:
            _replace_norm_activation(
                self,
                norm_mode=self.norm_mode,
                act_mode=self.act_mode,
                groupnorm_groups=self.groupnorm_groups,
            )

    def _forward_lite_dual_input(self, x):
        if self.lite_pair_channels is None:
            return x

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"lite_dual expects input channels={self.in_channels}, got {x.shape[1]}."
            )

        image = x[:, : self.lite_pair_channels]
        recon = x[:, self.lite_pair_channels : self.lite_pair_channels * 2]
        diff = (image - recon).abs()

        fused = torch.cat(
            [
                self.lite_img_stem(image),
                self.lite_recon_stem(recon),
                self.lite_diff_stem(diff),
            ],
            dim=1,
        )
        return self.lite_fuse(fused)

    def forward(self, x):
        if self.arch_mode == "lite_dual":
            x = self._forward_lite_dual_input(x)

        b1,b2,b3,b4,b5,b6 = self.encoder_segment(x)

        if self.lite_context is not None:
            b6 = self.lite_context(b6)

        if self.bottleneck_context is not None:
            b6 = self.bottleneck_context(b6)

        if self.boundary_head is not None:
            output_segment, decoder_feat = self.decoder_segment(b1, b2, b3, b4, b5, b6, return_feature=True)
            boundary_map = torch.sigmoid(self.boundary_head(decoder_feat))
            output_segment = torch.clamp(
                output_segment * (1.0 + self.boundary_gate_alpha * boundary_map),
                min=0.0,
                max=1.0,
            )
        else:
            output_segment = self.decoder_segment(b1,b2,b3,b4,b5,b6)

        if self.out_features:
            return output_segment, b2, b3, b4, b5, b6
        else:
            return output_segment

class EncoderSegmentation(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderSegmentation, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1,b2,b3,b4,b5,b6


class FullScaleFusionBlock(nn.Module):
    def __init__(self, source_channels, stage_channels, include_prev=False, prev_channels=None):
        super().__init__()
        self.include_prev = bool(include_prev)
        self.source_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, stage_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(stage_channels),
                    nn.ReLU(inplace=True),
                )
                for ch in source_channels
            ]
        )

        self.prev_proj = None
        total_inputs = len(source_channels)
        if self.include_prev:
            prev_channels = stage_channels if prev_channels is None else int(prev_channels)
            self.prev_proj = nn.Sequential(
                nn.Conv2d(prev_channels, stage_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stage_channels),
                nn.ReLU(inplace=True),
            )
            total_inputs += 1

        fuse_in = total_inputs * stage_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, stage_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stage_channels * 2, stage_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _resize_to(feat, target_size):
        if feat.shape[-2:] == target_size:
            return feat
        return F.interpolate(feat, size=target_size, mode="bilinear", align_corners=True)

    def forward(self, sources, target_size, prev=None):
        feats = []
        for src, proj in zip(sources, self.source_projs):
            feats.append(proj(self._resize_to(src, target_size)))

        if self.include_prev:
            if prev is None:
                raise ValueError("FullScaleFusionBlock requires prev feature when include_prev=True.")
            feats.append(self.prev_proj(self._resize_to(prev, target_size)))

        return self.fuse(torch.cat(feats, dim=1))


class DecoderSegmentationUNet3Plus(nn.Module):
    """
    UNet3+-style full-scale fusion decoder.
    """

    def __init__(self, base_width, out_channels=1, stage_channels=64):
        super(DecoderSegmentationUNet3Plus, self).__init__()
        stage_channels = max(8, int(stage_channels))
        source_channels = [
            base_width,
            base_width * 2,
            base_width * 4,
            base_width * 8,
            base_width * 8,
            base_width * 8,
        ]

        self.fuse5 = FullScaleFusionBlock(source_channels, stage_channels, include_prev=False)
        self.fuse4 = FullScaleFusionBlock(source_channels, stage_channels, include_prev=True, prev_channels=stage_channels)
        self.fuse3 = FullScaleFusionBlock(source_channels, stage_channels, include_prev=True, prev_channels=stage_channels)
        self.fuse2 = FullScaleFusionBlock(source_channels, stage_channels, include_prev=True, prev_channels=stage_channels)
        self.fuse1 = FullScaleFusionBlock(source_channels, stage_channels, include_prev=True, prev_channels=stage_channels)

        self.fin_out = nn.Sequential(
            nn.Conv2d(stage_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, b1, b2, b3, b4, b5, b6, return_feature=False):
        enc_feats = [b1, b2, b3, b4, b5, b6]

        d5 = self.fuse5(enc_feats, target_size=b5.shape[-2:])
        d4 = self.fuse4(enc_feats, target_size=b4.shape[-2:], prev=d5)
        d3 = self.fuse3(enc_feats, target_size=b3.shape[-2:], prev=d4)
        d2 = self.fuse2(enc_feats, target_size=b2.shape[-2:], prev=d3)
        d1 = self.fuse1(enc_feats, target_size=b1.shape[-2:], prev=d2)

        out = self.fin_out(d1)
        if return_feature:
            return out, d1
        return out

class DecoderSegmentation(nn.Module):
    def __init__(self, base_width, out_channels=1):  
        super(DecoderSegmentation, self).__init__()

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1), 
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        
        #cat 96
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1), 
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )


        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )



        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        ,nn.Sigmoid()
        )

    def forward(self, b1,b2,b3,b4,b5,b6, return_feature=False):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b,b5),dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1,b4),dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2,b3),dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3,b2),dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4,b1),dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        if return_feature:
            return out, db4
        return out

