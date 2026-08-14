import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Recon_subnetwork import UNetModel


class StructureDecoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class ReconDualBranchModel(nn.Module):
    """
    Reconstruction backbone (diffusion noise predictor) + structure branch (+ optional light segmentation head).

    - forward(x_t, t): returns diffusion noise prediction (same contract as UNetModel)
    - predict_structure(recon): predicts structure map from reconstructed image
    - predict_segmentation(recon, image): predicts anomaly probability map from reconstruction
    - get_last_feature(): returns cached decoder feature from latest forward for feature KD
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
        in_channels=3,
        structure_hidden_channels=32,
        feature_kd_channels=64,
        use_segmentation_head=False,
        seg_input_mode="concat",
        seg_hidden_channels=32,
        seg_init_temperature=1.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.backbone = UNetModel(
            img_size=img_size,
            base_channels=base_channels,
            conv_resample=conv_resample,
            n_heads=n_heads,
            n_head_channels=n_head_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            attention_resolutions=attention_resolutions,
            biggan_updown=biggan_updown,
            in_channels=in_channels,
        )

        self.structure_head = StructureDecoder(
            in_channels=in_channels,
            hidden_channels=structure_hidden_channels,
            out_channels=1,
        )

        self.use_segmentation_head = bool(use_segmentation_head)
        self.seg_input_mode = str(seg_input_mode).strip().lower() if seg_input_mode is not None else "concat"
        if self.seg_input_mode not in {"concat", "recon"}:
            raise ValueError(f"Unsupported seg_input_mode: {seg_input_mode}. Use 'concat' or 'recon'.")

        self.seg_head = None
        self.seg_temperature_raw = None
        if self.use_segmentation_head:
            seg_in_channels = self.in_channels * 2 if self.seg_input_mode == "concat" else self.in_channels
            self.seg_head = SegmentationDecoder(
                in_channels=seg_in_channels,
                hidden_channels=int(seg_hidden_channels),
                out_channels=1,
            )

            seg_init_temperature = max(float(seg_init_temperature), 1e-4)
            raw_init = torch.log(torch.exp(torch.tensor(seg_init_temperature)) - 1.0)
            self.seg_temperature_raw = nn.Parameter(raw_init.view(1))

        self._feature_cache = None
        self._feature_hook_handle = None
        self._init_feature_hook()
        self.feature_kd_channels = int(feature_kd_channels)

    def _init_feature_hook(self):
        # Capture the decoder feature right before the final prediction conv.
        if isinstance(self.backbone.out, nn.Sequential) and len(self.backbone.out) > 1:
            target_layer = self.backbone.out[1]

            def _save_feature(_module, _inputs, output):
                self._feature_cache = output

            self._feature_hook_handle = target_layer.register_forward_hook(_save_feature)

    def forward(self, x, time):
        self._feature_cache = None
        return self.backbone(x, time)

    def get_last_feature(self):
        if self._feature_cache is None:
            return None
        feat = self._feature_cache
        if self.feature_kd_channels <= 0:
            return feat

        c = feat.shape[1]
        target_c = self.feature_kd_channels
        if c == target_c:
            return feat
        if c > target_c:
            return feat[:, :target_c]

        pad_c = target_c - c
        pad = torch.zeros(
            feat.shape[0],
            pad_c,
            feat.shape[2],
            feat.shape[3],
            device=feat.device,
            dtype=feat.dtype,
        )
        return torch.cat([feat, pad], dim=1)

    def predict_structure(self, recon):
        return self.structure_head(recon)

    def has_segmentation_head(self):
        return self.seg_head is not None

    def get_seg_temperature(self):
        if self.seg_temperature_raw is None:
            return None
        return F.softplus(self.seg_temperature_raw) + 1e-4

    def predict_segmentation(self, recon, image=None, return_logits=False):
        if self.seg_head is None:
            raise AttributeError("Model has no segmentation head")

        if self.seg_input_mode == "concat":
            if image is None:
                image = recon
            seg_input = torch.cat([image, recon], dim=1)
        else:
            seg_input = recon

        seg_logits = self.seg_head(seg_input)
        seg_temperature = self.get_seg_temperature()
        if seg_temperature is not None:
            seg_logits = seg_logits / seg_temperature

        if return_logits:
            return seg_logits
        return torch.sigmoid(seg_logits)
