import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_beta_thresh import (
    DAGMTestDataset,
    MPDDTestDataset,
    MVTecTestDataset,
    VisATestDataset,
)
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from models.Recon_subnetwork import UNetModel
from models.Seg_subnetwork import SegmentationSubNetwork

try:
    from models.Recon_subnetwork_mamba import UNetModelMamba
except Exception:
    UNetModelMamba = None

try:
    from models.Recon_subnetwork_mamba_compromise import UNetModelMambaCompromise
except Exception:
    UNetModelMambaCompromise = None


MVTEC_CLASSES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

VISA_CLASSES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

MPDD_CLASSES = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
]

DAGM_CLASSES = [f"Class{i}" for i in range(1, 11)]


def defaultdict_from_json(json_dict):
    dd = defaultdict(str)
    dd.update(json_dict)
    return dd


def resolve_config_path(config_arg: str) -> str:
    candidates = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    if os.path.isabs(config_arg):
        candidates.append(config_arg)
    else:
        base_name = os.path.basename(config_arg)
        candidates.extend(
            [
                config_arg,
                os.path.join(cwd, config_arg),
                os.path.join(script_dir, config_arg),
            ]
        )
        if not config_arg.startswith("args/"):
            candidates.extend(
                [
                    os.path.join("args", base_name),
                    os.path.join(cwd, "args", base_name),
                    os.path.join(script_dir, "args", base_name),
                ]
            )

    deduped = []
    seen = set()
    for c in candidates:
        c_norm = os.path.normpath(c)
        if c_norm not in seen:
            deduped.append(c_norm)
            seen.add(c_norm)

    for c in deduped:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(f"Config file not found: {config_arg}. Tried: {deduped}")


def get_arg_int(args, key, default):
    value = args[key]
    if value == "" or value is None:
        return default
    return int(value)


def get_arg_float(args, key, default):
    value = args[key]
    if value == "" or value is None:
        return default
    return float(value)


def get_arg_bool(args, key, default):
    value = args[key]
    if value == "" or value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_auroc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return round(roc_auc_score(y_true, y_score), 3) * 100


def safe_ap(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return round(average_precision_score(y_true, y_score), 3) * 100


def min_max_norm(image):
    a_min = image.min()
    a_max = image.max()
    if a_max - a_min < 1e-12:
        return np.zeros_like(image)
    return (image - a_min) / (a_max - a_min)


def image_transform(image):
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def cvt2heatmap(gray):
    return cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    vmax = np.max(cam)
    if vmax > 0:
        cam = cam / vmax
    return np.uint8(255 * cam)


def pixel_pro(mask, pred):
    mask = np.asarray(mask, dtype=np.bool_)
    pred = np.asarray(pred)

    max_step = 1000
    expect_fpr = 0.3
    max_th = pred.max()
    min_th = pred.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    fprs = []
    binary_score_maps = np.zeros_like(pred, dtype=np.bool_)

    for step in range(max_step):
        thred = max_th - step * delta
        binary_score_maps[pred <= thred] = 0
        binary_score_maps[pred > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = label(mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)

        gt_masks_neg = ~mask
        denom = gt_masks_neg.sum()
        fpr = 0.0 if denom == 0 else np.logical_and(gt_masks_neg, binary_score_maps).sum() / denom
        fprs.append(fpr)
        pros_mean.append(np.array(pro).mean() if len(pro) > 0 else 0.0)

    fprs = np.array(fprs)
    pros_mean = np.array(pros_mean)

    idx = fprs <= expect_fpr
    fprs_selected = fprs[idx]
    pros_mean_selected = pros_mean[idx]
    if len(fprs_selected) < 2:
        return float("nan")

    fprs_selected = min_max_norm(fprs_selected)
    return np.trapz(pros_mean_selected, fprs_selected)


def parse_classes(args, class_set):
    if class_set.strip():
        return [x.strip() for x in class_set.split(",") if x.strip()]

    selected_classes = args.get("selected_classes", [])
    if isinstance(selected_classes, list) and len(selected_classes) > 0:
        return list(selected_classes)

    return list(MVTEC_CLASSES)


def get_class_type(sub_class):
    if sub_class in VISA_CLASSES:
        return "VisA"
    if sub_class in MPDD_CLASSES:
        return "MPDD"
    if sub_class in MVTEC_CLASSES:
        return "MVTec"
    if sub_class in DAGM_CLASSES:
        return "DAGM"
    raise ValueError(f"Unknown class name: {sub_class}")


def build_test_dataset(args, sub_class):
    class_type = get_class_type(sub_class)
    if class_type == "VisA":
        subclass_path = os.path.join(args["visa_root_path"], sub_class)
        return VisATestDataset(subclass_path, sub_class, img_size=args["img_size"]), class_type
    if class_type == "MPDD":
        subclass_path = os.path.join(args["mpdd_root_path"], sub_class)
        return MPDDTestDataset(subclass_path, sub_class, img_size=args["img_size"]), class_type
    if class_type == "MVTec":
        subclass_path = os.path.join(args["mvtec_root_path"], sub_class)
        return MVTecTestDataset(subclass_path, sub_class, img_size=args["img_size"]), class_type

    subclass_path = os.path.join(args["dagm_root_path"], sub_class)
    return DAGMTestDataset(subclass_path, sub_class, img_size=args["img_size"]), class_type


def detect_unet_impl(unet_impl, args):
    if unet_impl != "auto":
        return unet_impl

    variant = str(args.get("recon_mamba_variant", "")).strip().lower()
    if variant == "compromise":
        return "mamba_compromise"

    mode = str(args.get("recon_mamba_mode", "")).strip().lower()
    if mode and mode != "none":
        return "mamba"

    return "baseline"


def build_unet_model(args, device, unet_impl):
    in_channels = args["channels"]

    if unet_impl == "baseline":
        return UNetModel(
            args["img_size"][0],
            args["base_channels"],
            channel_mults=args["channel_mults"],
            dropout=args["dropout"],
            n_heads=args["num_heads"],
            n_head_channels=args["num_head_channels"],
            in_channels=in_channels,
        ).to(device)

    if unet_impl == "mamba":
        if UNetModelMamba is None:
            raise RuntimeError("UNetModelMamba is unavailable. Check mamba dependencies.")

        return UNetModelMamba(
            img_size=args["img_size"][0],
            base_channels=args["base_channels"],
            channel_mults=args["channel_mults"],
            dropout=args["dropout"],
            n_heads=args["num_heads"],
            n_head_channels=args["num_head_channels"],
            in_channels=in_channels,
            mamba_mode=args.get("recon_mamba_mode", "low"),
            mamba_resolutions=args.get("mamba_resolutions", "32,16,8"),
            mamba_d_state=get_arg_int(args, "mamba_d_state", 16),
            mamba_d_conv=get_arg_int(args, "mamba_d_conv", 4),
            mamba_expand=get_arg_int(args, "mamba_expand", 2),
            mamba_dropout=get_arg_float(args, "mamba_dropout", 0.0),
            mamba_bidirectional=get_arg_bool(args, "mamba_bidirectional", True),
            mamba_medium_min_ds=get_arg_int(args, "mamba_medium_min_ds", 8),
            mamba_medium_extra_middle=get_arg_bool(args, "mamba_medium_extra_middle", True),
            mamba_medium_broad_coverage=get_arg_bool(args, "mamba_medium_broad_coverage", True),
        ).to(device)

    if unet_impl == "mamba_compromise":
        if UNetModelMambaCompromise is None:
            raise RuntimeError("UNetModelMambaCompromise is unavailable. Check mamba dependencies.")

        return UNetModelMambaCompromise(
            img_size=args["img_size"][0],
            base_channels=args["base_channels"],
            channel_mults=args["channel_mults"],
            dropout=args["dropout"],
            n_heads=args["num_heads"],
            n_head_channels=args["num_head_channels"],
            in_channels=in_channels,
            mamba_mode=args.get("recon_mamba_mode", "low"),
            mamba_resolutions=args.get("mamba_resolutions", "32,16,8"),
            mamba_d_state=get_arg_int(args, "mamba_d_state", 16),
            mamba_d_conv=get_arg_int(args, "mamba_d_conv", 4),
            mamba_expand=get_arg_int(args, "mamba_expand", 2),
            mamba_dropout=get_arg_float(args, "mamba_dropout", 0.0),
            mamba_medium_min_ds=get_arg_int(args, "mamba_medium_min_ds", 8),
            directional_profile=args.get("directional_profile", "balanced"),
            deep_bidirectional_min_ds=get_arg_int(args, "deep_bidirectional_min_ds", 8),
            balanced_bottleneck_vertical=get_arg_bool(args, "balanced_bottleneck_vertical", True),
            accuracy_vertical_min_ds=get_arg_int(args, "accuracy_vertical_min_ds", 8),
        ).to(device)

    raise ValueError(f"Unsupported unet implementation: {unet_impl}")


def load_checkpoint(args, device, sub_class, checkpoint_type):
    ck_path = (
        f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/'
        f'params-{checkpoint_type}.pt'
    )
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")

    print(f"[checkpoint] {ck_path}")
    return torch.load(ck_path, map_location=device, weights_only=False)


def evaluate_one_class(
    args,
    unet_model,
    seg_model,
    test_loader,
    sub_class,
    class_type,
    checkpoint_type,
    device,
    output_tag,
    heatmap_sigma,
):
    normal_t = args["eval_normal_t"]
    noiser_t = args["eval_noisier_t"]

    vis_dir = os.path.join(
        args["output_path"],
        "metrics",
        f'ARGS={args["arg_num"]}',
        sub_class,
        f"visualization_{normal_t}_{noiser_t}_{args['condition_w']}condition_{checkpoint_type}ck_{output_tag}",
    )
    os.makedirs(vis_dir, exist_ok=True)

    in_channels = args["channels"]
    betas = get_beta_schedule(args["T"], args["beta_schedule"])
    ddpm_sample = GaussianDiffusionModel(
        args["img_size"],
        betas,
        loss_weight=args["loss_weight"],
        loss_type=args["loss-type"],
        noise=args["noise_fn"],
        img_channels=in_channels,
    )

    total_image_pred = np.array([])
    total_image_gt = np.array([])
    total_pixel_gt = np.array([])
    total_pixel_pred = np.array([])
    gt_matrix_pixel = []
    pred_matrix_pixel = []

    unet_model.eval()
    seg_model.eval()

    tbar = tqdm(test_loader, desc=f"{sub_class} eval", leave=False)
    for sample in tbar:
        image = sample["image"].to(device)
        target = sample["has_anomaly"].to(device)
        gt_mask = sample["mask"].to(device)
        image_path = sample["file_name"]

        normal_t_tensor = torch.tensor([normal_t], device=image.device).repeat(image.shape[0])
        noiser_t_tensor = torch.tensor([noiser_t], device=image.device).repeat(image.shape[0])

        (
            _loss,
            pred_x_0_condition,
            pred_x_0_normal,
            pred_x_0_noisier,
            x_normal_t,
            x_noiser_t,
            pred_x_t_noisier,
        ) = ddpm_sample.norm_guided_one_step_denoising_eval(
            unet_model, image, normal_t_tensor, noiser_t_tensor, args
        )

        pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1))
        out_mask = pred_mask

        gt_matrix_pixel.extend(gt_mask[0].detach().cpu().numpy().astype(int))
        pred_matrix_pixel.extend(out_mask[0].detach().cpu().numpy())

        topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
        topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
        image_score = torch.mean(topk_out_mask)

        total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
        total_image_gt = np.append(total_image_gt, target[0].detach().cpu().numpy())

        flatten_pred_mask = out_mask[0].flatten().detach().cpu().numpy()
        flatten_gt_mask = gt_mask[0].flatten().detach().cpu().numpy().astype(int)
        total_pixel_gt = np.append(total_pixel_gt, flatten_gt_mask)
        total_pixel_pred = np.append(total_pixel_pred, flatten_pred_mask)

        x_noiser_t_img = image_transform(x_noiser_t.detach().cpu().numpy()[0])
        x_normal_t_img = image_transform(x_normal_t.detach().cpu().numpy()[0])
        pred_x_t_noisier_img = image_transform(pred_x_t_noisier.detach().cpu().numpy()[0])
        raw_image = image_transform(image.detach().cpu().numpy()[0])

        ano_map_gray = gaussian_filter(out_mask[0, 0, :, :].detach().cpu().numpy(), sigma=heatmap_sigma)
        ano_map_gray = min_max_norm(ano_map_gray)
        heatmap_bgr = cvt2heatmap(ano_map_gray * 255.0)

        image_rgb = np.uint8(np.transpose(raw_image, (1, 2, 0)))
        overlay_rgb = show_cam_on_image(image_rgb[..., ::-1], heatmap_bgr)[..., ::-1]
        heatmap_rgb = heatmap_bgr[..., ::-1]

        recon_condition = image_transform(pred_x_0_condition.detach().cpu().numpy()[0])
        recon_normal_t = image_transform(pred_x_0_normal.detach().cpu().numpy()[0])
        recon_noisier_t = image_transform(pred_x_0_noisier.detach().cpu().numpy()[0])

        if isinstance(image_path, (list, tuple)):
            image_path_str = str(image_path[0])
        else:
            image_path_str = str(image_path)

        parts = Path(image_path_str).parts
        merged_name = "_".join(parts[-4:]) if len(parts) >= 4 else Path(image_path_str).name
        stem = Path(merged_name).stem

        panel_path = os.path.join(vis_dir, f"{stem}_panel.png")
        input_path = os.path.join(vis_dir, f"{stem}_input.png")
        gt_path = os.path.join(vis_dir, f"{stem}_gt.png")
        heatmap_path = os.path.join(vis_dir, f"{stem}_heatmap.png")
        overlay_path = os.path.join(vis_dir, f"{stem}_overlay.png")

        cv2.imwrite(input_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(gt_path, (gt_mask[0][0] * 255.0).detach().cpu().numpy().astype(np.uint8))
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        fig, axes = plt.subplots(2, 5)
        fig.suptitle(f"image score: {float(image_score.detach().cpu().numpy()):.6f}")

        axes[0][0].imshow(image_rgb)
        axes[0][0].set_title("Input")
        axes[0][0].axis("off")

        axes[0][1].imshow((gt_mask[0][0] * 255.0).detach().cpu().numpy().astype(np.uint8), cmap="gray")
        axes[0][1].set_title("GT")
        axes[0][1].axis("off")

        axes[0][2].imshow(x_normal_t_img.transpose(1, 2, 0).astype(np.uint8))
        axes[0][2].set_title("x_normal_t")
        axes[0][2].axis("off")

        axes[0][3].imshow(x_noiser_t_img.transpose(1, 2, 0).astype(np.uint8))
        axes[0][3].set_title("x_noiser_t")
        axes[0][3].axis("off")

        axes[0][4].imshow(pred_x_t_noisier_img.transpose(1, 2, 0).astype(np.uint8))
        axes[0][4].set_title("pred_x_t_noisier")
        axes[0][4].axis("off")

        axes[1][0].imshow(overlay_rgb)
        axes[1][0].set_title("heatmap")
        axes[1][0].axis("off")

        axes[1][1].imshow((out_mask[0][0] * 255.0).detach().cpu().numpy().astype(np.uint8), cmap="gray")
        axes[1][1].set_title("out_mask")
        axes[1][1].axis("off")

        axes[1][2].imshow(recon_normal_t.transpose(1, 2, 0).astype(np.uint8))
        axes[1][2].set_title("recon_normal")
        axes[1][2].axis("off")

        axes[1][3].imshow(recon_noisier_t.transpose(1, 2, 0).astype(np.uint8))
        axes[1][3].set_title("recon_noisier")
        axes[1][3].axis("off")

        axes[1][4].imshow(recon_condition.transpose(1, 2, 0).astype(np.uint8))
        axes[1][4].set_title("recon_con")
        axes[1][4].axis("off")

        fig.set_size_inches(15, 6)
        fig.tight_layout()
        fig.savefig(panel_path)
        plt.close(fig)

    auroc_image = safe_auroc(total_image_gt, total_image_pred)
    auroc_pixel = safe_auroc(total_pixel_gt, total_pixel_pred)
    ap_pixel = safe_ap(total_pixel_gt, total_pixel_pred)

    try:
        aupro_pixel = round(pixel_pro(gt_matrix_pixel, pred_matrix_pixel), 3) * 100
    except Exception:
        aupro_pixel = float("nan")

    print(f"[{sub_class}] Image AUROC: {auroc_image}")
    print(f"[{sub_class}] Pixel AUROC: {auroc_pixel}")
    print(f"[{sub_class}] Pixel AP: {ap_pixel}")
    print(f"[{sub_class}] Pixel AUPRO: {aupro_pixel}")

    return {
        "classname": sub_class,
        "class_type": class_type,
        "Image-AUROC": auroc_image,
        "Pixel-AUROC": auroc_pixel,
        "Pixel-AUPRO": aupro_pixel,
        "Pixel-AP": ap_pixel,
        "checkpoint_type": checkpoint_type,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluation script with selectable config/checkpoint/model")
    parser.add_argument("--config", default="args/args1.json", help="Path to config json")
    parser.add_argument("--checkpoint-type", default="best", choices=["best", "last"], help="Checkpoint type")
    parser.add_argument("--class-set", default="", help="Comma-separated class names. Overrides config selected_classes")
    parser.add_argument(
        "--unet-impl",
        default="auto",
        choices=["auto", "baseline", "mamba", "mamba_compromise"],
        help="UNet implementation selection",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Num workers for evaluation dataloader")
    parser.add_argument("--output-tag", default="eval_selectable", help="Suffix tag for visualization folder naming")
    parser.add_argument("--heatmap-sigma", type=float, default=4.0, help="Gaussian sigma for anomaly map smoothing")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip classes with missing checkpoints instead of stopping",
    )
    cli_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = resolve_config_path(cli_args.config)
    with open(config_path, "r") as f:
        raw_args = json.load(f)

    base_name = os.path.splitext(os.path.basename(config_path))[0]
    arg_num = raw_args.get("arg_num", "")
    if not arg_num:
        arg_num = base_name.replace("args", "").strip("_")
        if not arg_num:
            arg_num = base_name
    raw_args["arg_num"] = arg_num

    args = defaultdict_from_json(raw_args)

    if args["loss-type"] == "" and args["loss_type"] != "":
        args["loss-type"] = args["loss_type"]

    classes = parse_classes(args, cli_args.class_set)
    unet_impl = detect_unet_impl(cli_args.unet_impl, args)

    print("Using config:", config_path)
    print("arg_num:", args["arg_num"])
    print("checkpoint_type:", cli_args.checkpoint_type)
    print("unet_impl:", unet_impl)
    print("classes:", classes)

    summary_rows = []

    for sub_class in classes:
        try:
            checkpoint = load_checkpoint(args, device, sub_class, cli_args.checkpoint_type)
        except FileNotFoundError as e:
            if cli_args.skip_missing:
                print(f"[skip] {e}")
                continue
            raise

        test_dataset, class_type = build_test_dataset(args, sub_class)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, cli_args.num_workers),
        )

        unet_model = build_unet_model(args, device, unet_impl)
        seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)

        unet_model.load_state_dict(checkpoint["unet_model_state_dict"])
        seg_model.load_state_dict(checkpoint["seg_model_state_dict"])

        row = evaluate_one_class(
            args=args,
            unet_model=unet_model,
            seg_model=seg_model,
            test_loader=test_loader,
            sub_class=sub_class,
            class_type=class_type,
            checkpoint_type=cli_args.checkpoint_type,
            device=device,
            output_tag=cli_args.output_tag,
            heatmap_sigma=cli_args.heatmap_sigma,
        )
        row["n_epoch"] = checkpoint.get("n_epoch", "")
        summary_rows.append(row)

    if len(summary_rows) == 0:
        print("No class was evaluated. Nothing to write.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.join(
        args["output_path"],
        "metrics",
        f'ARGS={args["arg_num"]}',
        f'{args["eval_normal_t"]}_{args["eval_noisier_t"]}t_{args["condition_w"]}condition_{cli_args.checkpoint_type}ck_{cli_args.output_tag}.csv',
    )
    os.makedirs(os.path.dirname(summary_out), exist_ok=True)
    summary_df.to_csv(summary_out, index=False)
    print("Summary saved to:", summary_out)


if __name__ == "__main__":
    main()
