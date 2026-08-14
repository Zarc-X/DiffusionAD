from random import seed
import argparse
import os
import json
import time
from collections import defaultdict


def sanitize_thread_env():
    # libgomp requires positive thread counts.
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        raw = os.environ.get(env_name, "").strip()
        try:
            value = int(raw) if raw else None
        except ValueError:
            value = None
        if value is None or value <= 0:
            os.environ[env_name] = "1"


sanitize_thread_env()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from data.dataset_beta_thresh import (
    MVTecTestDataset,
    VisATestDataset,
    DAGMTestDataset,
    MPDDTestDataset,
)
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from models.Recon_subnetwork import UNetModel
from models.Recon_dualbranch import ReconDualBranchModel


def resolve_config_path(config_arg: str) -> str:
    candidates = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    if os.path.isabs(config_arg):
        candidates.append(config_arg)
    else:
        base_name = os.path.basename(config_arg)
        candidates.extend([
            config_arg,
            os.path.join(cwd, config_arg),
            os.path.join(script_dir, config_arg),
        ])
        if not config_arg.startswith("args/"):
            candidates.extend([
                os.path.join("args", base_name),
                os.path.join(cwd, "args", base_name),
                os.path.join(script_dir, "args", base_name),
            ])

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


def defaultdict_from_json(json_dict):
    dd = defaultdict(str)
    dd.update(json_dict)
    return dd


def get_arg_int(args, key, default):
    value = args[key]
    if value == "" or value is None:
        return default
    return int(value)


def parse_channel_mults(value, fallback):
    if value == "" or value is None:
        return tuple(int(v) for v in fallback)
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return tuple(int(v) for v in fallback)
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    raise ValueError(f"Unsupported channel_mults type: {type(value)}")


def safe_auroc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return round(roc_auc_score(y_true, y_score), 3) * 100


def safe_ap(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return round(average_precision_score(y_true, y_score), 3) * 100


def safe_best_f1(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denom = precision + recall
    f1_scores = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    best_idx = int(np.nanargmax(f1_scores))
    best_f1 = round(float(f1_scores[best_idx]), 3) * 100

    if thresholds.size == 0:
        best_threshold = float("nan")
    else:
        threshold_idx = min(best_idx, thresholds.size - 1)
        best_threshold = round(float(thresholds[threshold_idx]), 6)

    return best_f1, best_threshold


def build_test_dataset(args, sub_class):
    mvtec_classes = [
        "carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule",
        "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper",
    ]
    visa_classes = [
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
        "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
    ]
    mpdd_classes = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
    dagm_classes = [f"Class{i}" for i in range(1, 11)]

    if sub_class in visa_classes:
        subclass_path = os.path.join(args["visa_root_path"], sub_class)
        testing_dataset = VisATestDataset(subclass_path, sub_class, img_size=args["img_size"])
    elif sub_class in mpdd_classes:
        subclass_path = os.path.join(args["mpdd_root_path"], sub_class)
        testing_dataset = MPDDTestDataset(subclass_path, sub_class, img_size=args["img_size"])
    elif sub_class in mvtec_classes:
        subclass_path = os.path.join(args["mvtec_root_path"], sub_class)
        testing_dataset = MVTecTestDataset(subclass_path, sub_class, img_size=args["img_size"])
    elif sub_class in dagm_classes:
        subclass_path = os.path.join(args["dagm_root_path"], sub_class)
        testing_dataset = DAGMTestDataset(subclass_path, sub_class, img_size=args["img_size"])
    else:
        raise ValueError(f"Unknown class name: {sub_class}")

    return testing_dataset


def load_student_model(args, sub_class, checkpoint_type, device):
    in_channels = args["channels"]
    student_base_channels = get_arg_int(args, "student_base_channels", get_arg_int(args, "base_channels", 128))
    student_channel_mults = parse_channel_mults(args["student_channel_mults"], args["channel_mults"])

    ckpt_path = (
        f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/'
        f'{sub_class}/params-{checkpoint_type}.pt'
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "student_model_state_dict" in ckpt:
        state = ckpt["student_model_state_dict"]
    elif "unet_model_state_dict" in ckpt:
        state = ckpt["unet_model_state_dict"]
    else:
        raise KeyError(f"No student/unet state dict found in {ckpt_path}")

    if any(k.startswith("backbone.") for k in state.keys()):
        use_segmentation_head = any(k.startswith("seg_head.") or k.startswith("seg_temperature_raw") for k in state.keys())
        model = ReconDualBranchModel(
            img_size=args["img_size"][0],
            base_channels=student_base_channels,
            channel_mults=student_channel_mults,
            dropout=args["dropout"],
            n_heads=args["num_heads"],
            n_head_channels=args["num_head_channels"],
            in_channels=in_channels,
            structure_hidden_channels=get_arg_int(args, "student_structure_hidden_channels", 32),
            feature_kd_channels=get_arg_int(args, "student_feature_kd_channels", 64),
            use_segmentation_head=use_segmentation_head,
            seg_input_mode=args.get("seg_input_mode", "concat") if hasattr(args, "get") else "concat",
            seg_hidden_channels=get_arg_int(args, "student_seg_hidden_channels", 24),
            seg_init_temperature=1.0,
        ).to(device)
    else:
        model = UNetModel(
            args["img_size"][0],
            student_base_channels,
            channel_mults=student_channel_mults,
            dropout=args["dropout"],
            n_heads=args["num_heads"],
            n_head_channels=args["num_head_channels"],
            in_channels=in_channels,
        ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt_path


def collect_residual_maps(test_loader, args, model, device):
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

    maps_mean = []
    maps_max = []
    masks = []
    img_gt = []
    total_infer_time = 0.0
    total_images = 0

    with torch.no_grad():
        tbar = tqdm(test_loader, desc="Collect residual maps", leave=False)
        for sample in tbar:
            image = sample["image"].to(device)
            target = sample["has_anomaly"].to(device)
            gt_mask = sample["mask"].to(device)

            normal_t_tensor = torch.tensor([args["eval_normal_t"]], device=image.device).repeat(image.shape[0])
            noiser_t_tensor = torch.tensor([args["eval_noisier_t"]], device=image.device).repeat(image.shape[0])

            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_start = time.perf_counter()

            (
                _loss,
                pred_x_0_condition,
                _pred_x_0_normal,
                _pred_x_0_noisier,
                _x_normal_t,
                _x_noiser_t,
                _pred_x_t_noisier,
            ) = ddpm_sample.norm_guided_one_step_denoising_eval(
                model,
                image,
                normal_t_tensor,
                noiser_t_tensor,
                args,
            )

            residual = torch.abs(image - pred_x_0_condition)
            residual_mean = torch.mean(residual, dim=1)
            residual_max = torch.max(residual, dim=1)[0]

            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_end = time.perf_counter()

            total_infer_time += infer_end - infer_start
            total_images += int(image.shape[0])

            maps_mean.append(residual_mean.detach().cpu().numpy())
            maps_max.append(residual_max.detach().cpu().numpy())
            masks.append(gt_mask[:, 0].detach().cpu().numpy().astype(np.int32))
            img_gt.append(target.view(-1).detach().cpu().numpy().astype(np.int32))

    maps_mean = np.concatenate(maps_mean, axis=0)
    maps_max = np.concatenate(maps_max, axis=0)
    masks = np.concatenate(masks, axis=0)
    img_gt = np.concatenate(img_gt, axis=0)

    base_fps = total_images / total_infer_time if total_infer_time > 0 else float("nan")
    base_ms = (total_infer_time / total_images) * 1000.0 if total_images > 0 else float("nan")

    return {
        "residual_mean": maps_mean,
        "residual_max": maps_max,
        "mask": masks,
        "image_gt": img_gt,
        "base_fps": base_fps,
        "base_ms_per_img": base_ms,
    }


def normalize_maps(maps, mode):
    if mode == "none":
        return maps

    out = maps.copy()
    if mode == "per_image_minmax":
        mins = out.reshape(out.shape[0], -1).min(axis=1)
        maxs = out.reshape(out.shape[0], -1).max(axis=1)
        denom = np.maximum(maxs - mins, 1e-12)
        out = (out - mins[:, None, None]) / denom[:, None, None]
        return out

    if mode == "global_minmax":
        gmin = float(out.min())
        gmax = float(out.max())
        denom = max(gmax - gmin, 1e-12)
        out = (out - gmin) / denom
        return out

    raise ValueError(f"Unknown norm mode: {mode}")


def smooth_maps(maps, sigma):
    if sigma <= 0:
        return maps
    out = np.empty_like(maps)
    for i in range(maps.shape[0]):
        out[i] = gaussian_filter(maps[i], sigma=sigma)
    return out


def topk_image_score(maps, topk):
    n, h, w = maps.shape
    flat = maps.reshape(n, h * w)
    k = max(1, min(int(topk), h * w))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def evaluate_from_maps(maps, masks, image_gt, topk):
    image_pred = topk_image_score(maps, topk)
    pixel_pred = maps.reshape(-1)
    pixel_gt = masks.reshape(-1).astype(np.int32)

    metrics = {
        "Image-AUROC": safe_auroc(image_gt, image_pred),
        "Pixel-AUROC": safe_auroc(pixel_gt, pixel_pred),
        "Image-AP": safe_ap(image_gt, image_pred),
        "Pixel-AP": safe_ap(pixel_gt, pixel_pred),
    }
    img_f1, img_thr = safe_best_f1(image_gt, image_pred)
    pix_f1, pix_thr = safe_best_f1(pixel_gt, pixel_pred)
    metrics["Image-F1"] = img_f1
    metrics["Pixel-F1"] = pix_f1
    metrics["Image-BestThreshold"] = img_thr
    metrics["Pixel-BestThreshold"] = pix_thr
    return metrics


def parse_list(raw, cast_fn):
    items = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        items.append(cast_fn(x))
    return items


def main():
    parser = argparse.ArgumentParser(description="Postprocess calibration sweep for teacher-student residual anomaly maps")
    parser.add_argument("--config", default="args/args_recon_teacher_student.json", help="Path to args json")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint suffix to load, e.g. best/last")
    parser.add_argument("--class-set", default="", help="Comma-separated class list to override config")
    parser.add_argument("--sigma-list", default="0,1,2", help="Gaussian sigma sweep, comma-separated")
    parser.add_argument("--norm-modes", default="none,per_image_minmax", help="Normalization sweep")
    parser.add_argument("--topk-list", default="50,100,200", help="Image-level top-k sweep")
    parser.add_argument("--reductions", default="residual_mean,residual_max", help="Reduction sweep")
    parser.add_argument("--save-tag", default="calib", help="Suffix tag for output csv files")
    cli_args = parser.parse_args()

    config_path = resolve_config_path(cli_args.config)
    with open(config_path, "r") as f:
        args = json.load(f)

    args["arg_num"] = args.get("arg_num", os.path.splitext(os.path.basename(config_path))[0])
    args = defaultdict_from_json(args)

    if cli_args.class_set.strip():
        current_classes = [x.strip() for x in cli_args.class_set.split(",") if x.strip()]
    else:
        selected_classes = args.get("selected_classes", [])
        if not isinstance(selected_classes, list) or len(selected_classes) == 0:
            raise ValueError("selected_classes must be a non-empty list in config when --class-set is not provided")
        current_classes = list(selected_classes)

    sigma_list = parse_list(cli_args.sigma_list, float)
    norm_modes = parse_list(cli_args.norm_modes, str)
    topk_list = parse_list(cli_args.topk_list, int)
    reductions = parse_list(cli_args.reductions, str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using config:", config_path)
    print("Using device:", device)
    print("Checkpoint:", cli_args.checkpoint)
    print("Classes:", current_classes)
    print("Sweep settings:")
    print("  reductions:", reductions)
    print("  sigma_list:", sigma_list)
    print("  norm_modes:", norm_modes)
    print("  topk_list:", topk_list)

    all_rows = []
    best_rows = []

    for sub_class in current_classes:
        print("=" * 72)
        print(f"Class: {sub_class}")

        testing_dataset = build_test_dataset(args, sub_class)
        num_workers_test = max(0, get_arg_int(args, "num_workers_test", 2))
        test_loader = DataLoader(
            testing_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers_test,
            persistent_workers=num_workers_test > 0,
        )

        model, ckpt_path = load_student_model(args, sub_class, cli_args.checkpoint, device)
        print("Checkpoint loaded:", ckpt_path)

        cached = collect_residual_maps(test_loader, args, model, device)
        print(
            f"Base infer throughput (no postprocess): "
            f"FPS={cached['base_fps']:.2f}, ms/img={cached['base_ms_per_img']:.2f}"
        )

        class_rows = []
        for reduction in reductions:
            base_maps = cached[reduction]
            for sigma in sigma_list:
                smoothed = smooth_maps(base_maps, sigma)
                for norm_mode in norm_modes:
                    processed = normalize_maps(smoothed, norm_mode)
                    for topk in topk_list:
                        metrics = evaluate_from_maps(processed, cached["mask"], cached["image_gt"], topk)
                        row = {
                            "classname": sub_class,
                            "reduction": reduction,
                            "sigma": sigma,
                            "norm_mode": norm_mode,
                            "topk": topk,
                            "Image-AUROC": metrics["Image-AUROC"],
                            "Pixel-AUROC": metrics["Pixel-AUROC"],
                            "Image-AP": metrics["Image-AP"],
                            "Pixel-AP": metrics["Pixel-AP"],
                            "Image-F1": metrics["Image-F1"],
                            "Pixel-F1": metrics["Pixel-F1"],
                            "Image-BestThreshold": metrics["Image-BestThreshold"],
                            "Pixel-BestThreshold": metrics["Pixel-BestThreshold"],
                            "Eval-FPS": cached["base_fps"],
                            "Eval-ms-per-img": cached["base_ms_per_img"],
                        }
                        row["PixelScore"] = (
                            (0.7 * row["Pixel-F1"] if np.isfinite(row["Pixel-F1"]) else 0.0)
                            + (0.3 * row["Pixel-AP"] if np.isfinite(row["Pixel-AP"]) else 0.0)
                        )
                        class_rows.append(row)

        class_df = pd.DataFrame(class_rows)
        class_df = class_df.sort_values(["PixelScore", "Pixel-F1", "Pixel-AP"], ascending=False)
        best = class_df.iloc[0].to_dict()
        best_rows.append(best)

        print("Best setting for class:")
        print(
            f"  reduction={best['reduction']}, sigma={best['sigma']}, "
            f"norm={best['norm_mode']}, topk={int(best['topk'])}"
        )
        print(
            f"  Pixel-F1={best['Pixel-F1']:.2f}, Pixel-AP={best['Pixel-AP']:.2f}, "
            f"Image-AUROC={best['Image-AUROC']:.2f}, Pixel-AUROC={best['Pixel-AUROC']:.2f}"
        )

        all_rows.extend(class_rows)

    grid_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows)

    metrics_root = f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}'
    os.makedirs(metrics_root, exist_ok=True)
    grid_csv = os.path.join(metrics_root, f"recon_postprocess_grid_{cli_args.save_tag}.csv")
    best_csv = os.path.join(metrics_root, f"recon_postprocess_best_{cli_args.save_tag}.csv")

    grid_df.to_csv(grid_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    print("=" * 72)
    print("Saved grid:", grid_csv)
    print("Saved best:", best_csv)

    baseline_csv = os.path.join(
        metrics_root,
        f'{args["eval_normal_t"]}_{args["eval_noisier_t"]}t_{args["condition_w"]}_MVTec_image_pixel_auroc_train_recon_teacher_student.csv',
    )

    if os.path.exists(baseline_csv):
        baseline_df = pd.read_csv(baseline_csv).drop_duplicates("classname", keep="last")
        merged = best_df.merge(baseline_df, on="classname", suffixes=("_bestpp", "_baseline"))
        for m in ["Image-AUROC", "Pixel-AUROC", "Image-AP", "Pixel-AP", "Image-F1", "Pixel-F1", "Eval-FPS"]:
            merged[f"delta_{m}"] = merged[f"{m}_bestpp"] - merged[f"{m}_baseline"]

        delta_csv = os.path.join(metrics_root, f"recon_postprocess_delta_vs_baseline_{cli_args.save_tag}.csv")
        merged.to_csv(delta_csv, index=False)

        print("Saved delta:", delta_csv)
        print("Mean delta vs baseline:")
        for m in ["Image-AUROC", "Pixel-AUROC", "Image-AP", "Pixel-AP", "Image-F1", "Pixel-F1", "Eval-FPS"]:
            print(f"  {m}: {merged[f'delta_{m}'].mean():+.2f}")
    else:
        print("Baseline summary not found, skip delta report:", baseline_csv)


if __name__ == "__main__":
    seed(42)
    main()
