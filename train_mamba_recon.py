from random import seed
import argparse
import csv
import os
import json
import shutil
import time
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from data.dataset_beta_thresh import (
    MVTecTrainDataset,
    MVTecTestDataset,
    VisATrainDataset,
    VisATestDataset,
    DAGMTrainDataset,
    DAGMTestDataset,
    MPDDTestDataset,
    MPDDTrainDataset,
)
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from models.Recon_subnetwork_mamba import UNetModelMamba, MAMBA_AVAILABLE
from models.Seg_subnetwork import SegmentationSubNetwork


SUMMARY_COLUMNS = [
    "classname",
    "Image-AUROC",
    "Pixel-AUROC",
    "Image-AP",
    "Pixel-AP",
    "Image-F1",
    "Pixel-F1",
    "Image-BestThreshold",
    "Pixel-BestThreshold",
    "Eval-FPS",
    "Eval-ms-per-img",
    "epoch",
    "recon_mamba_mode",
]

LEGACY_SUMMARY_COLUMNS = [
    "classname",
    "Image-AUROC",
    "Pixel-AUROC",
    "epoch",
    "recon_mamba_mode",
]


def resolve_config_path(config_arg: str) -> str:
    """Resolve config path with fallbacks for args/ directory usage."""
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
        # If user passes "args_mamba_low.json", also try "args/args_mamba_low.json".
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

    raise FileNotFoundError(
        f"Config file not found: {config_arg}. Tried: {deduped}"
    )


def defaultdict_from_json(json_dict):
    dd = defaultdict(str)
    dd.update(json_dict)
    return dd


def normalize_summary_csv(csv_path: str):
    """
    Normalize summary CSV to a single schema with headers.

    Supports:
    - legacy rows with 5 fields: classname,image_auroc,pixel_auroc,epoch,mode
    - new rows with 13 fields matching SUMMARY_COLUMNS order
    """
    if not os.path.exists(csv_path):
        return

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    header_present = rows[0] == SUMMARY_COLUMNS
    data_rows = rows[1:] if header_present else rows

    has_legacy_or_malformed = False
    normalized_rows = []
    for row in data_rows:
        if not row:
            continue
        if row == SUMMARY_COLUMNS:
            continue

        if len(row) == len(SUMMARY_COLUMNS):
            rec = dict(zip(SUMMARY_COLUMNS, row))
        elif len(row) == len(LEGACY_SUMMARY_COLUMNS):
            has_legacy_or_malformed = True
            rec = {k: np.nan for k in SUMMARY_COLUMNS}
            rec["classname"] = row[0]
            rec["Image-AUROC"] = row[1]
            rec["Pixel-AUROC"] = row[2]
            rec["epoch"] = row[3]
            rec["recon_mamba_mode"] = row[4]
        else:
            has_legacy_or_malformed = True
            padded = row[: len(SUMMARY_COLUMNS)] + [""] * (len(SUMMARY_COLUMNS) - len(row))
            rec = dict(zip(SUMMARY_COLUMNS, padded))

        normalized_rows.append(rec)

    if header_present and not has_legacy_or_malformed:
        return

    backup_path = f"{csv_path}.bak_{int(time.time())}"
    shutil.copy2(csv_path, backup_path)

    normalized_df = pd.DataFrame(normalized_rows, columns=SUMMARY_COLUMNS)
    normalized_df.to_csv(csv_path, index=False)
    print(f"[CSV MIGRATION] normalized summary file: {csv_path}")
    print(f"[CSV MIGRATION] backup saved to: {backup_path}")


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(f_loss) if self.reduce else f_loss


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
    f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    best_idx = int(np.nanargmax(f1_scores))
    best_f1 = round(float(f1_scores[best_idx]), 3) * 100

    if thresholds.size == 0:
        best_threshold = float("nan")
    else:
        threshold_idx = min(best_idx, thresholds.size - 1)
        best_threshold = round(float(thresholds[threshold_idx]), 6)

    return best_f1, best_threshold


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


def build_monitor_score(metrics, args):
    weighted_terms = [
        ("image_auroc", get_arg_float(args, "early_stop_w_image_auroc", 1.0)),
        ("pixel_auroc", get_arg_float(args, "early_stop_w_pixel_auroc", 1.0)),
        ("image_ap", get_arg_float(args, "early_stop_w_image_ap", 1.0)),
        ("pixel_ap", get_arg_float(args, "early_stop_w_pixel_ap", 1.0)),
        ("image_f1", get_arg_float(args, "early_stop_w_image_f1", 1.0)),
        ("pixel_f1", get_arg_float(args, "early_stop_w_pixel_f1", 1.0)),
    ]

    score = 0.0
    used_count = 0
    for metric_name, weight in weighted_terms:
        if weight == 0:
            continue
        value = metrics.get(metric_name, float("nan"))
        if np.isfinite(value):
            score += weight * float(value)
            used_count += 1

    if used_count == 0:
        return float("-inf")
    return score


def configure_runtime_environment(args):
    requested_tmp = args["mp_tmp_root"]
    if requested_tmp:
        tmp_root = requested_tmp
    elif os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK | os.X_OK):
        tmp_root = "/dev/shm/diffusionad_mp_tmp"
    else:
        tmp_root = os.path.join(os.getcwd(), ".diffusionad_mp_tmp")

    os.makedirs(tmp_root, exist_ok=True)
    os.environ["TMPDIR"] = tmp_root
    os.environ["TMP"] = tmp_root
    os.environ["TEMP"] = tmp_root
    tempfile.tempdir = tmp_root

    sharing_strategy = args["torch_sharing_strategy"] if args["torch_sharing_strategy"] else "file_system"
    try:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    except RuntimeError as e:
        print(f"[runtime] warning: failed to set torch sharing strategy to {sharing_strategy}: {e}")

    print(f"[runtime] tmpdir={tmp_root}, torch_sharing_strategy={sharing_strategy}")


def evaluate(testing_dataset_loader, args, unet_model, seg_model, sub_class, device):
    unet_model.eval()
    seg_model.eval()
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/', exist_ok=True)

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
    total_infer_time = 0.0
    total_images = 0

    with torch.no_grad():
        tbar = tqdm(testing_dataset_loader, desc=f"{sub_class} Eval", leave=False)
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
            ) = ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, image, normal_t_tensor, noiser_t_tensor, args)

            pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1))
            out_mask = pred_mask

            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_end = time.perf_counter()

            total_infer_time += infer_end - infer_start
            total_images += int(image.shape[0])
            if total_infer_time > 0:
                tbar.set_postfix(fps=f"{(total_images / total_infer_time):.2f}")

            topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
            topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
            image_score = torch.mean(topk_out_mask)

            total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
            total_image_gt = np.append(total_image_gt, target[0].detach().cpu().numpy())

            flatten_pred_mask = out_mask[0].flatten().detach().cpu().numpy()
            flatten_gt_mask = gt_mask[0].flatten().detach().cpu().numpy().astype(int)
            total_pixel_gt = np.append(total_pixel_gt, flatten_gt_mask)
            total_pixel_pred = np.append(total_pixel_pred, flatten_pred_mask)

    metrics = {
        "image_auroc": safe_auroc(total_image_gt, total_image_pred),
        "pixel_auroc": safe_auroc(total_pixel_gt, total_pixel_pred),
        "image_ap": safe_ap(total_image_gt, total_image_pred),
        "pixel_ap": safe_ap(total_pixel_gt, total_pixel_pred),
    }
    metrics["image_f1"], metrics["image_f1_threshold"] = safe_best_f1(total_image_gt, total_image_pred)
    metrics["pixel_f1"], metrics["pixel_f1_threshold"] = safe_best_f1(total_pixel_gt, total_pixel_pred)

    if total_infer_time > 0:
        metrics["fps"] = total_images / total_infer_time
    else:
        metrics["fps"] = float("nan")

    if total_images > 0:
        metrics["ms_per_img"] = (total_infer_time / total_images) * 1000.0
    else:
        metrics["ms_per_img"] = float("nan")

    return metrics


def save_models(unet_model, seg_model, args, final, epoch, sub_class):
    save_path = (
        f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/'
        f'params-{final}.pt'
    )
    torch.save(
        {
            "n_epoch": epoch,
            "unet_model_state_dict": unet_model.state_dict(),
            "seg_model_state_dict": seg_model.state_dict(),
            "args": args,
        },
        save_path,
    )


def train_one_class(training_loader, testing_loader, args, sub_class, class_type, device):
    in_channels = args["channels"]

    unet_model = UNetModelMamba(
        img_size=args["img_size"][0],
        base_channels=args["base_channels"],
        channel_mults=args["channel_mults"],
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        n_head_channels=args["num_head_channels"],
        in_channels=in_channels,
        mamba_mode=args["recon_mamba_mode"],
        mamba_resolutions=args["mamba_resolutions"],
        mamba_d_state=args["mamba_d_state"],
        mamba_d_conv=args["mamba_d_conv"],
        mamba_expand=args["mamba_expand"],
        mamba_dropout=args["mamba_dropout"],
        mamba_bidirectional=bool(args["mamba_bidirectional"]),
        mamba_medium_min_ds=args["mamba_medium_min_ds"],
    ).to(device)

    betas = get_beta_schedule(args["T"], args["beta_schedule"])
    ddpm_sample = GaussianDiffusionModel(
        args["img_size"],
        betas,
        loss_weight=args["loss_weight"],
        loss_type=args["loss-type"],
        noise=args["noise_fn"],
        img_channels=in_channels,
    )

    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)

    optimizer_ddpm = optim.Adam(
        unet_model.parameters(), lr=args["diffusion_lr"], weight_decay=args["weight_decay"]
    )
    optimizer_seg = optim.Adam(
        seg_model.parameters(), lr=args["seg_lr"], weight_decay=args["weight_decay"]
    )
    scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_seg, T_max=10, eta_min=0, last_epoch=-1
    )

    eval_interval = max(1, get_arg_int(args, "eval_interval", 5))
    early_stop_patience = max(0, get_arg_int(args, "early_stop_patience", 0))
    early_stop_min_delta = max(0.0, get_arg_float(args, "early_stop_min_delta", 0.0))
    early_stop_warmup_evals = max(0, get_arg_int(args, "early_stop_warmup_evals", 0))

    loss_focal = BinaryFocalLoss().to(device)
    loss_smL1 = nn.SmoothL1Loss().to(device)

    best_image_auroc = 0.0
    best_pixel_auroc = 0.0
    best_epoch = 0
    best_score = float("-inf")
    no_improve_evals = 0
    eval_count = 0
    early_stopped = False
    last_epoch_ran = 0
    best_eval_metrics = {
        "image_ap": float("nan"),
        "pixel_ap": float("nan"),
        "image_f1": float("nan"),
        "pixel_f1": float("nan"),
        "image_f1_threshold": float("nan"),
        "pixel_f1_threshold": float("nan"),
        "fps": float("nan"),
        "ms_per_img": float("nan"),
    }

    eval_history_csv = (
        f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/'
        f'{sub_class}_eval_history_mamba.csv'
    )

    if early_stop_patience > 0:
        print(
            f"[{sub_class}] Early stopping enabled: "
            f"patience={early_stop_patience} evals, min_delta={early_stop_min_delta}, "
            f"warmup_evals={early_stop_warmup_evals}, eval_interval={eval_interval}"
        )
        print(
            f"[{sub_class}] Monitor weights: "
            f"img_auc={get_arg_float(args, 'early_stop_w_image_auroc', 1.0)}, "
            f"px_auc={get_arg_float(args, 'early_stop_w_pixel_auroc', 1.0)}, "
            f"img_ap={get_arg_float(args, 'early_stop_w_image_ap', 1.0)}, "
            f"px_ap={get_arg_float(args, 'early_stop_w_pixel_ap', 1.0)}, "
            f"img_f1={get_arg_float(args, 'early_stop_w_image_f1', 1.0)}, "
            f"px_f1={get_arg_float(args, 'early_stop_w_pixel_f1', 1.0)}"
        )
    else:
        print(
            f"[{sub_class}] Early stopping disabled (early_stop_patience=0), "
            f"eval_interval={eval_interval}"
        )

    for epoch in range(0, args["EPOCHS"]):
        last_epoch_ran = epoch + 1
        unet_model.train()
        seg_model.train()
        train_loss = 0.0

        tbar = tqdm(training_loader, desc=f"{sub_class} Epoch {epoch}", leave=False)
        for sample in tbar:
            aug_image = sample["augmented_image"].to(device)
            anomaly_mask = sample["anomaly_mask"].to(device)
            anomaly_label = sample["has_anomaly"].to(device).squeeze()

            noise_loss, pred_x0, _normal_t, _x_normal_t, _x_noiser_t = ddpm_sample.norm_guided_one_step_denoising(
                unet_model, aug_image, anomaly_label, args
            )
            pred_mask = seg_model(torch.cat((aug_image, pred_x0), dim=1))

            focal_loss = loss_focal(pred_mask, anomaly_mask)
            smL1_loss = loss_smL1(pred_mask, anomaly_mask)
            loss = noise_loss + 5 * focal_loss + smL1_loss

            optimizer_ddpm.zero_grad()
            optimizer_seg.zero_grad()
            loss.backward()
            optimizer_ddpm.step()
            optimizer_seg.step()
            scheduler_seg.step()

            train_loss += loss.item()
            tbar.set_postfix(loss=f"{train_loss:.3f}")

        if (epoch + 1) % eval_interval == 0:
            eval_metrics = evaluate(
                testing_loader,
                args,
                unet_model,
                seg_model,
                sub_class,
                device,
            )
            eval_count += 1
            temp_image_auroc = eval_metrics["image_auroc"]
            temp_pixel_auroc = eval_metrics["pixel_auroc"]
            monitor_score = build_monitor_score(eval_metrics, args)
            monitor_score_text = f"{monitor_score:.2f}" if np.isfinite(monitor_score) else "nan"

            print(
                f"[{sub_class}] Epoch {epoch + 1}/{args['EPOCHS']} | "
                f"Image-AUROC: {temp_image_auroc:.2f} | Pixel-AUROC: {temp_pixel_auroc:.2f} | "
                f"Image-AP: {eval_metrics['image_ap']:.2f} | Pixel-AP: {eval_metrics['pixel_ap']:.2f} | "
                f"Image-BestF1: {eval_metrics['image_f1']:.2f} | Pixel-BestF1: {eval_metrics['pixel_f1']:.2f} | "
                f"Monitor-Score: {monitor_score_text} | "
                f"FPS: {eval_metrics['fps']:.2f} | ms/img: {eval_metrics['ms_per_img']:.2f}"
            )

            eval_row = {
                "classname": sub_class,
                "class_type": class_type,
                "epoch": epoch + 1,
                "Image-AUROC": temp_image_auroc,
                "Pixel-AUROC": temp_pixel_auroc,
                "Image-AP": eval_metrics["image_ap"],
                "Pixel-AP": eval_metrics["pixel_ap"],
                "Image-F1": eval_metrics["image_f1"],
                "Pixel-F1": eval_metrics["pixel_f1"],
                "Image-BestThreshold": eval_metrics["image_f1_threshold"],
                "Pixel-BestThreshold": eval_metrics["pixel_f1_threshold"],
                "Eval-FPS": eval_metrics["fps"],
                "Eval-ms-per-img": eval_metrics["ms_per_img"],
                "recon_mamba_mode": args["recon_mamba_mode"],
            }
            eval_history_df = pd.DataFrame([eval_row])
            eval_history_df.to_csv(
                eval_history_csv,
                mode="a",
                header=not os.path.exists(eval_history_csv),
                index=False,
            )

            current_score = monitor_score
            is_improved = np.isfinite(current_score) and (current_score > best_score + early_stop_min_delta)

            if is_improved:
                save_models(unet_model, seg_model, args=args, final="best", epoch=epoch + 1, sub_class=sub_class)
                best_image_auroc = temp_image_auroc
                best_pixel_auroc = temp_pixel_auroc
                best_epoch = epoch + 1
                best_score = current_score
                no_improve_evals = 0
                best_eval_metrics = {
                    "image_ap": eval_metrics["image_ap"],
                    "pixel_ap": eval_metrics["pixel_ap"],
                    "image_f1": eval_metrics["image_f1"],
                    "pixel_f1": eval_metrics["pixel_f1"],
                    "image_f1_threshold": eval_metrics["image_f1_threshold"],
                    "pixel_f1_threshold": eval_metrics["pixel_f1_threshold"],
                    "fps": eval_metrics["fps"],
                    "ms_per_img": eval_metrics["ms_per_img"],
                }
            else:
                no_improve_evals += 1

            if (
                early_stop_patience > 0
                and eval_count > early_stop_warmup_evals
                and no_improve_evals >= early_stop_patience
            ):
                print(
                    f"[{sub_class}] Early stopping triggered at epoch {epoch + 1}: "
                    f"no improvement for {no_improve_evals} evals "
                    f"(patience={early_stop_patience}, min_delta={early_stop_min_delta})."
                )
                early_stopped = True
                break

    save_models(unet_model, seg_model, args=args, final="last", epoch=last_epoch_ran, sub_class=sub_class)

    if early_stopped:
        print(f"[{sub_class}] Training stopped early at epoch {last_epoch_ran}/{args['EPOCHS']}.")

    temp = {
        "classname": [sub_class],
        "Image-AUROC": [best_image_auroc],
        "Pixel-AUROC": [best_pixel_auroc],
        "Image-AP": [best_eval_metrics["image_ap"]],
        "Pixel-AP": [best_eval_metrics["pixel_ap"]],
        "Image-F1": [best_eval_metrics["image_f1"]],
        "Pixel-F1": [best_eval_metrics["pixel_f1"]],
        "Image-BestThreshold": [best_eval_metrics["image_f1_threshold"]],
        "Pixel-BestThreshold": [best_eval_metrics["pixel_f1_threshold"]],
        "Eval-FPS": [best_eval_metrics["fps"]],
        "Eval-ms-per-img": [best_eval_metrics["ms_per_img"]],
        "epoch": [best_epoch],
        "recon_mamba_mode": [args["recon_mamba_mode"]],
    }
    df_class = pd.DataFrame(temp)
    out_csv = (
        f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/'
        f'{args["eval_normal_t"]}_{args["eval_noisier_t"]}t_{args["condition_w"]}_'
        f'{class_type}_image_pixel_auroc_train_mamba.csv'
    )
    normalize_summary_csv(out_csv)
    df_class.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)


def build_datasets(args, sub_class):
    mvtec_classes = [
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
    visa_classes = [
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
    mpdd_classes = [
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
    ]
    dagm_classes = [f"Class{i}" for i in range(1, 11)]

    if sub_class in visa_classes:
        subclass_path = os.path.join(args["visa_root_path"], sub_class)
        training_dataset = VisATrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
        testing_dataset = VisATestDataset(subclass_path, sub_class, img_size=args["img_size"])
        class_type = "VisA"
    elif sub_class in mpdd_classes:
        subclass_path = os.path.join(args["mpdd_root_path"], sub_class)
        training_dataset = MPDDTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
        testing_dataset = MPDDTestDataset(subclass_path, sub_class, img_size=args["img_size"])
        class_type = "MPDD"
    elif sub_class in mvtec_classes:
        subclass_path = os.path.join(args["mvtec_root_path"], sub_class)
        training_dataset = MVTecTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
        testing_dataset = MVTecTestDataset(subclass_path, sub_class, img_size=args["img_size"])
        class_type = "MVTec"
    elif sub_class in dagm_classes:
        subclass_path = os.path.join(args["dagm_root_path"], sub_class)
        training_dataset = DAGMTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
        testing_dataset = DAGMTestDataset(subclass_path, sub_class, img_size=args["img_size"])
        class_type = "DAGM"
    else:
        raise ValueError(f"Unknown class name: {sub_class}")

    return training_dataset, testing_dataset, class_type


def main():
    parser = argparse.ArgumentParser(description="Independent training script for Mamba-in-Recon experiments")
    parser.add_argument("--config", default="args/args_mamba_low.json", help="Path to independent args json")
    parser.add_argument("--class-set", default="", help="Comma separated class list. Overrides json key selected_classes")
    cli_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = resolve_config_path(cli_args.config)
    with open(config_path, "r") as f:
        args = json.load(f)

    args["arg_num"] = args.get("arg_num", os.path.splitext(os.path.basename(config_path))[0])
    args = defaultdict_from_json(args)
    configure_runtime_environment(args)

    if args["recon_mamba_mode"] != "none":
        if not MAMBA_AVAILABLE:
            raise RuntimeError(
                "Mamba mode is enabled but mamba_ssm is not available. Install mamba-ssm and causal-conv1d first."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Mamba mode requires CUDA in this environment, but CUDA is not available."
            )

    if cli_args.class_set.strip():
        current_classes = [x.strip() for x in cli_args.class_set.split(",") if x.strip()]
    else:
        selected_classes = args.get("selected_classes", [])
        if not isinstance(selected_classes, list) or len(selected_classes) == 0:
            raise ValueError(
                "selected_classes must be a non-empty list in the config when --class-set is not provided."
            )
        current_classes = list(selected_classes)

    print("Using config:", config_path)
    print("Selected classes:", current_classes)
    print("Mamba mode:", args["recon_mamba_mode"])

    for sub_class in current_classes:
        print("Training class:", sub_class)
        training_dataset, testing_dataset, class_type = build_datasets(args, sub_class)

        train_loader = DataLoader(
            training_dataset,
            batch_size=args["Batch_Size"],
            shuffle=True,
            num_workers=args["num_workers_train"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=args["num_workers_train"] > 0,
        )
        test_loader = DataLoader(
            testing_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args["num_workers_test"],
            persistent_workers=args["num_workers_test"] > 0,
        )

        for folder in [
            f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}',
            f'{args["output_path"]}/diffusion-training-images/ARGS={args["arg_num"]}/{sub_class}',
            f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}',
        ]:
            os.makedirs(folder, exist_ok=True)

        train_one_class(train_loader, test_loader, args, sub_class, class_type, device)


if __name__ == "__main__":
    seed(42)
    main()
