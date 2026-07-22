from random import seed
import argparse
import os
import json
import time
import tempfile
from collections import defaultdict


def sanitize_thread_env():
    # libgomp requires a positive integer thread count; fix invalid values (e.g., 0).
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        raw_value = os.environ.get(env_name, "").strip()
        try:
            parsed = int(raw_value) if raw_value else None
        except ValueError:
            parsed = None
        if parsed is None or parsed <= 0:
            os.environ[env_name] = "1"


sanitize_thread_env()

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
from models.DDPM import GaussianDiffusionModel, get_beta_schedule, extract, mean_flat
from models.Recon_subnetwork import UNetModel


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


def get_arg_float(args, key, default):
    value = args[key]
    if value == "" or value is None:
        return default
    return float(value)


def get_arg_bool(args, key, default):
    value = args[key]
    if value == "" or value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def setup_distributed(args):
    ddp_enable = get_arg_bool(args, "ddp_enable", True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    distributed = ddp_enable and world_size > 1
    if not distributed:
        return False, 0, 0, 1

    backend = args["ddp_backend"] if args["ddp_backend"] else ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()
    return True, rank, local_rank, world_size


def configure_runtime_environment(args, rank=0):
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

    if is_main_process(rank):
        print(f"[runtime] tmpdir={tmp_root}, torch_sharing_strategy={sharing_strategy}")


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


def build_datasets(args, sub_class):
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


def build_residual_anomaly_map(image, recon, args):
    mode = args["test_anomaly_map_mode"] if args["test_anomaly_map_mode"] else "residual_mean"
    residual = torch.abs(image - recon)
    if mode == "residual_mean":
        return torch.mean(residual, dim=1, keepdim=True)
    if mode == "residual_max":
        return torch.max(residual, dim=1, keepdim=True)[0]
    raise ValueError(f"Unknown test_anomaly_map_mode: {mode}")


def diffusion_loss_vector(loss_type, estimate_noise, noise):
    if loss_type == "l1":
        return mean_flat((estimate_noise - noise).abs())
    return mean_flat((estimate_noise - noise).square())


def sample_shared_diffusion_conditions(ddpm_sample, x_0, args):
    normal_t = torch.randint(0, args["less_t_range"], (x_0.shape[0],), device=x_0.device)
    noisier_t = torch.randint(args["less_t_range"], ddpm_sample.num_timesteps, (x_0.shape[0],), device=x_0.device)
    noise_normal = ddpm_sample.noise_fn(x_0, normal_t).float()
    noise_noisier = ddpm_sample.noise_fn(x_0, noisier_t).float()
    return normal_t, noisier_t, noise_normal, noise_noisier


def norm_guided_forward_shared(
    ddpm_sample,
    model,
    x_0,
    anomaly_label,
    args,
    normal_t,
    noisier_t,
    noise_normal,
    noise_noisier,
):
    x_normal_t = ddpm_sample.sample_q(x_0, normal_t, noise_normal)
    x_noisier_t = ddpm_sample.sample_q(x_0, noisier_t, noise_noisier)

    estimate_noise_normal = model(x_normal_t, normal_t)
    estimate_noise_noisier = model(x_noisier_t, noisier_t)

    normal_loss_vec = diffusion_loss_vector(ddpm_sample.loss_type, estimate_noise_normal, noise_normal)
    noisier_loss_vec = diffusion_loss_vector(ddpm_sample.loss_type, estimate_noise_noisier, noise_noisier)

    valid_mask = anomaly_label == 0
    if torch.any(valid_mask):
        noise_loss = (normal_loss_vec + noisier_loss_vec)[valid_mask].mean()
    else:
        noise_loss = torch.zeros((), dtype=x_0.dtype, device=x_0.device)

    pred_x_0_noisier = ddpm_sample.predict_x_0_from_eps(x_noisier_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
    pred_x_t_noisier = ddpm_sample.sample_q(pred_x_0_noisier, normal_t, estimate_noise_normal)

    estimate_noise_hat = estimate_noise_normal - extract(
        ddpm_sample.sqrt_one_minus_alphas_cumprod,
        normal_t,
        x_normal_t.shape,
        x_0.device,
    ) * args["condition_w"] * (pred_x_t_noisier - x_normal_t)
    pred_x_0_norm_guided = ddpm_sample.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)

    return noise_loss, pred_x_0_norm_guided


def evaluate_student(testing_loader, args, student_model, sub_class, device):
    student_model.eval()
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
        tbar = tqdm(testing_loader, desc=f"{sub_class} Eval", leave=False)
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
                student_model,
                image,
                normal_t_tensor,
                noiser_t_tensor,
                args,
            )

            out_mask = build_residual_anomaly_map(image, pred_x_0_condition, args)

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

    metrics["fps"] = total_images / total_infer_time if total_infer_time > 0 else float("nan")
    metrics["ms_per_img"] = (total_infer_time / total_images) * 1000.0 if total_images > 0 else float("nan")
    return metrics


def save_checkpoint(teacher_model, student_model, args, final, epoch, sub_class, stage):
    save_path = f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt'
    torch.save(
        {
            "n_epoch": epoch,
            "stage": stage,
            "teacher_model_state_dict": unwrap_model(teacher_model).state_dict(),
            "student_model_state_dict": unwrap_model(student_model).state_dict(),
            "args": args,
        },
        save_path,
    )


def train_teacher_stage(training_loader, args, teacher_model, ddpm_sample, optimizer_teacher, device, rank=0, distributed=False):
    teacher_pretrain_epochs = max(1, get_arg_int(args, "teacher_pretrain_epochs", 200))
    teacher_recon_l1_w = get_arg_float(args, "teacher_recon_l1_w", 0.0)
    recon_loss = nn.SmoothL1Loss().to(device)

    if is_main_process(rank):
        print(
            f"[teacher-stage] epochs={teacher_pretrain_epochs}, "
            f"teacher_recon_l1_w={teacher_recon_l1_w}"
        )

    for epoch in range(teacher_pretrain_epochs):
        if distributed and isinstance(training_loader.sampler, DistributedSampler):
            training_loader.sampler.set_epoch(epoch)

        teacher_model.train()
        epoch_loss = 0.0
        tbar = tqdm(
            training_loader,
            desc=f"Teacher Epoch {epoch + 1}/{teacher_pretrain_epochs}",
            leave=False,
            disable=not is_main_process(rank),
        )

        for sample in tbar:
            aug_image = sample["augmented_image"].to(device)
            anomaly_label = sample["has_anomaly"].to(device).view(-1)

            normal_t, noisier_t, noise_normal, noise_noisier = sample_shared_diffusion_conditions(ddpm_sample, aug_image, args)
            teacher_noise_loss, teacher_pred = norm_guided_forward_shared(
                ddpm_sample,
                teacher_model,
                aug_image,
                anomaly_label,
                args,
                normal_t,
                noisier_t,
                noise_normal,
                noise_noisier,
            )

            loss = teacher_noise_loss + teacher_recon_l1_w * recon_loss(teacher_pred, aug_image)

            optimizer_teacher.zero_grad()
            loss.backward()
            optimizer_teacher.step()

            epoch_loss += loss.item()
            if is_main_process(rank):
                tbar.set_postfix(loss=f"{epoch_loss:.3f}")


def train_student_stage(
    training_loader,
    testing_loader,
    args,
    teacher_model,
    student_model,
    ddpm_sample,
    optimizer_student,
    sub_class,
    class_type,
    device,
    rank=0,
    distributed=False,
):
    student_epochs = max(1, get_arg_int(args, "student_epochs", args["EPOCHS"]))
    eval_interval = max(1, get_arg_int(args, "eval_interval", 50))
    early_stop_patience = max(0, get_arg_int(args, "early_stop_patience", 0))
    early_stop_min_delta = max(0.0, get_arg_float(args, "early_stop_min_delta", 0.0))
    early_stop_warmup_evals = max(0, get_arg_int(args, "early_stop_warmup_evals", 0))

    student_noise_w = get_arg_float(args, "student_noise_w", 1.0)
    kd_recon_w = get_arg_float(args, "kd_recon_w", 1.0)
    kd_res_w = get_arg_float(args, "kd_res_w", 1.0)
    student_recon_w = get_arg_float(args, "student_recon_w", 0.2)

    smooth_l1 = nn.SmoothL1Loss().to(device)

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

    eval_history_csv = f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}_eval_history_recon_kd.csv'

    if is_main_process(rank):
        print(
            f"[student-stage] epochs={student_epochs}, eval_interval={eval_interval}, "
            f"weights(noise/kd_recon/kd_res/student_recon)="
            f"{student_noise_w}/{kd_recon_w}/{kd_res_w}/{student_recon_w}"
        )

    for p in teacher_model.parameters():
        p.requires_grad_(False)

    for epoch in range(student_epochs):
        last_epoch_ran = epoch + 1
        if distributed and isinstance(training_loader.sampler, DistributedSampler):
            training_loader.sampler.set_epoch(get_arg_int(args, "teacher_pretrain_epochs", 200) + epoch)

        teacher_model.eval()
        student_model.train()
        train_loss = 0.0

        tbar = tqdm(
            training_loader,
            desc=f"{sub_class} Student Epoch {epoch + 1}/{student_epochs}",
            leave=False,
            disable=not is_main_process(rank),
        )
        for sample in tbar:
            aug_image = sample["augmented_image"].to(device)
            anomaly_label = sample["has_anomaly"].to(device).view(-1)

            normal_t, noisier_t, noise_normal, noise_noisier = sample_shared_diffusion_conditions(ddpm_sample, aug_image, args)

            with torch.no_grad():
                _teacher_noise_loss, teacher_pred = norm_guided_forward_shared(
                    ddpm_sample,
                    teacher_model,
                    aug_image,
                    anomaly_label,
                    args,
                    normal_t,
                    noisier_t,
                    noise_normal,
                    noise_noisier,
                )

            student_noise_loss, student_pred = norm_guided_forward_shared(
                ddpm_sample,
                student_model,
                aug_image,
                anomaly_label,
                args,
                normal_t,
                noisier_t,
                noise_normal,
                noise_noisier,
            )

            kd_recon_loss = smooth_l1(student_pred, teacher_pred)
            kd_res_loss = smooth_l1(torch.abs(aug_image - student_pred), torch.abs(aug_image - teacher_pred))
            student_recon_loss = smooth_l1(student_pred, aug_image)

            loss = (
                student_noise_w * student_noise_loss
                + kd_recon_w * kd_recon_loss
                + kd_res_w * kd_res_loss
                + student_recon_w * student_recon_loss
            )

            optimizer_student.zero_grad()
            loss.backward()
            optimizer_student.step()

            train_loss += loss.item()
            if is_main_process(rank):
                tbar.set_postfix(loss=f"{train_loss:.3f}")

        if (epoch + 1) % eval_interval == 0 and epoch > 0:
            should_stop = False

            if is_main_process(rank):
                if testing_loader is None:
                    raise RuntimeError("Rank-0 requires a valid testing_loader for evaluation")

                eval_metrics = evaluate_student(
                    testing_loader,
                    args,
                    unwrap_model(student_model),
                    sub_class,
                    device,
                )
                eval_count += 1

                temp_image_auroc = eval_metrics["image_auroc"]
                temp_pixel_auroc = eval_metrics["pixel_auroc"]
                monitor_score = build_monitor_score(eval_metrics, args)
                monitor_score_text = f"{monitor_score:.2f}" if np.isfinite(monitor_score) else "nan"

                print(
                    f"[{sub_class}] Epoch {epoch + 1}/{student_epochs} | "
                    f"Image-AUROC: {temp_image_auroc:.2f} | Pixel-AUROC: {temp_pixel_auroc:.2f} | "
                    f"Image-AP: {eval_metrics['image_ap']:.2f} | Pixel-AP: {eval_metrics['pixel_ap']:.2f} | "
                    f"Image-BestF1: {eval_metrics['image_f1']:.2f} | Pixel-BestF1: {eval_metrics['pixel_f1']:.2f} | "
                    f"Monitor-Score: {monitor_score_text} | FPS: {eval_metrics['fps']:.2f} | "
                    f"ms/img: {eval_metrics['ms_per_img']:.2f}"
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
                    "test_anomaly_map_mode": args["test_anomaly_map_mode"],
                }
                pd.DataFrame([eval_row]).to_csv(
                    eval_history_csv,
                    mode="a",
                    header=not os.path.exists(eval_history_csv),
                    index=False,
                )

                current_score = monitor_score
                is_improved = np.isfinite(current_score) and (current_score > best_score + early_stop_min_delta)

                if is_improved:
                    save_checkpoint(
                        teacher_model,
                        student_model,
                        args=args,
                        final="best",
                        epoch=epoch + 1,
                        sub_class=sub_class,
                        stage="student",
                    )
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
                    should_stop = True

            if distributed:
                stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                early_stopped = True
                break

    if is_main_process(rank):
        save_checkpoint(
            teacher_model,
            student_model,
            args=args,
            final="last",
            epoch=last_epoch_ran,
            sub_class=sub_class,
            stage="student",
        )

        if early_stopped:
            print(f"[{sub_class}] Training stopped early at epoch {last_epoch_ran}/{student_epochs}.")

        summary_row = {
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
            "test_anomaly_map_mode": [args["test_anomaly_map_mode"]],
        }
        out_csv = (
            f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/'
            f'{args["eval_normal_t"]}_{args["eval_noisier_t"]}t_{args["condition_w"]}_'
            f'{class_type}_image_pixel_auroc_train_recon_teacher_student.csv'
        )
        pd.DataFrame(summary_row).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)


def train_one_class(
    training_loader,
    testing_loader,
    args,
    sub_class,
    class_type,
    device,
    rank=0,
    world_size=1,
    distributed=False,
):
    in_channels = args["channels"]

    teacher_base_channels = get_arg_int(args, "teacher_base_channels", get_arg_int(args, "base_channels", 128))
    student_base_channels = get_arg_int(args, "student_base_channels", max(32, teacher_base_channels // 2))

    teacher_channel_mults = parse_channel_mults(args["teacher_channel_mults"], args["channel_mults"])
    student_channel_mults = parse_channel_mults(args["student_channel_mults"], args["channel_mults"])

    teacher_model = UNetModel(
        args["img_size"][0],
        teacher_base_channels,
        channel_mults=teacher_channel_mults,
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        n_head_channels=args["num_head_channels"],
        in_channels=in_channels,
    ).to(device)

    student_model = UNetModel(
        args["img_size"][0],
        student_base_channels,
        channel_mults=student_channel_mults,
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        n_head_channels=args["num_head_channels"],
        in_channels=in_channels,
    ).to(device)

    if distributed:
        ddp_find_unused = get_arg_bool(args, "ddp_find_unused_parameters", False)
        if device.type == "cuda":
            ddp_kwargs = {
                "device_ids": [device.index],
                "output_device": device.index,
                "find_unused_parameters": ddp_find_unused,
            }
        else:
            ddp_kwargs = {"find_unused_parameters": ddp_find_unused}
        teacher_model = DDP(teacher_model, **ddp_kwargs)
        student_model = DDP(student_model, **ddp_kwargs)

    betas = get_beta_schedule(args["T"], args["beta_schedule"])
    ddpm_sample = GaussianDiffusionModel(
        args["img_size"],
        betas,
        loss_weight=args["loss_weight"],
        loss_type=args["loss-type"],
        noise=args["noise_fn"],
        img_channels=in_channels,
    )

    lr_scale = 1.0
    if get_arg_bool(args, "lr_scale_with_world_size", True):
        base_ws = max(1, get_arg_int(args, "lr_scale_base_world_size", 1))
        lr_scale = float(world_size) / float(base_ws)

    teacher_lr = get_arg_float(args, "teacher_lr", get_arg_float(args, "diffusion_lr", 1e-4)) * lr_scale
    student_lr = get_arg_float(args, "student_lr", get_arg_float(args, "diffusion_lr", 1e-4)) * lr_scale

    if is_main_process(rank):
        global_batch_size = int(args["Batch_Size"]) * int(world_size)
        print(
            f"[{sub_class}] distributed={distributed}, world_size={world_size}, "
            f"batch_per_gpu={args['Batch_Size']}, global_batch={global_batch_size}, lr_scale={lr_scale:.2f}"
        )
        print(
            f"[{sub_class}] teacher(base={teacher_base_channels}, mults={teacher_channel_mults}) | "
            f"student(base={student_base_channels}, mults={student_channel_mults})"
        )
        print(f"[{sub_class}] lr: teacher={teacher_lr:.2e}, student={student_lr:.2e}")

    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=teacher_lr, weight_decay=args["weight_decay"])
    optimizer_student = optim.Adam(student_model.parameters(), lr=student_lr, weight_decay=args["weight_decay"])

    train_teacher_stage(
        training_loader,
        args,
        teacher_model,
        ddpm_sample,
        optimizer_teacher,
        device,
        rank=rank,
        distributed=distributed,
    )

    if is_main_process(rank):
        save_checkpoint(
            teacher_model,
            student_model,
            args=args,
            final="teacher-last",
            epoch=get_arg_int(args, "teacher_pretrain_epochs", 200),
            sub_class=sub_class,
            stage="teacher",
        )

    if distributed:
        dist.barrier()

    train_student_stage(
        training_loader,
        testing_loader,
        args,
        teacher_model,
        student_model,
        ddpm_sample,
        optimizer_student,
        sub_class,
        class_type,
        device,
        rank=rank,
        distributed=distributed,
    )


def main():
    parser = argparse.ArgumentParser(description="Teacher-first student distillation for reconstruction-only anomaly detection")
    parser.add_argument("--config", default="args/args_recon_teacher_student.json", help="Path to args json")
    parser.add_argument("--class-set", default="", help="Comma-separated class list to override config")
    cli_args = parser.parse_args()

    config_path = resolve_config_path(cli_args.config)
    with open(config_path, "r") as f:
        args = json.load(f)

    args["arg_num"] = args.get("arg_num", os.path.splitext(os.path.basename(config_path))[0])
    args = defaultdict_from_json(args)

    distributed, rank, local_rank, world_size = setup_distributed(args)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    configure_runtime_environment(args, rank=rank)

    if cli_args.class_set.strip():
        current_classes = [x.strip() for x in cli_args.class_set.split(",") if x.strip()]
    else:
        selected_classes = args.get("selected_classes", [])
        if not isinstance(selected_classes, list) or len(selected_classes) == 0:
            raise ValueError("selected_classes must be a non-empty list in config when --class-set is not provided")
        current_classes = list(selected_classes)

    if is_main_process(rank):
        print("Using config:", config_path)
        print("Selected classes:", current_classes)
        print("Train mode: teacher pretrain -> student distill")
        print("Test mode: residual map only, mode:", args["test_anomaly_map_mode"])
        print(f"DDP enabled: {distributed}, world_size={world_size}, rank={rank}, local_rank={local_rank}")

    if distributed:
        dist.barrier()

    for sub_class in current_classes:
        if is_main_process(rank):
            print("Training class:", sub_class)

        training_dataset, testing_dataset, class_type = build_datasets(args, sub_class)

        num_workers_train = max(0, get_arg_int(args, "num_workers_train", 4))
        num_workers_test = max(0, get_arg_int(args, "num_workers_test", 2))

        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(training_dataset, num_replicas=world_size, rank=rank, shuffle=True)

        train_loader = DataLoader(
            training_dataset,
            batch_size=args["Batch_Size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers_train,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers_train > 0,
        )

        if is_main_process(rank):
            test_loader = DataLoader(
                testing_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers_test,
                persistent_workers=num_workers_test > 0,
            )
        else:
            test_loader = None

        if is_main_process(rank):
            for folder in [
                f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}',
                f'{args["output_path"]}/diffusion-training-images/ARGS={args["arg_num"]}/{sub_class}',
                f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}',
            ]:
                os.makedirs(folder, exist_ok=True)

        if distributed:
            dist.barrier()

        train_one_class(
            train_loader,
            test_loader,
            args,
            sub_class,
            class_type,
            device,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )

        if distributed:
            dist.barrier()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    seed(42)
    main()
