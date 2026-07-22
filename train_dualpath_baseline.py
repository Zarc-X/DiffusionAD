from random import seed
import argparse
import os
import json
import time
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from models.Recon_subnetwork import UNetModel
from models.Seg_subnetwork import SegmentationSubNetwork
from models.DualPath_head import MainBranchAnomalyHead, residual_anomaly_map


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


def build_anomaly_map(image, recon, main_head, args):
    mode = args["test_anomaly_map_mode"] if args["test_anomaly_map_mode"] else "main_head"
    head_input = torch.cat((image, recon), dim=1)
    if mode == "main_head":
        return main_head(head_input)
    if mode == "residual_mean":
        return residual_anomaly_map(image, recon, reduce_mode="mean")
    if mode == "residual_max":
        return residual_anomaly_map(image, recon, reduce_mode="max")
    raise ValueError(f"Unknown test_anomaly_map_mode: {mode}")


def evaluate(testing_loader, args, unet_model, main_head, sub_class, device):
    unet_model.eval()
    main_head.eval()
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
            ) = ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, image, normal_t_tensor, noiser_t_tensor, args)

            out_mask = build_anomaly_map(image, pred_x_0_condition, main_head, args)

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


def save_models(unet_model, seg_model, main_head, args, final, epoch, sub_class):
    save_path = f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt'
    torch.save(
        {
            "n_epoch": epoch,
            "unet_model_state_dict": unwrap_model(unet_model).state_dict(),
            "seg_model_state_dict": unwrap_model(seg_model).state_dict(),
            "main_head_state_dict": unwrap_model(main_head).state_dict(),
            "args": args,
        },
        save_path,
    )


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

    unet_model = UNetModel(
        args["img_size"][0],
        args["base_channels"],
        channel_mults=args["channel_mults"],
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        n_head_channels=args["num_head_channels"],
        in_channels=in_channels,
    ).to(device)

    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)
    main_head = MainBranchAnomalyHead(in_channels=6, hidden_channels=args["main_head_channels"], out_channels=1).to(device)

    if distributed and get_arg_bool(args, "ddp_sync_batchnorm", False):
        seg_model = nn.SyncBatchNorm.convert_sync_batchnorm(seg_model)
        main_head = nn.SyncBatchNorm.convert_sync_batchnorm(main_head)

    if distributed:
        ddp_find_unused = get_arg_bool(args, "ddp_find_unused_parameters", False)
        if device.type == "cuda":
            ddp_kwargs = {
                "device_ids": [device.index],
                "output_device": device.index,
                "find_unused_parameters": ddp_find_unused,
            }
        else:
            ddp_kwargs = {
                "find_unused_parameters": ddp_find_unused,
            }
        unet_model = DDP(unet_model, **ddp_kwargs)
        seg_model = DDP(seg_model, **ddp_kwargs)
        main_head = DDP(main_head, **ddp_kwargs)

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

    diffusion_lr = get_arg_float(args, "diffusion_lr", 1e-4) * lr_scale
    seg_lr = get_arg_float(args, "seg_lr", 1e-5) * lr_scale
    main_head_lr = get_arg_float(args, "main_head_lr", 1e-5) * lr_scale

    if is_main_process(rank):
        global_batch_size = int(args["Batch_Size"]) * int(world_size)
        print(
            f"[{sub_class}] distributed={distributed}, world_size={world_size}, "
            f"batch_per_gpu={args['Batch_Size']}, global_batch={global_batch_size}, lr_scale={lr_scale:.2f}"
        )
        print(
            f"[{sub_class}] lr: diffusion={diffusion_lr:.2e}, seg={seg_lr:.2e}, main_head={main_head_lr:.2e}"
        )

    optimizer_ddpm = optim.Adam(unet_model.parameters(), lr=diffusion_lr, weight_decay=args["weight_decay"])
    optimizer_seg = optim.Adam(seg_model.parameters(), lr=seg_lr, weight_decay=args["weight_decay"])
    optimizer_head = optim.Adam(main_head.parameters(), lr=main_head_lr, weight_decay=args["weight_decay"])

    try:
        scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=10, eta_min=0, last_epoch=-1, verbose=False)
    except TypeError:
        scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=10, eta_min=0, last_epoch=-1)

    loss_focal = BinaryFocalLoss().to(device)
    loss_smL1 = nn.SmoothL1Loss().to(device)
    loss_kd = nn.MSELoss().to(device)

    eval_interval = max(1, get_arg_int(args, "eval_interval", 50))
    early_stop_patience = max(0, get_arg_int(args, "early_stop_patience", 0))
    early_stop_min_delta = max(0.0, get_arg_float(args, "early_stop_min_delta", 0.0))
    early_stop_warmup_evals = max(0, get_arg_int(args, "early_stop_warmup_evals", 0))

    lambda_aux = get_arg_float(args, "lambda_aux", 1.0)
    lambda_main = get_arg_float(args, "lambda_main", 1.0)
    lambda_kd = get_arg_float(args, "lambda_kd", 0.2)

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

    eval_history_csv = f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}_eval_history_dualpath.csv'

    for epoch in range(0, args["EPOCHS"]):
        last_epoch_ran = epoch + 1
        if distributed and isinstance(training_loader.sampler, DistributedSampler):
            training_loader.sampler.set_epoch(epoch)

        unet_model.train()
        seg_model.train()
        main_head.train()
        train_loss = 0.0

        tbar = tqdm(
            training_loader,
            desc=f"{sub_class} Epoch {epoch}",
            leave=False,
            disable=not is_main_process(rank),
        )
        for sample in tbar:
            aug_image = sample["augmented_image"].to(device)
            anomaly_mask = sample["anomaly_mask"].to(device)
            anomaly_label = sample["has_anomaly"].to(device).squeeze()

            noise_loss, pred_x0, _normal_t, _x_normal_t, _x_noiser_t = ddpm_sample.norm_guided_one_step_denoising(
                unet_model, aug_image, anomaly_label, args
            )

            aux_mask = seg_model(torch.cat((aug_image, pred_x0), dim=1))
            main_mask = main_head(torch.cat((aug_image, pred_x0), dim=1))

            aux_loss = loss_focal(aux_mask, anomaly_mask) + loss_smL1(aux_mask, anomaly_mask)
            main_loss = loss_focal(main_mask, anomaly_mask) + loss_smL1(main_mask, anomaly_mask)
            kd_loss = loss_kd(main_mask, aux_mask.detach())

            loss = noise_loss + lambda_aux * aux_loss + lambda_main * main_loss + lambda_kd * kd_loss

            optimizer_ddpm.zero_grad()
            optimizer_seg.zero_grad()
            optimizer_head.zero_grad()
            loss.backward()
            optimizer_ddpm.step()
            optimizer_seg.step()
            optimizer_head.step()
            scheduler_seg.step()

            train_loss += loss.item()
            if is_main_process(rank):
                tbar.set_postfix(loss=f"{train_loss:.3f}")

        if (epoch + 1) % eval_interval == 0 and epoch > 0:
            should_stop = False

            if is_main_process(rank):
                if testing_loader is None:
                    raise RuntimeError("Rank-0 requires a valid testing_loader for evaluation")

                # Rank-0-only evaluation must use local modules (not DDP wrappers)
                # to avoid cross-rank collectives while other ranks wait at broadcast.
                eval_metrics = evaluate(
                    testing_loader,
                    args,
                    unwrap_model(unet_model),
                    unwrap_model(main_head),
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
                    save_models(unet_model, seg_model, main_head, args=args, final="best", epoch=epoch + 1, sub_class=sub_class)
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
        save_models(unet_model, seg_model, main_head, args=args, final="last", epoch=last_epoch_ran, sub_class=sub_class)

        if early_stopped:
            print(f"[{sub_class}] Training stopped early at epoch {last_epoch_ran}/{args['EPOCHS']}.")

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
        summary_df = pd.DataFrame(summary_row)
        out_csv = (
            f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/'
            f'{args["eval_normal_t"]}_{args["eval_noisier_t"]}t_{args["condition_w"]}_'
            f'{class_type}_image_pixel_auroc_train_dualpath.csv'
        )
        summary_df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)


def main():
    parser = argparse.ArgumentParser(description="Dual-path baseline: train with aux branch, test with main branch only")
    parser.add_argument("--config", default="args/args_dualpath_baseline.json", help="Path to args json")
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
        print("Train mode: dual-path (unet + seg + main-head)")
        print("Test mode: main branch only, map mode:", args["test_anomaly_map_mode"])
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

        # Evaluate only on rank 0 to avoid duplicate metrics and conflicting writes.
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
