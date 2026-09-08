import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from statistics import mean, median

import numpy as np
import torch
from torch.utils.data import DataLoader

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
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_class_set(class_set: str):
    if class_set.strip():
        return [x.strip() for x in class_set.split(",") if x.strip()]
    return list(MVTEC_CLASSES)


def infer_arg_num(raw_args, config_path):
    base_name = os.path.splitext(os.path.basename(config_path))[0]
    arg_num = raw_args.get("arg_num", "")
    if not arg_num:
        arg_num = base_name.replace("args", "").strip("_")
        if not arg_num:
            arg_num = base_name
    return arg_num


def load_args(config_arg: str):
    config_path = resolve_config_path(config_arg)
    with open(config_path, "r") as f:
        raw_args = json.load(f)

    raw_args["arg_num"] = infer_arg_num(raw_args, config_path)
    args = defaultdict_from_json(raw_args)

    if args["loss-type"] == "" and args["loss_type"] != "":
        args["loss-type"] = args["loss_type"]

    return args, config_path


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
        return VisATestDataset(subclass_path, sub_class, img_size=args["img_size"])
    if class_type == "MPDD":
        subclass_path = os.path.join(args["mpdd_root_path"], sub_class)
        return MPDDTestDataset(subclass_path, sub_class, img_size=args["img_size"])
    if class_type == "MVTec":
        subclass_path = os.path.join(args["mvtec_root_path"], sub_class)
        return MVTecTestDataset(subclass_path, sub_class, img_size=args["img_size"])

    subclass_path = os.path.join(args["dagm_root_path"], sub_class)
    return DAGMTestDataset(subclass_path, sub_class, img_size=args["img_size"])


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
            raise RuntimeError("UNetModelMamba is unavailable. Please check mamba dependencies.")

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

    raise ValueError(f"Unsupported unet implementation: {unet_impl}")


def resolve_checkpoint_path(args, sub_class, checkpoint_type, allow_fallback=False):
    root = f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}'
    preferred = os.path.join(root, f"params-{checkpoint_type}.pt")
    if os.path.exists(preferred):
        return preferred, checkpoint_type

    if not allow_fallback:
        raise FileNotFoundError(f"Checkpoint not found: {preferred}")

    fallback_type = "best" if checkpoint_type == "last" else "last"
    fallback = os.path.join(root, f"params-{fallback_type}.pt")
    if os.path.exists(fallback):
        return fallback, fallback_type

    raise FileNotFoundError(
        f"Checkpoint not found: {preferred}; fallback missing: {fallback}"
    )


def load_checkpoint(args, device, sub_class, checkpoint_type, allow_fallback=False):
    ck_path, used_type = resolve_checkpoint_path(
        args=args,
        sub_class=sub_class,
        checkpoint_type=checkpoint_type,
        allow_fallback=allow_fallback,
    )
    return torch.load(ck_path, map_location=device, weights_only=False), ck_path, used_type


def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def collect_image_samples(dataset, max_samples, num_workers, pin_memory):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )

    images = []
    for sample in loader:
        images.append(sample["image"].contiguous())
        if max_samples > 0 and len(images) >= max_samples:
            break

    return images


def run_one_forward(ddpm_sample, unet_model, seg_model, image_cpu, args, device, use_amp):
    image = image_cpu.to(device, non_blocking=True)
    normal_t_tensor = torch.tensor([args["eval_normal_t"]], device=image.device).repeat(image.shape[0])
    noiser_t_tensor = torch.tensor([args["eval_noisier_t"]], device=image.device).repeat(image.shape[0])

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            amp_ctx = torch.cuda.amp.autocast()
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            (
                _loss,
                pred_x_0_condition,
                _pred_x_0_normal,
                _pred_x_0_noisier,
                _x_normal_t,
                _x_noiser_t,
                _pred_x_t_noisier,
            ) = ddpm_sample.norm_guided_one_step_denoising_eval(
                unet_model, image, normal_t_tensor, noiser_t_tensor, args
            )

            _pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1))

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000.0


def benchmark_one_class(
    exp_name,
    args,
    unet_impl,
    sub_class,
    checkpoint_type,
    device,
    max_samples_per_class,
    warmup,
    repeats,
    num_workers,
    pin_memory,
    use_amp,
    allow_checkpoint_fallback,
):
    checkpoint, ck_path, ck_used_type = load_checkpoint(
        args,
        device,
        sub_class,
        checkpoint_type,
        allow_fallback=allow_checkpoint_fallback,
    )

    test_dataset = build_test_dataset(args, sub_class)
    cached_images = collect_image_samples(test_dataset, max_samples_per_class, num_workers, pin_memory)
    if len(cached_images) == 0:
        raise RuntimeError(f"No test images found for class: {sub_class}")

    warmup_count = min(max(0, int(warmup)), len(cached_images))
    timed_images = cached_images[warmup_count:]
    if len(timed_images) == 0:
        timed_images = cached_images

    unet_model = build_unet_model(args, device, unet_impl)
    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)
    unet_model.load_state_dict(checkpoint["unet_model_state_dict"])
    seg_model.load_state_dict(checkpoint["seg_model_state_dict"])
    unet_model.eval()
    seg_model.eval()

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

    for image_cpu in cached_images[:warmup_count]:
        run_one_forward(ddpm_sample, unet_model, seg_model, image_cpu, args, device, use_amp)

    repeat_rows = []
    for repeat_idx in range(max(1, int(repeats))):
        sample_latencies_ms = []
        for image_cpu in timed_images:
            latency_ms = run_one_forward(ddpm_sample, unet_model, seg_model, image_cpu, args, device, use_amp)
            sample_latencies_ms.append(latency_ms)

        avg_latency_ms = float(mean(sample_latencies_ms))
        p50_latency_ms = float(np.percentile(sample_latencies_ms, 50))
        p90_latency_ms = float(np.percentile(sample_latencies_ms, 90))
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else float("nan")

        repeat_rows.append(
            {
                "experiment": exp_name,
                "arg_num": args["arg_num"],
                "unet_impl": unet_impl,
                "classname": sub_class,
                "repeat_idx": repeat_idx,
                "dataset_images": len(test_dataset),
                "cached_images": len(cached_images),
                "warmup_images": warmup_count,
                "timed_images": len(timed_images),
                "checkpoint_type_requested": checkpoint_type,
                "checkpoint_type_used": ck_used_type,
                "checkpoint_path": ck_path,
                "best_epoch_or_last": checkpoint.get("n_epoch", ""),
                "avg_latency_ms": avg_latency_ms,
                "p50_latency_ms": p50_latency_ms,
                "p90_latency_ms": p90_latency_ms,
                "fps": fps,
                "eval_normal_t": args["eval_normal_t"],
                "eval_noisier_t": args["eval_noisier_t"],
                "condition_w": args["condition_w"],
                "amp": int(bool(use_amp and device.type == "cuda")),
            }
        )

    class_summary = {
        "experiment": exp_name,
        "arg_num": args["arg_num"],
        "unet_impl": unet_impl,
        "classname": sub_class,
        "dataset_images": len(test_dataset),
        "cached_images": len(cached_images),
        "warmup_images": warmup_count,
        "timed_images": len(timed_images),
        "checkpoint_type_requested": checkpoint_type,
        "checkpoint_type_used": ck_used_type,
        "checkpoint_path": ck_path,
        "best_epoch_or_last": checkpoint.get("n_epoch", ""),
        "fps_mean_over_repeats": float(mean([r["fps"] for r in repeat_rows])),
        "fps_median_over_repeats": float(median([r["fps"] for r in repeat_rows])),
        "latency_ms_mean_over_repeats": float(mean([r["avg_latency_ms"] for r in repeat_rows])),
        "latency_ms_median_over_repeats": float(median([r["avg_latency_ms"] for r in repeat_rows])),
        "p90_ms_mean_over_repeats": float(mean([r["p90_latency_ms"] for r in repeat_rows])),
    }

    return repeat_rows, class_summary


def write_csv(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_experiment_summary(class_rows):
    grouped = {}
    for row in class_rows:
        grouped.setdefault(row["experiment"], []).append(row)

    summary_rows = []
    for exp_name, rows in grouped.items():
        fps_values = [r["fps_median_over_repeats"] for r in rows]
        lat_values = [r["latency_ms_median_over_repeats"] for r in rows]
        p90_values = [r["p90_ms_mean_over_repeats"] for r in rows]

        summary_rows.append(
            {
                "experiment": exp_name,
                "n_classes": len(rows),
                "fps_mean_over_classes": float(mean(fps_values)),
                "fps_median_over_classes": float(median(fps_values)),
                "latency_ms_mean_over_classes": float(mean(lat_values)),
                "latency_ms_median_over_classes": float(median(lat_values)),
                "p90_ms_mean_over_classes": float(mean(p90_values)),
            }
        )

    return summary_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fair inference speed benchmark for orig_1 / mamba_low / mamba_low_unidir. "
            "Uses fixed class list, fixed sample count, warmup, and repeated timing."
        )
    )
    parser.add_argument("--orig-config", default="args/args1.json", help="Config for original baseline model")
    parser.add_argument("--low-config", default="args/args_mamba_low.json", help="Config for mamba_low model")
    parser.add_argument(
        "--unidir-config",
        default="args/args_mamba_low_unidir.json",
        help="Config for mamba_low_unidir model",
    )
    parser.add_argument("--checkpoint-type", default="last", choices=["best", "last"], help="Checkpoint type")
    parser.add_argument(
        "--allow-checkpoint-fallback",
        action="store_true",
        help="When requested checkpoint type is missing for a class, automatically fallback to the other type.",
    )
    parser.add_argument(
        "--align-classes",
        default="intersection",
        choices=["intersection", "strict"],
        help=(
            "How to select classes across experiments. "
            "intersection: use only classes available in all experiments for requested checkpoint type. "
            "strict: use class-set exactly as provided."
        ),
    )
    parser.add_argument(
        "--class-set",
        default=",".join(MVTEC_CLASSES),
        help="Comma-separated classes to benchmark. Default is full MVTec-15.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=100,
        help="Max number of test images cached per class. <=0 means use all.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup images per class (not timed)")
    parser.add_argument("--repeats", type=int, default=5, help="Timing repeats per class")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers for sample caching")
    parser.add_argument("--pin-memory", action="store_true", help="Enable dataloader pin_memory while caching")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Benchmark device")
    parser.add_argument("--amp", action="store_true", help="Enable fp16 autocast on CUDA")
    parser.add_argument(
        "--detail-csv",
        default="outputs/metrics/speed_benchmark_detail.csv",
        help="Output CSV for repeat-level timing rows",
    )
    parser.add_argument(
        "--class-csv",
        default="outputs/metrics/speed_benchmark_class_summary.csv",
        help="Output CSV for per-class aggregated timing",
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/metrics/speed_benchmark_experiment_summary.csv",
        help="Output CSV for experiment-level summary",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing checkpoints/classes instead of stopping",
    )
    cli_args = parser.parse_args()

    set_seed(cli_args.seed)

    if cli_args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested.")
    device = torch.device(cli_args.device)

    classes = parse_class_set(cli_args.class_set)

    exp_specs = [
        ("orig_1", cli_args.orig_config, "baseline"),
        ("mamba_low", cli_args.low_config, "mamba"),
        ("mamba_low_unidir", cli_args.unidir_config, "mamba"),
    ]

    loaded_specs = []
    for exp_name, config_arg, unet_impl in exp_specs:
        args, config_path = load_args(config_arg)
        loaded_specs.append((exp_name, args, config_path, unet_impl))

    # Build class coverage report and optionally align to intersection for fair comparison.
    requested_classes = list(classes)
    available_sets = {}
    missing_by_exp = {}
    for exp_name, args, _config_path, _unet_impl in loaded_specs:
        available = []
        missing = []
        for sub_class in requested_classes:
            try:
                resolve_checkpoint_path(
                    args=args,
                    sub_class=sub_class,
                    checkpoint_type=cli_args.checkpoint_type,
                    allow_fallback=cli_args.allow_checkpoint_fallback,
                )
                available.append(sub_class)
            except FileNotFoundError:
                missing.append(sub_class)
        available_sets[exp_name] = set(available)
        missing_by_exp[exp_name] = missing

    if cli_args.align_classes == "intersection":
        class_set = set(requested_classes)
        for exp_name in available_sets:
            class_set = class_set.intersection(available_sets[exp_name])
        classes = sorted(class_set)
        dropped = [c for c in requested_classes if c not in class_set]
        if dropped:
            print("[align-classes] dropped classes not shared by all experiments:", dropped)
    else:
        classes = requested_classes

    print("\nCheckpoint coverage report:")
    for exp_name in [x[0] for x in loaded_specs]:
        print(f"  {exp_name}: missing={missing_by_exp[exp_name]}")

    if len(classes) == 0:
        raise RuntimeError("No classes left to benchmark after class alignment and checkpoint filtering.")

    detail_rows = []
    class_rows = []

    print("Benchmark classes:", classes)
    print("Device:", device)
    print("Checkpoint:", cli_args.checkpoint_type)
    print("Allow checkpoint fallback:", bool(cli_args.allow_checkpoint_fallback))
    print("Class alignment:", cli_args.align_classes)
    print("Warmup images:", cli_args.warmup)
    print("Repeats:", cli_args.repeats)
    print("Max samples per class:", cli_args.max_samples_per_class)

    for exp_name, args, config_path, unet_impl in loaded_specs:
        print(f"\n[experiment] {exp_name}")
        print("  config:", config_path)
        print("  arg_num:", args["arg_num"])
        print("  output_path:", args["output_path"])
        print("  unet_impl:", unet_impl)

        for sub_class in classes:
            print(f"  - class: {sub_class}")
            try:
                rep_rows, class_summary = benchmark_one_class(
                    exp_name=exp_name,
                    args=args,
                    unet_impl=unet_impl,
                    sub_class=sub_class,
                    checkpoint_type=cli_args.checkpoint_type,
                    device=device,
                    max_samples_per_class=cli_args.max_samples_per_class,
                    warmup=cli_args.warmup,
                    repeats=cli_args.repeats,
                    num_workers=cli_args.num_workers,
                    pin_memory=cli_args.pin_memory,
                    use_amp=cli_args.amp,
                    allow_checkpoint_fallback=cli_args.allow_checkpoint_fallback,
                )
            except Exception as e:
                if cli_args.skip_missing:
                    print(f"    skip: {e}")
                    continue
                raise

            detail_rows.extend(rep_rows)
            class_rows.append(class_summary)

            print(
                "    done: "
                f"fps_med={class_summary['fps_median_over_repeats']:.2f}, "
                f"lat_med={class_summary['latency_ms_median_over_repeats']:.2f} ms"
            )

    if not detail_rows:
        print("No results collected. Nothing to write.")
        return

    exp_summary_rows = build_experiment_summary(class_rows)

    write_csv(cli_args.detail_csv, detail_rows)
    write_csv(cli_args.class_csv, class_rows)
    write_csv(cli_args.summary_csv, exp_summary_rows)

    print("\nSaved detail CSV:", cli_args.detail_csv)
    print("Saved class CSV:", cli_args.class_csv)
    print("Saved summary CSV:", cli_args.summary_csv)

    print("\nExperiment-level summary:")
    for row in exp_summary_rows:
        print(
            f"  {row['experiment']}: "
            f"fps_mean={row['fps_mean_over_classes']:.2f}, "
            f"fps_median={row['fps_median_over_classes']:.2f}, "
            f"lat_mean={row['latency_ms_mean_over_classes']:.2f} ms"
        )


if __name__ == "__main__":
    main()
