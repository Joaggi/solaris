from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime, timedelta
import json
from pathlib import Path

import argparse


APP_NAME = os.environ.get("SOLARIS_MODAL_APP_NAME", "solaris-pretrain")
DATASET_REPO = os.environ.get("SOLARIS_DATASET_REPO", "hrrsmjd/AIA_12hour_512x512")
VOLUME_NAME = os.environ.get("SOLARIS_MODAL_VOLUME_NAME", "solaris-aia-data")
WANDB_SECRET_NAME = os.environ.get("SOLARIS_MODAL_WANDB_SECRET_NAME", "")
VOLUME_ROOT = Path("/home/joag/088-solaris/")
DATA_DIR = VOLUME_ROOT / "data/AIA_12hour_512x512"
SPLIT_DIR = DATA_DIR / "splits"
CHECKPOINT_DIR = VOLUME_ROOT / "checkpoints"
PLOT_DIR = VOLUME_ROOT / "plots"
EVAL_DIR = VOLUME_ROOT / "eval"
SCALE_FACTORS_PATH = DATA_DIR / "train_scale_factors.json"
SPLIT_METADATA_PATH = SPLIT_DIR / "split_metadata.json"
DEFAULT_PATCH_SIZE = 8
SPLIT_SCHEME = "chronological_valid_samples_80_10_10_guard_24h_v1"

WAVELENGTHS = ("0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600")
FALLBACK_SCALE_FACTORS = (
    59.55241278048226,
    217.18662860219484,
    1619.8840310287178,
    2573.643248589347,
    1191.8919143907103,
    889.3599319561527,
    113.23721481062092,
    267.50519244567655,
)


def _timestamp_parts(dt: datetime) -> list[str]:
    return [str(dt.year), f"{dt.month:02d}", f"{dt.day:02d}", f"H{dt.hour:02d}00"]


def _hdf5_path(dt: datetime) -> str:
    return f"{dt.year}/{dt.month:02d}/{dt.day:02d}/H{dt.hour:02d}00"


def _is_present(ds) -> bool:
    exists = ds.attrs.get("exists", True)
    if isinstance(exists, bytes):
        exists = exists.decode("utf-8")
    if isinstance(exists, str):
        return exists.lower() not in {"false", "0", "no"}
    return bool(exists)


def _build_split_files(force: bool = False) -> dict[str, int]:
    import h5py

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_paths = {name: SPLIT_DIR / f"{name}_id.txt" for name in ("train", "val", "test")}
    if not force and all(path.exists() for path in split_paths.values()):
        counts = {name: sum(1 for _ in path.open("r", encoding="utf-8")) for name, path in split_paths.items()}
        metadata = {}
        if SPLIT_METADATA_PATH.exists():
            metadata = json.loads(SPLIT_METADATA_PATH.read_text(encoding="utf-8"))
        if min(counts.values()) >= 100 and metadata.get("scheme") == SPLIT_SCHEME:
            return counts

    h5_files = sorted(DATA_DIR.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {DATA_DIR}.")

    handles: dict[int, h5py.File] = {}

    def handle_for(year: int):
        if year not in handles:
            matches = sorted(DATA_DIR.glob(f"*{year}.h5"))
            if not matches:
                return None
            handles[year] = h5py.File(matches[0], "r")
        return handles[year]

    def sample_is_valid(current_dt: datetime) -> bool:
        for dt in (current_dt - timedelta(hours=12), current_dt, current_dt + timedelta(hours=12)):
            handle = handle_for(dt.year)
            if handle is None:
                return False
            group_path = _hdf5_path(dt)
            if group_path not in handle:
                return False
            group = handle[group_path]
            for wavelength in WAVELENGTHS:
                if wavelength not in group or not _is_present(group[wavelength]):
                    return False
        return True

    valid_rows = []
    current = datetime(2010, 7, 1, 12)
    end = datetime(2023, 12, 31, 0)
    while current <= end:
        if sample_is_valid(current):
            valid_rows.append(" ".join(_timestamp_parts(current)))
        current += timedelta(hours=12)

    for handle in handles.values():
        handle.close()

    train_split = int(0.8 * len(valid_rows))
    val_split = int(0.9 * len(valid_rows))
    split_rows = {
        "train": valid_rows[:train_split],
        "val": valid_rows[train_split:val_split],
        "test": valid_rows[val_split:],
    }
    split_rows["train"] = split_rows["train"][:-2] if len(split_rows["train"]) > 2 else split_rows["train"]
    split_rows["val"] = split_rows["val"][2:-2] if len(split_rows["val"]) > 4 else split_rows["val"]
    split_rows["test"] = split_rows["test"][2:] if len(split_rows["test"]) > 2 else split_rows["test"]

    for name, rows in split_rows.items():
        split_paths[name].write_text("\n".join(rows) + "\n", encoding="utf-8")
    SPLIT_METADATA_PATH.write_text(
        json.dumps(
            {
                "scheme": SPLIT_SCHEME,
                "valid_sample_count_before_guard": len(valid_rows),
                "counts": {name: len(rows) for name, rows in split_rows.items()},
                "guard_band_hours": 24,
                "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {name: len(rows) for name, rows in split_rows.items()}


def _open_year_handle(handles, year, h5py):
    if year not in handles:
        matches = sorted(DATA_DIR.glob(f"*{year}.h5"))
        if not matches:
            return None
        handles[year] = h5py.File(matches[0], "r")
    return handles[year]


def _compute_train_scale_factors(force: bool = False) -> list[float]:
    import h5py
    import numpy as np

    if SCALE_FACTORS_PATH.exists() and not force:
        with SCALE_FACTORS_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("split_scheme") == SPLIT_SCHEME:
            return metadata["scale_factors"]

    split_counts = _build_split_files()
    if split_counts["train"] == 0:
        raise ValueError("Cannot compute scale factors because train split is empty.")

    timestamps: set[tuple[str, str, str, str]] = set()
    train_path = SPLIT_DIR / "train_missing_channel_id.txt"
    for row in train_path.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        current = row.split()
        for offset in (-12, 0, 12):
            timestamps.add(tuple(_timestamp_parts(_parts_to_datetime(current) + timedelta(hours=offset))))

    handles = {}
    max_sums = np.zeros(len(WAVELENGTHS), dtype=np.float64)
    counts = np.zeros(len(WAVELENGTHS), dtype=np.int64)
    for year, month, day, hour in sorted(timestamps):
        handle = _open_year_handle(handles, int(year), h5py)
        if handle is None:
            continue
        group_path = f"{year}/{month}/{day}/{hour}"
        if group_path not in handle:
            continue
        group = handle[group_path]
        for index, wavelength in enumerate(WAVELENGTHS):
            if wavelength not in group or not _is_present(group[wavelength]):
                continue
            max_sums[index] += float(np.asarray(group[wavelength]).max())
            counts[index] += 1

    for handle in handles.values():
        handle.close()

    if np.any(counts == 0):
        missing = [WAVELENGTHS[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"Cannot compute scale factors for wavelengths without samples: {missing}")

    scale_factors = (0.5 * max_sums / counts).tolist()
    SCALE_FACTORS_PATH.write_text(
        json.dumps(
            {
                "wavelengths": WAVELENGTHS,
                "scale_factors": scale_factors,
                "counts": counts.tolist(),
                "source": "half average per-image maximum over unique train-split timestamps",
                "split_scheme": SPLIT_SCHEME,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return scale_factors


def _parts_to_datetime(parts: list[str]) -> datetime:
    return datetime(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3][1:3]))


def _ensure_dataset(force_download: bool = False, force_splits: bool = False) -> dict[str, int]:
    from huggingface_hub import snapshot_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if force_download or not any(DATA_DIR.glob("*.h5")):
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
            allow_patterns=["*.h5"],
        )

    counts = _build_split_files(force=force_splits)
    _compute_train_scale_factors(force=force_splits)
    return counts


def _collate_pretrain(batch):
    import torch
    from torch.utils.data import default_collate

    data, target, timestamps = zip(*batch)
    return default_collate(data), default_collate(target), tuple(timestamps)


def _make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    import random

    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _normalise_raw(data, scale_factors):
    if data.dim() == 5:
        return data / scale_factors.view(1, 1, -1, 1, 1)
    if data.dim() == 4:
        return data / scale_factors.view(1, -1, 1, 1)
    raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(data.shape)}.")


def _unscale_prediction(prediction, scale_factors):
    return prediction * scale_factors.view(1, -1, 1, 1)


def _paper_weighted_mae(prediction_normalised, target_raw, scale_factors):
    import torch

    target_normalised = _normalise_raw(target_raw, scale_factors)
    return torch.abs(prediction_normalised - target_normalised).mean()


def _raw_rmse(prediction_normalised, target_raw, scale_factors):
    import torch

    prediction_raw = _unscale_prediction(prediction_normalised, scale_factors)
    per_channel = torch.sqrt(torch.mean((prediction_raw - target_raw) ** 2, dim=(0, 2, 3)))
    return per_channel, per_channel.mean()


def _param_groups(model, extra_no_decay, weight_decay):
    decay_params = []
    no_decay_params = []
    no_decay_names = ("pos_embed", "absolute_time_embed", "lead_time_embed")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or any(key in name for key in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    for param in extra_no_decay:
        if param.requires_grad:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _checkpoint_path_for_run(run_name: str) -> Path:
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_final_step_07750.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_latest.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    raise FileNotFoundError(f"No checkpoint found for run {run_name!r} under {CHECKPOINT_DIR}.")


class CpuEMA:
    def __init__(self, model, norm_coeff_1, norm_coeff_2, decay: float = 0.999, update_every: int = 10):
        import torch

        self.decay = decay
        self.update_every = max(1, update_every)
        self.num_updates = 0
        self.model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        self.norm_coeff_1 = norm_coeff_1.detach().cpu().clone()
        self.norm_coeff_2 = norm_coeff_2.detach().cpu().clone()
        self._torch = torch

    def load_state_dict(self, checkpoint: dict) -> None:
        self.model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in checkpoint["ema_model_state_dict"].items()
        }
        self.norm_coeff_1 = checkpoint["ema_norm_coeff_1"].detach().cpu().clone()
        self.norm_coeff_2 = checkpoint["ema_norm_coeff_2"].detach().cpu().clone()
        self.num_updates = int(checkpoint.get("ema_num_updates", 0))

    def update(self, model, norm_coeff_1, norm_coeff_2, step: int) -> None:
        if step % self.update_every != 0:
            return
        self.num_updates += 1
        with self._torch.no_grad():
            for name, tensor in model.state_dict().items():
                source = tensor.detach().cpu()
                target = self.model_state[name]
                if self._torch.is_floating_point(target):
                    target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
                else:
                    target.copy_(source)
            self.norm_coeff_1.mul_(self.decay).add_(norm_coeff_1.detach().cpu(), alpha=1.0 - self.decay)
            self.norm_coeff_2.mul_(self.decay).add_(norm_coeff_2.detach().cpu(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {
            "ema_model_state_dict": self.model_state,
            "ema_norm_coeff_1": self.norm_coeff_1,
            "ema_norm_coeff_2": self.norm_coeff_2,
            "ema_decay": self.decay,
            "ema_update_every": self.update_every,
            "ema_num_updates": self.num_updates,
        }


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    norm_coeff_1,
    norm_coeff_2,
    step: int,
    loss_value: float,
    scale_factors,
    patch_size: int,
    seed: int,
    ema: CpuEMA | None = None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "loss": loss_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "norm_coeff_1": norm_coeff_1.detach().cpu(),
        "norm_coeff_2": norm_coeff_2.detach().cpu(),
        "wavelengths": WAVELENGTHS,
        "scale_factors": scale_factors,
        "patch_size": patch_size,
        "seed": seed,
    }
    if ema is not None:
        checkpoint.update(ema.state_dict())
    torch.save(checkpoint, path)


def prepare_dataset(force_download: bool = False, force_splits: bool = False) -> dict[str, int]:
    counts = _ensure_dataset(force_download=force_download, force_splits=force_splits)
    return counts



def list_checkpoints() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(str(path.relative_to(VOLUME_ROOT)) for path in CHECKPOINT_DIR.glob("*.pt"))


def get_scale_factors() -> list[float]:
    _ensure_dataset()
    return _compute_train_scale_factors()


def train_missing_channel(
    max_steps: int = 7750,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    patch_size: int = DEFAULT_PATCH_SIZE,
    seed: int = 42,
    deterministic: bool = False,
    lr: float = 5e-4,
    min_lr: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.05,
    checkpoint_every: int = 250,
    num_workers: int = 8,
    stop_after_seconds: int | None = None,
    backbone_name: str = "local_pretrain",
    run_name: str = "solaris_missing_channel",
    resume: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "solaris",
    wandb_entity: str | None = None,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    ema_update_every: int = 10,
) -> str:
    sys.path.insert(0, "/root/solaris")

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from solaris.load_data import CustomDataset_missing_channel
    from solaris.model.solaris import SolarisSmall
    from solaris.normalization import transform
    from solaris.utils_data import build_metadata

    _seed_everything(seed, deterministic=deterministic)
    split_counts = _ensure_dataset()
    print(f"Split counts: {split_counts}")

    device = torch.device("cuda:1")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = CustomDataset_missing_channel(root_dir=DATA_DIR, data_set="train_missing_channel", id_dir=SPLIT_DIR)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_pretrain,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    model = SolarisSmall(out_levels=len(WAVELENGTHS), patch_size=patch_size).to(device)
    norm_coeff_1 = torch.nn.Parameter(torch.tensor(0.5, device=device))
    norm_coeff_2 = torch.nn.Parameter(torch.tensor(0.5, device=device))
    optimizer = AdamW(
        _param_groups(model, [norm_coeff_1, norm_coeff_2], weight_decay),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    all_optim_params = [p for g in optimizer.param_groups for p in g["params"]]
    scheduler = _make_scheduler(optimizer, warmup_steps, max_steps, min_lr / lr)
    scale_factor_values = _compute_train_scale_factors()
    print(f"scale_factors={scale_factor_values}", flush=True)
    scale_factors = torch.tensor(scale_factor_values, device=device, dtype=torch.float32)
    ema = CpuEMA(model, norm_coeff_1, norm_coeff_2, decay=ema_decay, update_every=ema_update_every) if use_ema else None

    model_backbone = SolarisSmall(out_levels=len(WAVELENGTHS), patch_size=patch_size).to(device)

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_config = {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps,
            "patch_size": patch_size,
            "seed": seed,
            "deterministic": deterministic,
            "lr": lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
            "checkpoint_every": checkpoint_every,
            "use_ema": use_ema,
            "ema_decay": ema_decay,
            "ema_update_every": ema_update_every,
            "wavelengths": WAVELENGTHS,
            "scale_factors": scale_factor_values,
        }
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            id=run_name,
            resume="allow",
            config=wandb_config,
        )

    global_step = 0
    # Loading backbone
    latest_path = CHECKPOINT_DIR / f"{backbone_name}_latest.pt"
    if resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model_backbone.load_state_dict(checkpoint["model_state_dict"])
        print(f"resumed_checkpoint={latest_path} step={global_step}", flush=True)
        if checkpoint.get("seed") not in (None, seed):
            print(f"warning=checkpoint_seed_mismatch checkpoint_seed={checkpoint.get('seed')} requested_seed={seed}", flush=True)

    # Assign the pretrain backbone to the current model
    model.backbone = model_backbone.backbone
    model.encoder = model_backbone.encoder

    start_time = time.monotonic()
    last_loss = float("nan")
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        for batch_idx, (data, target, timestamps) in enumerate(loader):
            data = data.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            data = transform(data, norm_coeff_1, norm_coeff_2, scale_factors)
            metadata = build_metadata(data, timestamps)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(data, metadata, 12, 0).squeeze(1)
            loss = _paper_weighted_mae(output.float(), target.float(), scale_factors)
            loss = loss / grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(all_optim_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_loss = float(loss.detach().cpu()) * grad_accum_steps
            if ema is not None:
                ema.update(model, norm_coeff_1, norm_coeff_2, global_step)

            if global_step == 1 or global_step % 10 == 0:
                with torch.no_grad():
                    rmse_channels, rmse_mean = _raw_rmse(output.detach().float(), target, scale_factors)
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.monotonic() - start_time
                print(
                    f"step={global_step} loss={last_loss:.6f} lr={current_lr:.6e} "
                    f"rmse_mean_raw={float(rmse_mean.detach().cpu()):.3f} "
                    f"rmse_raw={[round(float(v), 3) for v in rmse_channels.detach().cpu()]} "
                    f"elapsed_min={elapsed / 60:.1f}",
                    flush=True,
                )
                if wandb_run is not None:
                    log_data = {
                        "train/loss_weighted_mae": last_loss,
                        "train/lr": current_lr,
                        "train/rmse_mean_raw": float(rmse_mean.detach().cpu()),
                        "train/elapsed_min": elapsed / 60,
                    }
                    for wavelength, value in zip(WAVELENGTHS, rmse_channels.detach().cpu()):
                        log_data[f"train/rmse_raw/{wavelength}"] = float(value)
                    wandb_run.log(log_data, step=global_step)

            if global_step % checkpoint_every == 0 or global_step >= max_steps:
                ckpt_path = CHECKPOINT_DIR / f"{run_name}_step_{global_step:05d}.pt"
                _save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    norm_coeff_1,
                    norm_coeff_2,
                    global_step,
                    last_loss,
                    scale_factor_values,
                    patch_size,
                    seed,
                    ema,
                )
                _save_checkpoint(
                    CHECKPOINT_DIR / f"{run_name}_latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    norm_coeff_1,
                    norm_coeff_2,
                    global_step,
                    last_loss,
                    scale_factor_values,
                    patch_size,
                    seed,
                    ema,
                )
                print(f"saved_checkpoint={ckpt_path}", flush=True)

            if stop_after_seconds is not None and time.monotonic() - start_time >= stop_after_seconds:
                ckpt_path = CHECKPOINT_DIR / f"{run_name}_quick_check_step_{global_step:05d}.pt"
                _save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    norm_coeff_1,
                    norm_coeff_2,
                    global_step,
                    last_loss,
                    scale_factor_values,
                    patch_size,
                    seed,
                    ema,
                )
                if wandb_run is not None:
                    wandb_run.finish()
                return str(ckpt_path)

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break
        if (batch_idx + 1) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(all_optim_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_loss = float(loss.detach().cpu()) * grad_accum_steps
            if ema is not None:
                ema.update(model, norm_coeff_1, norm_coeff_2, global_step)

    final_path = CHECKPOINT_DIR / f"{run_name}_final_step_{global_step:05d}.pt"
    _save_checkpoint(
        final_path,
        model,
        optimizer,
        scheduler,
        norm_coeff_1,
        norm_coeff_2,
        global_step,
        last_loss,
        scale_factor_values,
        patch_size,
        seed,
        ema,
    )
    if wandb_run is not None:
        wandb_run.finish()
    return str(final_path)

def plot_missing_channel(
    data_set: str = "val",
    sample_index: int = 0,
    run_name: str = "solaris_pretrain_paperloss_p8",
    use_ema: bool = False,
) -> str:
    sys.path.insert(0, "/root/solaris")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from solaris.load_data import CustomDataset_missing_channel
    from solaris.model.solaris import SolarisSmall
    from solaris.normalization import transform
    from solaris.utils_data import build_metadata

    _ensure_dataset()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _checkpoint_path_for_run(run_name)

    device = torch.device("cuda:1")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    wavelengths = tuple(checkpoint.get("wavelengths", WAVELENGTHS))
    scale_factors = torch.tensor(
        checkpoint.get("scale_factors", _compute_train_scale_factors()),
        device=device,
        dtype=torch.float32,
    )
    norm_coeff_1 = checkpoint["norm_coeff_1"].to(device=device, dtype=torch.float32)
    norm_coeff_2 = checkpoint["norm_coeff_2"].to(device=device, dtype=torch.float32)
    if use_ema and "ema_model_state_dict" in checkpoint:
        norm_coeff_1 = checkpoint["ema_norm_coeff_1"].to(device=device, dtype=torch.float32)
        norm_coeff_2 = checkpoint["ema_norm_coeff_2"].to(device=device, dtype=torch.float32)

    model = SolarisSmall(out_levels=len(wavelengths), patch_size=int(checkpoint.get("patch_size", DEFAULT_PATCH_SIZE))).to(device)
    model.load_state_dict(checkpoint["ema_model_state_dict"] if use_ema and "ema_model_state_dict" in checkpoint else checkpoint["model_state_dict"])
    model.eval()

    dataset = CustomDataset_missing_channel(root_dir=DATA_DIR, data_set=data_set, id_dir=SPLIT_DIR)
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index={sample_index} is outside {data_set} split length {len(dataset)}.")

    data, target, timestamp = dataset[sample_index]
    data_batch = data.unsqueeze(0).to(device=device)
    target_batch = target.unsqueeze(0).to(device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        transformed = transform(data_batch, norm_coeff_1, norm_coeff_2, scale_factors)
        metadata = build_metadata(transformed, (timestamp,))
        prediction_normalised = model(transformed, metadata, 12, 0).squeeze(1)
    prediction_raw = _unscale_prediction(prediction_normalised.float(), scale_factors)

    rmse = torch.sqrt(torch.mean((prediction_raw - target_batch.float()) ** 2, dim=(0, 2, 3)))
    mae = torch.mean(torch.abs(prediction_raw - target_batch.float()), dim=(0, 2, 3))

    inputs = data.cpu().numpy()
    target_np = target.cpu().numpy()
    prediction_np = prediction_raw.squeeze(0).detach().cpu().numpy()
    error_np = prediction_np - target_np

    def log_image(array):
        return np.log10(np.clip(array, a_min=0, a_max=None) + 1.0)

    rows = [
        ("input t", inputs[0], "magma"),
        ("input t", inputs[0], "magma"),
        ("target t+12h", target_np, "magma"),
        ("prediction t+12h", prediction_np, "magma"),
        ("prediction - target", error_np, "coolwarm"),
    ]
    fig, axes = plt.subplots(len(rows), len(wavelengths), figsize=(2.6 * len(wavelengths), 12), constrained_layout=True)
    for col, wavelength in enumerate(wavelengths):
        raw_stack = np.stack((inputs[0, col], inputs[0, col], target_np[col], prediction_np[col]))
        log_stack = log_image(raw_stack)
        vmin, vmax = np.percentile(log_stack, [1, 99.7])
        max_abs_error = np.percentile(np.abs(error_np[col]), 99.5)
        max_abs_error = float(max(max_abs_error, 1e-6))

        for row, (label, images, cmap) in enumerate(rows):
            axis = axes[row, col]
            if row < 4:
                image_data = log_image(images[col])
                axis.imshow(image_data, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                axis.imshow(images[col], cmap=cmap, vmin=-max_abs_error, vmax=max_abs_error)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"{wavelength} A\nRMSE {float(rmse[col]):.2f}", fontsize=9)
            if col == 0:
                axis.set_ylabel(label, fontsize=10)
            if row == len(rows) - 1:
                axis.set_xlabel(f"MAE {float(mae[col]):.2f}", fontsize=8)

    fig.suptitle(
        f"Solaris-S checkpoint {checkpoint_path.name}{' EMA' if use_ema and 'ema_model_state_dict' in checkpoint else ''} | {data_set}[{sample_index}] | current time {timestamp.isoformat()}",
        fontsize=12,
    )
    ema_suffix = "_ema" if use_ema and "ema_model_state_dict" in checkpoint else ""
    output_path = PLOT_DIR / f"{run_name}{ema_suffix}_{data_set}_{sample_index:04d}_prediction.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)



def evaluate_mse_subset(
    data_set: str = "test_missing_channel",
    max_samples: int = 64,
    batch_size: int = 4,
    run_name: str = "missing_channel",
    use_ema: bool = False,
) -> dict:
    sys.path.insert(0, "/root/solaris")

    import torch
    from torch.utils.data import DataLoader, Subset

    from solaris.load_data import CustomDataset_missing_channel
    from solaris.model.solaris import SolarisSmall
    from solaris.normalization import transform
    from solaris.utils_data import build_metadata

    _ensure_dataset()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _checkpoint_path_for_run(run_name)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    wavelengths = tuple(checkpoint.get("wavelengths", WAVELENGTHS))
    scale_factors = torch.tensor(
        checkpoint.get("scale_factors", _compute_train_scale_factors()),
        device=device,
        dtype=torch.float32,
    )
    norm_coeff_1 = checkpoint["norm_coeff_1"].to(device=device, dtype=torch.float32)
    norm_coeff_2 = checkpoint["norm_coeff_2"].to(device=device, dtype=torch.float32)
    has_ema = use_ema and "ema_model_state_dict" in checkpoint
    if has_ema:
        norm_coeff_1 = checkpoint["ema_norm_coeff_1"].to(device=device, dtype=torch.float32)
        norm_coeff_2 = checkpoint["ema_norm_coeff_2"].to(device=device, dtype=torch.float32)

    model = SolarisSmall(out_levels=len(wavelengths), patch_size=int(checkpoint.get("patch_size", DEFAULT_PATCH_SIZE))).to(device)
    model.load_state_dict(checkpoint["ema_model_state_dict"] if has_ema else checkpoint["model_state_dict"])
    model.eval()

    dataset = CustomDataset_missing_channel(root_dir=DATA_DIR, data_set=data_set, id_dir=SPLIT_DIR)
    if len(dataset) == 0:
        raise ValueError(f"{data_set!r} split is empty.")
    sample_count = min(max_samples, len(dataset))
    if sample_count <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}.")
    if sample_count == len(dataset):
        indices = list(range(len(dataset)))
    elif sample_count == 1:
        indices = [0]
    else:
        indices = torch.linspace(0, len(dataset) - 1, sample_count).round().to(torch.long).tolist()
        indices = sorted(dict.fromkeys(indices))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=_collate_pretrain,
    )

    total_squared_error_sum = torch.tensor(0, dtype=torch.float64, device=device)
    total_absolute_error_sum = torch.tensor(0, dtype=torch.float64, device=device)
    squared_error_sum = torch.zeros(len(wavelengths), dtype=torch.float64, device=device)
    absolute_error_sum = torch.zeros(len(wavelengths), dtype=torch.float64, device=device)
    sum_squares = torch.zeros(len(wavelengths), dtype=torch.float64, device=device)
    pixel_count = 0
    total_pixel_count = torch.tensor(0, dtype=torch.int64, device=device)
    evaluated_samples = torch.tensor(0, dtype=torch.int64, device=device)
    first_timestamp = None
    last_timestamp = None

    with torch.no_grad():
        for data, target, timestamps in loader:
            data = data.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True).float()
            transformed = transform(data, norm_coeff_1, norm_coeff_2, scale_factors)
            metadata = build_metadata(transformed, timestamps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction_normalised = model(transformed, metadata, 12, 0).squeeze(1)
            prediction_raw = _unscale_prediction(prediction_normalised.float(), scale_factors)
            error = prediction_raw - target
            total_squared_error_sum += torch.sum(error.double() ** 2)
            total_absolute_error_sum += torch.sum(torch.abs(error).double())
            squared_error_sum += torch.sum(error.double() ** 2, dim=(0, 2, 3))
            absolute_error_sum += torch.sum(torch.abs(error).double(), dim=(0, 2, 3))
            sum_squares += torch.sum(((target - target.mean(dim=(2,3), keepdim=True))**2).double(), dim=(0, 2, 3))

            pixel_count += target.shape[0] * target.shape[-2] * target.shape[-1]
            total_pixel_count += torch.prod(torch.tensor(target.size()))
            evaluated_samples += target.shape[0]
            first_timestamp = first_timestamp or timestamps[0].isoformat()
            last_timestamp = timestamps[-1].isoformat()

    total_rmse = torch.sqrt(total_squared_error_sum) / total_pixel_count
    total_mse = total_squared_error_sum / total_pixel_count
    total_mae = total_absolute_error_sum / total_pixel_count
    mse = squared_error_sum / pixel_count
    rmse = torch.sqrt(mse)
    mae = absolute_error_sum / pixel_count
    sum_squares = sum_squares / pixel_count
    r_square = 1 - mse / sum_squares
    evaluated_samples = evaluated_samples.detach().cpu().item()
    result = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "use_ema": has_ema,
        "data_set": data_set,
        "split_size": len(dataset),
        "sample_count": evaluated_samples,
        "sample_strategy": "evenly_spaced",
        "indices": indices,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "wavelengths": list(wavelengths),
        "mse_raw": [float(value) for value in mse.detach().cpu()],
        "rmse_raw": [float(value) for value in rmse.detach().cpu()],
        "mae_raw": [float(value) for value in mae.detach().cpu()],
        "r_square": [float(value) for value in r_square.detach().cpu()],
        "mean_mse_raw": float(mse.mean().detach().cpu()),
        "mean_rmse_raw": float(rmse.mean().detach().cpu()),
        "mean_mae_raw": float(mae.mean().detach().cpu()),
        "mean_r_square": float(r_square.mean().detach().cpu()),
        "total_rmse": float(total_rmse.detach().cpu()),
        "total_mse": float(total_mse.detach().cpu()),
        "total_mae": float(total_mae.detach().cpu()),
    }

    ema_suffix = "_ema" if has_ema else ""
    json_path = EVAL_DIR / f"{run_name}{ema_suffix}_{data_set}_mse_subset_{evaluated_samples:04d}.json"
    md_path = EVAL_DIR / f"{run_name}{ema_suffix}_{data_set}_mse_subset_{evaluated_samples:04d}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        f"# Solaris-S {data_set} subset raw-scale MSE",
        "",
        f"- Checkpoint: `{checkpoint_path.name}`",
        f"- EMA weights: `{has_ema}`",
        f"- Split size: {len(dataset)}",
        f"- Evaluated samples: {evaluated_samples}",
        f"- Sampling: evenly spaced indices across the split",
        f"- Timestamp range in evaluated batches: `{first_timestamp}` to `{last_timestamp}`",
        "",
        "| Wavelength (A) | MSE (raw intensity^2) | RMSE (raw intensity) | MAE (raw intensity) | R^2 (raw intensity) |",
        "|---:|---:|---:|---:|",
    ]
    for wavelength, mse_value, rmse_value, mae_value, r_square in zip(wavelengths, result["mse_raw"], result["rmse_raw"], result["mae_raw"], result["r_square"]):
        lines.append(f"| {wavelength} | {mse_value:.6g} | {rmse_value:.6g} | {mae_value:.6g} | {r_square:.6g} |")
    lines.extend(
        [
            f"| **Mean** | **{result['mean_mse_raw']:.6g}** | **{result['mean_rmse_raw']:.6g}** | **{result['mean_mae_raw']:.6g}** |**{result['mean_r_square']:.6g}** |",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    result["json_path"] = str(json_path)
    result["markdown_path"] = str(md_path)
    return result


def inspect_dataset(limit: int = 8) -> dict:
    import h5py

    files = sorted(DATA_DIR.glob("*.h5"))
    report = {"files": [path.name for path in files], "samples": [], "hour_counts": {}}
    for path in files[:2]:
        with h5py.File(path, "r") as handle:
            years = list(handle.keys())
            report.setdefault("top_keys", {})[path.name] = years[:limit]
            for year in years[:1]:
                for month in list(handle[year].keys())[:limit]:
                    for day in list(handle[year][month].keys())[:limit]:
                        hours = list(handle[year][month][day].keys())
                        report["hour_counts"][f"{year}-{month}-{day}"] = hours
                        for hour in hours[:limit]:
                            group = handle[year][month][day][hour]
                            wavelengths = list(group.keys())
                            attrs = {
                                wavelength: str(group[wavelength].attrs.get("exists", "missing"))
                                for wavelength in wavelengths[:limit]
                            }
                            report["samples"].append(
                                {
                                    "file": path.name,
                                    "path": f"{year}/{month}/{day}/{hour}",
                                    "wavelengths": wavelengths,
                                    "exists_attrs": attrs,
                                }
                            )
                            if len(report["samples"]) >= limit:
                                return report
    return report


def main_mode(
    mode: str = "train_missing_channel",
    steps: int = 7750,
    quick_check_seconds: int = 300,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    patch_size: int = DEFAULT_PATCH_SIZE,
    seed: int = 42,
    deterministic: bool = False,
    run_name: str = "missing_channel",
    data_set: str = "val",
    sample_index: int = 0,
    eval_samples: int = 64,
    eval_batch_size: int = 4,
    use_wandb: bool = False,
    wandb_project: str = "solaris",
    wandb_entity: str | None = None,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    ema_update_every: int = 10,
    checkpoint_every: int = 1000,
):
    if mode == "prepare":
        print(prepare_dataset.remote())
   
    
    elif mode == "train_missing_channel":
        print(
            train_missing_channel(
                max_steps=steps,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                patch_size=patch_size,
                seed=seed,
                deterministic=deterministic,
                run_name=run_name,
                resume=True,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                use_ema=use_ema,
                ema_decay=ema_decay,
                ema_update_every=ema_update_every,
                checkpoint_every=checkpoint_every,
            )
        )
    elif mode == "list":
        print("\n".join(list_checkpoints.remote()))
    elif mode == "inspect":
        print(inspect_dataset.remote())
    elif mode == "scales":
        print(get_scale_factors.remote())
    elif mode == "plot_missing_channel":
        print(plot_missing_channel(data_set=data_set, sample_index=sample_index, run_name=run_name, use_ema=use_ema))
    elif mode == "eval_mse_missing_channel":
        result = evaluate_mse_subset(
            data_set=data_set,
            max_samples=eval_samples,
            batch_size=eval_batch_size,
            run_name=run_name,
            use_ema=use_ema,
        )
        print(json.dumps(result, indent=2))
    else:
        raise ValueError(
            "mode must be one of: prepare, quick_check, train, train_detached, list, inspect, scales, plot, eval_mse"
        )

        
def main():
        

    parser = argparse.ArgumentParser()

    parser.add_argument( "--mode", type=str, default="eval_mse_missing_channel",)
    parser.add_argument( "--steps", type=int, default=10000,)
    parser.add_argument( "--quick-check-seconds", type=int, default=300,)
    parser.add_argument( "--batch-size", type=int, default=8,)
    parser.add_argument( "--grad-accum-steps", type=int, default=4,)
    parser.add_argument( "--patch-size", type=int, default=DEFAULT_PATCH_SIZE,)
    parser.add_argument( "--seed", type=int, default=42,)
    parser.add_argument( "--deterministic", action="store_true", default=False)
    parser.add_argument( "--run-name", type=str, default="missing_channel",)
    parser.add_argument( "--data-set", type=str, default="test_missing_channel",)
    parser.add_argument( "--sample-index", type=int, default=0,)
    parser.add_argument( "--eval-samples", type=int, default=64,)
    parser.add_argument( "--eval-batch-size", type=int, default=4,)
    parser.add_argument( "--use-wandb", action="store_true",)
    parser.add_argument( "--wandb-project", type=str, default="solaris",)
    parser.add_argument( "--wandb-entity", type=str, default=None,)
    parser.add_argument( "--use-ema", action="store_true", default=False)
    parser.add_argument( "--ema-decay", type=float, default=0.999,)
    parser.add_argument( "--ema-update-every", type=int, default=10,)
    parser.add_argument( "--checkpoint-every", type=int, default=1000,)

    args = parser.parse_args()


    main_mode(
        mode=args.mode,
        steps=args.steps,
        quick_check_seconds=args.quick_check_seconds,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        patch_size=args.patch_size,
        seed=args.seed,
        deterministic=args.deterministic,
        run_name=args.run_name,
        data_set=args.data_set,
        sample_index=args.sample_index,
        eval_samples=args.eval_samples,
        eval_batch_size=args.eval_batch_size,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_update_every=args.ema_update_every,
        checkpoint_every=args.checkpoint_every
    )

if __name__ == '__main__':
    main()

