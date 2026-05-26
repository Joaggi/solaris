import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from solaris.load_data import CustomDataset_pretrain
from solaris.model.solaris import SolarisSmall
from solaris.normalization import transform
from solaris.utils_data import build_metadata

WAVELENGTHS = ("0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600")

PRETRAIN_SCALE_FACTORS = torch.tensor(
    [
        59.55241278048226,
        217.18662860219484,
        1619.8840310287178,
        2573.643248589347,
        1191.8919143907103,
        889.3599319561527,
        113.23721481062092,
        267.50519244567655,
    ],
    dtype=torch.float32,
)


def _seed_everything(seed: int, deterministic: bool = False) -> None:
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
    """Calculate the paper's weighted MAE on the normalised intensity scale."""
    target_normalised = _normalise_raw(target_raw, scale_factors)
    return torch.abs(prediction_normalised - target_normalised).mean()


def _raw_rmse(prediction_normalised, target_raw, scale_factors):
    """Calculate per-channel RMSE after unscaling predictions to original intensities."""
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


def collate_pretrain(batch):
    data, target, timestamps = zip(*batch)
    return default_collate(data), default_collate(target), tuple(timestamps)


def make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


class CpuEMA:
    def __init__(self, model, norm_coeff_1, norm_coeff_2, decay: float = 0.999, update_every: int = 10):
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
    ema: "CpuEMA | None" = None,
) -> None:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=7750)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--id_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_every', type=int, default=500)
    parser.add_argument('--run_name', type=str, default='local_pretrain')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--ema_update_every', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='solaris')
    parser.add_argument('--wandb_entity', type=str, default=None)

    args = parser.parse_args()

    _seed_everything(args.seed, deterministic=args.deterministic)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_dataset = CustomDataset_pretrain(root_dir=args.data_path, data_set="train", id_dir=args.id_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_pretrain,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    model = SolarisSmall(out_levels=len(PRETRAIN_SCALE_FACTORS), patch_size=args.patch_size).to(device)

    norm_coeff_1 = torch.nn.Parameter(torch.tensor(0.5, device=device))
    norm_coeff_2 = torch.nn.Parameter(torch.tensor(0.5, device=device))
    optimizer = AdamW(
        _param_groups(model, [norm_coeff_1, norm_coeff_2], args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    all_optim_params = [p for g in optimizer.param_groups for p in g["params"]]
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        args.warmup_steps,
        args.max_steps,
        args.min_lr / args.lr,
    )

    scale_factor_values = PRETRAIN_SCALE_FACTORS.tolist()
    print(f"scale_factors={scale_factor_values}", flush=True)
    scale_factors = PRETRAIN_SCALE_FACTORS.to(device=device, dtype=torch.float32)

    ema = (
        CpuEMA(model, norm_coeff_1, norm_coeff_2, decay=args.ema_decay, update_every=args.ema_update_every)
        if args.use_ema
        else None
    )

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_config = {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "patch_size": args.patch_size,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "betas": (0.9, 0.95),
            "checkpoint_every": args.checkpoint_every,
            "use_ema": args.use_ema,
            "ema_decay": args.ema_decay,
            "ema_update_every": args.ema_update_every,
            "wavelengths": WAVELENGTHS,
            "scale_factors": scale_factor_values,
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            id=args.run_name,
            resume="allow",
            config=wandb_config,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / f"{args.run_name}_latest.pt"

    global_step = 0
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        norm_coeff_1.data.copy_(checkpoint["norm_coeff_1"].to(device=device))
        norm_coeff_2.data.copy_(checkpoint["norm_coeff_2"].to(device=device))
        if ema is not None and "ema_model_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint)
        global_step = int(checkpoint["step"])
        print(f"resumed_checkpoint={latest_path} step={global_step}", flush=True)
        if checkpoint.get("seed") not in (None, args.seed):
            print(
                f"warning=checkpoint_seed_mismatch checkpoint_seed={checkpoint.get('seed')} requested_seed={args.seed}",
                flush=True,
            )

    start_time = time.monotonic()
    last_loss = float("nan")
    optimizer.zero_grad(set_to_none=True)

    model.train()

    while global_step < args.max_steps:
        batch_idx = -1
        for batch_idx, (data, target, timestamps) in enumerate(train_loader):
            data = data.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            data = transform(data, norm_coeff_1, norm_coeff_2, scale_factors)
            metadata = build_metadata(data, timestamps)

            if device.type == 'cuda':
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(data, metadata, 12, 0).squeeze(1)
            else:
                output = model(data, metadata, 12, 0).squeeze(1)
            loss = _paper_weighted_mae(output.float(), target.float(), scale_factors)
            loss = loss / args.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(all_optim_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_loss = float(loss.detach().cpu()) * args.grad_accum_steps
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

            if global_step % args.checkpoint_every == 0 or global_step >= args.max_steps:
                ckpt_path = checkpoint_dir / f"{args.run_name}_step_{global_step:05d}.pt"
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
                    args.patch_size,
                    args.seed,
                    ema,
                )
                _save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    scheduler,
                    norm_coeff_1,
                    norm_coeff_2,
                    global_step,
                    last_loss,
                    scale_factor_values,
                    args.patch_size,
                    args.seed,
                    ema,
                )
                print(f"saved_checkpoint={ckpt_path}", flush=True)

            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            break
        if batch_idx >= 0 and (batch_idx + 1) % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(all_optim_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_loss = float(loss.detach().cpu()) * args.grad_accum_steps
            if ema is not None:
                ema.update(model, norm_coeff_1, norm_coeff_2, global_step)

    final_path = checkpoint_dir / f"{args.run_name}_final_step_{global_step:05d}.pt"
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
        args.patch_size,
        args.seed,
        ema,
    )
    print(f"saved_checkpoint={final_path}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
