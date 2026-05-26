from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch

AIA_INPUT_WAVELENGTHS = ("0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600")
AIA_PRETRAIN_WAVELENGTHS = AIA_INPUT_WAVELENGTHS


def parse_custom_hour(hour_str: str) -> int:
    """Convert custom hour format ``H0000`` into an integer hour."""
    return int(hour_str[1:3])


def to_custom_hour(hour: int) -> str:
    """Convert an integer hour into the custom ``H0000`` format."""
    return f"H{hour:02d}00"


def add_hours(date_time_list: Iterable[str], hours_to_add: int) -> list[str]:
    """Offset a timestamp expressed as ``[year, month, day, hour]``."""
    year, month, day, hour_str = date_time_list
    hour = parse_custom_hour(hour_str)

    original_datetime = datetime(int(year), int(month), int(day), hour)
    new_datetime = original_datetime + timedelta(hours=hours_to_add)

    return [
        str(new_datetime.year),
        f"{new_datetime.month:02d}",
        f"{new_datetime.day:02d}",
        to_custom_hour(new_datetime.hour),
    ]


def timestamp_to_datetime(date_time_list: Sequence[str]) -> datetime:
    """Convert a timestamp expressed as ``[year, month, day, hour]`` to a datetime."""
    year, month, day, hour_str = date_time_list
    return datetime(int(year), int(month), int(day), parse_custom_hour(hour_str))


def resolve_data_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the directory containing Solaris HDF5 data files."""
    candidate = path if path not in (None, "", "CHANGE_PATH") else os.environ.get("SOLARIS_DATA_DIR")
    if not candidate:
        raise ValueError(
            "Data path is not configured. Pass an explicit path or set SOLARIS_DATA_DIR."
        )
    return Path(candidate).expanduser()


def resolve_id_dir(
    id_dir: str | os.PathLike[str] | None = None,
    *,
    data_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the directory containing train/val/test ID files."""
    candidate = id_dir if id_dir not in (None, "", "CHANGE_PATH") else os.environ.get("SOLARIS_ID_DIR")
    if candidate:
        return Path(candidate).expanduser()
    return resolve_data_root(data_root)


def read_id_file(path: str | os.PathLike[str]) -> list[list[str]]:
    """Read whitespace-separated timestamp IDs from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return [line.split() for line in file if line.strip()]


def resolve_year_h5(root_dir: str | os.PathLike[str], year: str | int) -> Path:
    """Resolve a year-specific HDF5 file, supporting original and Hugging Face names."""
    root_path = resolve_data_root(root_dir)
    direct = root_path / f"{year}.h5"
    if direct.exists():
        return direct

    matches = sorted(root_path.glob(f"*{year}.h5"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No HDF5 file found for year {year} under {root_path}.")


def load_wavelength_stack(
    root_dir: str | os.PathLike[str],
    timestamp: Iterable[str],
    wavelengths: Iterable[str] = AIA_INPUT_WAVELENGTHS,
) -> torch.Tensor:
    """Load a stack of wavelength channels for a single timestamp."""
    year, month, day, hour = timestamp
    with h5py.File(resolve_year_h5(root_dir, year), "r") as file:
        channels = [
            torch.from_numpy(
                np.asarray(file[year][month][day][hour][wavelength], dtype=np.float32)
            )[None, ...]
            for wavelength in wavelengths
        ]
    return torch.cat(channels, dim=0)


def load_target_channel(
    root_dir: str | os.PathLike[str],
    timestamp: Iterable[str],
    wavelength: str = "1700",
) -> torch.Tensor:
    """Load a single target wavelength channel for a timestamp."""
    year, month, day, hour = timestamp
    with h5py.File(resolve_year_h5(root_dir, year), "r") as file:
        return torch.from_numpy(
            np.asarray(file[year][month][day][hour][wavelength], dtype=np.float32)
        )[None, ...]


def load_target_stack(
    root_dir: str | os.PathLike[str],
    timestamp: Iterable[str],
    wavelengths: Iterable[str] = AIA_PRETRAIN_WAVELENGTHS,
) -> torch.Tensor:
    """Load target wavelength channels for a single timestamp."""
    return load_wavelength_stack(root_dir, timestamp, wavelengths)


def build_metadata(batch: torch.Tensor, timestamps: Sequence[datetime] | None = None):
    """Construct Solaris metadata from a batch of shape `(B, T, C, H, W)` or `(B, C, H, W)`."""
    if batch.dim() not in (4, 5):
        raise ValueError(
            "Expected a 4D `(B, C, H, W)` or 5D `(B, T, C, H, W)` batch, "
            f"got shape {tuple(batch.shape)}."
        )

    batch_size, height, width = batch.shape[0], batch.shape[-2], batch.shape[-1]
    if timestamps is None:
        timestamps = tuple(datetime(1970, 1, 1) for _ in range(batch_size))
    if len(timestamps) != batch_size:
        raise ValueError(f"Expected {batch_size} timestamps, got {len(timestamps)}.")

    return (
        torch.arange(height, device=batch.device, dtype=torch.float32),
        torch.arange(width, device=batch.device, dtype=torch.float32),
        tuple(timestamps),
    )
