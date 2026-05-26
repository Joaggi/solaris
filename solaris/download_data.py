import argparse
from datetime import datetime, timedelta
from pathlib import Path

YEARS = tuple(str(year) for year in range(2010, 2024))
DATASET_REPO_ID = "hrrsmjd/AIA_12hour_512x512"
FILE_PREFIX = "aia_12hour_512x512_"
PRETRAIN_WAVELENGTHS = ("0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600")


def _data_file(root_dir: str | Path, year: str) -> Path:
    from solaris.utils_data import resolve_data_root

    return resolve_data_root(root_dir) / f"{FILE_PREFIX}{year}.h5"


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


def download_year(year: str, output_dir: str | Path | None = None) -> Path:
    """Download a single year of processed Solaris data."""
    from huggingface_hub import hf_hub_download

    from solaris.utils_data import resolve_data_root

    target_dir = resolve_data_root(output_dir)
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename=f"{FILE_PREFIX}{year}.h5",
            local_dir=target_dir,
        )
    )


def check_data_exists(root_dir: str | Path, year: str = "2023") -> None:
    """Print the number of datasets flagged as present for a given year."""
    import h5py

    with h5py.File(_data_file(root_dir, year), "r") as file:
        count = 0
        for year_key in file.keys():
            for month_key in file[year_key].keys():
                for day_key in file[year_key][month_key].keys():
                    for hour_key in file[year_key][month_key][day_key].keys():
                        for wavelength_key in file[year_key][month_key][day_key][hour_key].keys():
                            dataset = file[year_key][month_key][day_key][hour_key][wavelength_key]
                            if "exists" in dataset.attrs and dataset.attrs["exists"]:
                                count += 1
        print(f"Total existing data points in {year}: {count}")


def get_valid_ids_for_downstream_task(root_dir: str | Path) -> list[list[str]]:
    """Return timestamps whose 12-hour target exists for the downstream task."""
    import h5py

    from solaris.utils_data import add_hours

    valid_ids = []

    for year in ("2019", "2020", "2021", "2022", "2023"):
        with h5py.File(_data_file(root_dir, year), "r") as file:
            for year_key in file.keys():
                for month_key in file[year_key].keys():
                    for day_key in file[year_key][month_key].keys():
                        for hour_key in file[year_key][month_key][day_key].keys():
                            timestamp = [year_key, month_key, day_key, hour_key]
                            future_timestamp = add_hours(timestamp, 12)
                            try:
                                with h5py.File(_data_file(root_dir, future_timestamp[0]), "r") as future_file:
                                    exists = future_file[future_timestamp[0]][future_timestamp[1]][
                                        future_timestamp[2]
                                    ][future_timestamp[3]]["1700"].attrs["exists"]
                                    if exists:
                                        valid_ids.append(timestamp)
                            except (OSError, KeyError):
                                continue

    return valid_ids


def save_downstream_ids(root_dir: str | Path, output_file: str | Path) -> None:
    """Persist valid downstream timestamps to disk."""
    valid_ids = get_valid_ids_for_downstream_task(root_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for id_list in valid_ids:
            file.write(" ".join(id_list) + "\n")
    print(f"Saved {len(valid_ids)} valid IDs to {output_path}")


def build_valid_pretrain_ids(root_dir: str | Path, years: tuple[str, ...] = YEARS) -> list[list[str]]:
    """Return timestamps with both history frames and the 12-hour target present."""
    import h5py

    from solaris.utils_data import resolve_data_root

    root = resolve_data_root(root_dir)
    years = tuple(sorted(str(year) for year in years))
    if not years:
        raise ValueError("At least one year is required.")

    handles: dict[int, h5py.File] = {}

    def handle_for(year: int):
        if str(year) not in years:
            return None
        if year not in handles:
            path = _data_file(root, str(year))
            if not path.exists():
                return None
            handles[year] = h5py.File(path, "r")
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
            for wavelength in PRETRAIN_WAVELENGTHS:
                if wavelength not in group or not _is_present(group[wavelength]):
                    return False
        return True

    valid_ids = []
    current = datetime(int(years[0]), 1, 1, 0)
    end = datetime(int(years[-1]), 12, 31, 12)
    while current <= end:
        if sample_is_valid(current):
            valid_ids.append(_timestamp_parts(current))
        current += timedelta(hours=12)

    for handle in handles.values():
        handle.close()

    return valid_ids


def save_pretrain_ids(root_dir: str | Path, output_dir: str | Path, years: tuple[str, ...] = YEARS) -> None:
    """Create chronological train/val/test split files for pretraining."""
    from solaris.utils_data import resolve_id_dir

    all_ids = build_valid_pretrain_ids(root_dir, years=years)

    train_split = int(0.8 * len(all_ids))
    val_split = int(0.9 * len(all_ids))
    splits = {
        "train": all_ids[:train_split],
        "val": all_ids[train_split:val_split],
        "test": all_ids[val_split:],
    }

    # Guard band of +/-24h at each boundary: samples are 12h apart, so drop 2 frames per side.
    splits["train"] = splits["train"][:-2] if len(splits["train"]) > 2 else splits["train"]
    splits["val"] = splits["val"][2:-2] if len(splits["val"]) > 4 else splits["val"]
    splits["test"] = splits["test"][2:] if len(splits["test"]) > 2 else splits["test"]

    output_root = resolve_id_dir(output_dir, data_root=root_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for split, ids in splits.items():
        with (output_root / f"{split}_id.txt").open("w", encoding="utf-8") as file:
            for id_list in ids:
                file.write(" ".join(id_list) + "\n")
        print(f"Saved {len(ids)} {split} IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download processed Solaris data and create pretraining splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/AIA_12hour_512x512"))
    parser.add_argument("--years", nargs="+", default=list(YEARS), help="Years to download/use, e.g. 2019 2020.")
    parser.add_argument("--skip-download", action="store_true", help="Only create split files from existing HDF5 data.")
    parser.add_argument("--make-pretrain-splits", action="store_true", help="Write train/val/test ID files.")
    args = parser.parse_args()

    years = tuple(sorted(str(year) for year in args.years))
    if not args.skip_download:
        for year in years:
            print(f"Downloading {year} to {args.data_dir}")
            download_year(year, args.data_dir)

    if args.make_pretrain_splits:
        save_pretrain_ids(args.data_dir, args.data_dir / "splits", years=years)


if __name__ == "__main__":
    main()
