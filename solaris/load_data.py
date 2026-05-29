import torch
from torch.utils.data import Dataset

from solaris.utils_data import (
    AIA_INPUT_WAVELENGTHS,
    AIA_PRETRAIN_WAVELENGTHS,
    add_hours,
    load_target_channel,
    load_target_stack,
    load_wavelength_stack,
    read_id_file,
    resolve_id_dir,
    timestamp_to_datetime,
)


class CustomDataset_downstream(Dataset):
    def __init__(self, root_dir, data_set="train", id_dir=None):
        self.root_dir = root_dir
        self.data_set = data_set
        self.id_dir = resolve_id_dir(id_dir, data_root=root_dir)
        self.ids = self._get_valid_ids()

    def _get_valid_ids(self):
        """Load valid data IDs from the downstream task ID file."""
        return read_id_file(self.id_dir / f"{self.data_set}_id_1700.txt")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Get input wavelengths and the 1700 channel 12 hours later."""
        current_timestamp = self.ids[idx]
        future_timestamp = add_hours(current_timestamp, 12)

        data = load_wavelength_stack(self.root_dir, current_timestamp, AIA_INPUT_WAVELENGTHS)
        target = load_target_channel(self.root_dir, future_timestamp, "1700")
        return data, target


class CustomDataset_pretrain(Dataset):
    def __init__(self, root_dir, data_set="train", id_dir=None):
        self.root_dir = root_dir
        self.data_set = data_set
        self.id_dir = resolve_id_dir(id_dir, data_root=root_dir)
        self.ids = self._get_valid_ids()

    def _get_valid_ids(self):
        """Load valid data IDs from the pretraining ID file."""
        return read_id_file(self.id_dir / f"{self.data_set}_id.txt")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Get two 12-hour-separated history states and the 12-hour forecast target."""
        current_timestamp = self.ids[idx]
        previous_timestamp = add_hours(current_timestamp, -12)
        future_timestamp = add_hours(current_timestamp, 12)

        previous_data = load_wavelength_stack(self.root_dir, previous_timestamp, AIA_INPUT_WAVELENGTHS)
        current_data = load_wavelength_stack(self.root_dir, current_timestamp, AIA_INPUT_WAVELENGTHS)
        data = torch.stack((previous_data, current_data), dim=0)
        target = load_target_stack(self.root_dir, future_timestamp, AIA_PRETRAIN_WAVELENGTHS)

        return data, target, timestamp_to_datetime(current_timestamp)



class CustomDataset_missing_channel(Dataset):
    def __init__(self, root_dir, data_set="train", id_dir=None):
        self.root_dir = root_dir
        self.data_set = data_set
        self.id_dir = resolve_id_dir(id_dir, data_root=root_dir)
        self.ids = self._get_valid_ids()

    def _get_valid_ids(self):
        """Load valid data IDs from the pretraining ID file."""
        return read_id_file(self.id_dir / f"{self.data_set}_id.txt")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Get two 12-hour-separated history states and the 12-hour forecast target."""
        corrupted_current_timestamp = self.ids[idx]
        target_current_timestamp = self.ids[idx]

        current_data = load_wavelength_stack(self.root_dir, corrupted_current_timestamp, AIA_INPUT_WAVELENGTHS)
        data = torch.unsqueeze(current_data, dim=0)
        target = load_target_stack(self.root_dir, target_current_timestamp, AIA_PRETRAIN_WAVELENGTHS)

        random_missing_channel = torch.randint(len(AIA_INPUT_WAVELENGTHS), (1,)).item()
        data[:, random_missing_channel, :, :] = 0

        return data, target, timestamp_to_datetime(corrupted_current_timestamp)

