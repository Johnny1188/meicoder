import os

from data import PerSampleStoredDataset
from torch.utils.data import DataLoader


class LatentDataLoader:
    def __init__(
        self,
        dataset_dir,
        latent_dataset_dir,
        session_id,
        resp_transform,
        batch_size,
        device,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.latent_dataset_dir = latent_dataset_dir
        self.resp_transform = resp_transform(dataset_dir, session_id, device)
        self.batch_size = batch_size
        self.sess_id = session_id
        self.device = device

    def train_data(self):
        data_set = PerSampleStoredDataset(
            os.path.join(self.dataset_dir, str(self.sess_id), "train"),
            os.path.join(self.latent_dataset_dir, str(self.sess_id), "train"),
            resp_transform=self.resp_transform,
            device=self.device,
        )
        train_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return train_loader

    def valid_data(self):
        data_set = PerSampleStoredDataset(
            os.path.join(self.dataset_dir, str(self.sess_id), "val"),
            os.path.join(self.latent_dataset_dir, str(self.sess_id), "val"),
            resp_transform=self.resp_transform,
            device=self.device,
        )
        valid_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
        )
        return valid_loader

    def test_data(self):
        data_set = PerSampleStoredDataset(
            os.path.join(self.dataset_dir, str(self.sess_id), "test"),
            os.path.join(self.latent_dataset_dir, str(self.sess_id), "test"),
            resp_transform=self.resp_transform,
            device=self.device,
        )
        test_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
        )
        return test_loader
