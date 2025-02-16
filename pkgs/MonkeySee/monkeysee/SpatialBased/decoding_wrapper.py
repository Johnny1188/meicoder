import os
import torch
from torch import nn
import dill

from monkeysee.SpatialBased.train_spatial_torch import compute_mean_std, get_RFs, get_inputs
from monkeysee.SpatialBased.generator import Generator
from csng.utils.mix import update_config_paths, seed_all
from csng.data import get_dataloaders


class MonkeySeeDecoder(nn.Module):
    def __init__(self, ckpt_dir, train_dl, new_data_path=None):
        super().__init__()

        self.ckpt_dir = ckpt_dir
        ckpt = torch.load(os.path.join(ckpt_dir, "generator.pt"), pickle_module=dill)

        ### set config
        self.cfg = ckpt["config"]
        if new_data_path is not None:
            update_config_paths(self.cfg, new_data_path=new_data_path)
            self.cfg["rfs"]["spatial_embeddings_path"] = os.path.join(
                new_data_path, "monkeysee", "spatial_embedding",
                self.cfg["rfs"]["spatial_embeddings_path"].split("/")[-2],
                self.cfg["rfs"]["spatial_embeddings_path"].split("/")[-1],
            )
        print(
            f"[WARNING] MonkeySee decoder expects images {'z-scored' if self.cfg['data']['brainreader_mouse']['normalize_stim'] else 'in [0, 1] range'}"
            f" and responses {'z-scored (per-neuron)' if self.cfg['data']['brainreader_mouse']['normalize_resp'] else 'not z-scored'}."
        )

        ### load RFs
        seed_all(self.cfg["seed"])
        self.RFs = get_RFs(**self.cfg["rfs"])

        ### initialize model
        seed_all(self.cfg["seed"])
        self.generator = Generator(**self.cfg["decoder"]["gen"], device=self.cfg["device"])

        ### load checkpoints
        print(f"[INFO] Loading checkpoint from {self.cfg['decoder']['load_ckpt']} ...")
        self.generator.load_state_dict(ckpt["state_dict"])

        ### collect statistics
        self.transform_inputs = lambda x: x
        if self.cfg["decoder"]["standardize_inputs"]:
            print("[INFO] Collecting statistics ...")
            mean, std = compute_mean_std(dl=train_dl, config=self.cfg, RFs=self.RFs)
            print(f"  mean: {mean}\n  std: {std}")
            self.mean = mean.unsqueeze(-1).unsqueeze(-1)
            self.std = std.unsqueeze(-1).unsqueeze(-1)
            self.transform_inputs = lambda x: (x - self.mean) / (self.std + 1e-6)

    def forward(self, x, data_key, neuron_coords=None, pupil_center=None):
        assert data_key == "6", "MonkeySeeDecoder only supports data_key='6'."

        inputs = get_inputs(brains=x, config=self.cfg, transform_inputs_fn=self.transform_inputs, RFs=self.RFs)
        recons = self.generator(inputs, return_inv_ret_maps=False)

        return recons
