import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
import dill
import lovely_tensors as lt
lt.monkey_patch()

from csng.data import get_dataloaders
from csng.utils.mix import seed_all


class InceptionV1FeatureExtractor(nn.Module):
    def __init__(self, layer="conv2.conv.weight"):
        super().__init__()

        #### load InceptionV1 (GoogleNet) model
        inception = models.googlenet(weights="IMAGENET1K_V1").eval()
        for p in inception.parameters():
            p.requires_grad = False

        ### define layers for feature extraction
        self.layer_dict = {
            "conv2.conv.weight": 3,
        }
        self.layer = layer
        self.features = nn.Sequential(*list(inception.children())[:self.layer_dict[self.layer]])

    def forward(self, x):
        for layer_i, layer in enumerate(self.features.children()):
            x = layer(x)
        return x


class SpatialEmbedding(nn.Module):
    def __init__(self, n_neurons=8587, feature_shape=(64, 56, 56)):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_neurons, *feature_shape[-2:]) * 0.1)  # spatial weights
        self.alpha = nn.Parameter(torch.randn(n_neurons, feature_shape[0]) * 0.1)  # feature weights

    def forward(self, resp, feature_map):
        B = resp.shape[0]

        ### compute retinal embeddings (B, H, W)
        W = self.W.expand(B, -1, -1, -1)  # (B, E, H, W)
        E = torch.einsum("be,bexy->bxy", resp, W)  # (B, E_H, E_W)

        ### compute estimated neural responses
        alpha = self.alpha.expand(B, -1, -1)  # (B, E, C)
        resp_est = torch.einsum("bec,bcxy,bexy->be", alpha, feature_map, W)  # (B, E)

        return E, resp_est


def compute_loss(
    resp,
    resp_est,
    weights,
    alpha,
    lambda_1=3e-4,
    lambda_2=3e-3,
    lambda_3=3e-3,
):
    resp_loss = (resp - resp_est).norm(2, dim=1).mean()
    l1_weights = weights.norm(1, dim=(-1, -2)).mean()
    l2_params = weights.norm(2, dim=(-1, -2)).mean() + alpha.norm(2, dim=-1).mean()
    laplacian_alpha = torch.sum(torch.abs(alpha[:, 1:] - alpha[:, :-1]))  # approximate Laplacian

    loss = resp_loss + lambda_1 * l1_weights + lambda_2 * l2_params + lambda_3 * laplacian_alpha

    return loss, {
        "resp_loss": resp_loss.item(),
        "l1_weights": l1_weights.item(),
        "l2_params": l2_params.item(),
        "laplacian_alpha": laplacian_alpha.item(),
    }


cfg = {
    "device": os.environ.get("DEVICE", "cpu"),
    "seed": 0,
    "data": {
        "mixing_strategy": "parallel_min", # needed only with multiple base dataloaders
        "max_training_batches": None,
    },
}

cfg["data"]["brainreader_mouse"] = {
    "device": cfg["device"],
    "mixing_strategy": cfg["data"]["mixing_strategy"],
    "max_batches": None,
    "data_dir": os.path.join((DATA_PATH_BRAINREADER := os.path.join(os.environ["DATA_PATH"], "brainreader")), "data"),
    "batch_size": 32,
    # "sessions": list(range(1, 23)),
    "sessions": [6],
    "resize_stim_to": (36, 64),
    "normalize_stim": True,
    "normalize_resp": True,
    "div_resp_by_std": True,
    "clamp_neg_resp": False,
    "additional_keys": None,
    "avg_test_resp": True,
}

cfg["model"] = {
    "feature_extractor": {
        "layer": "conv2.conv.weight",
    },
    "spatial_embedding": {
        "n_neurons": 8587,
        "feature_shape": (64, 36, 64),
    },
    "loss": {
        "lambda_1": 5e-1,
        "lambda_2": 1e-2,
        "lambda_3": 1e-2,
    },
    "optimizer": {
        "lr": 3e-3,
    },
    "epochs": 2000,
}


if __name__ == "__main__":
    print(f"... Running on {cfg['device']} ...")

    ### config
    cfg["run_name"] = datetime.now().strftime("%d-%m-%Y_%H-%M")
    cfg["save_dir"] = os.path.join(os.environ["DATA_PATH"], "monkeysee", "spatial_embedding", cfg["run_name"])
    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["save_dir"], "weights"), exist_ok=True)
    os.makedirs(os.path.join(cfg["save_dir"], "embeddings"), exist_ok=True)
    print(f"[INFO] Saving to {cfg['save_dir']}")

    ### initialize models
    seed_all(cfg["seed"])
    feature_extractor = InceptionV1FeatureExtractor(**cfg["model"]["feature_extractor"]).to(cfg["device"])
    embedding = SpatialEmbedding(**cfg["model"]["spatial_embedding"]).to(cfg["device"])
    opter = optim.Adam(embedding.parameters(), **cfg["model"]["optimizer"])

    ### data prep
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x), # Grayscale to RGB
        transforms.Resize((
            cfg["model"]["spatial_embedding"]["feature_shape"][-2] * 4,
            cfg["model"]["spatial_embedding"]["feature_shape"][-1] * 4
        )),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ### training
    history = {k: [] for k in ["resp_loss", "l1_weights", "l2_params", "laplacian_alpha"]}
    seed_all(cfg["seed"])
    for ep in range(cfg["model"]["epochs"]):
        embedding.train()
        dls, _ = get_dataloaders(config=cfg)
        train_dl, val_dl = dls["train"]["brainreader_mouse"], dls["val"]["brainreader_mouse"]
        print(
            f"[E {ep+1}/{cfg['model']['epochs']}  {datetime.now().strftime('%H:%M:%S')}]\n  "
            + '\n  '.join([f'{k}: {np.mean(v[-len(train_dl):]):.4f}' for k, v in history.items()] if ep > 0 else [])
        )

        for batch in tqdm(train_dl, total=len(train_dl)):
            ### prepare inputs
            resp = torch.cat([dp["resp"] for dp in batch], dim=0).to(cfg["device"])
            image = transform(torch.cat([dp["stim"] for dp in batch], dim=0)).to(cfg["device"])

            ### forward pass
            feature_map = feature_extractor(image)
            E, resp_est = embedding(resp=resp, feature_map=feature_map)
            loss, losses_all = compute_loss(
                resp=resp,
                resp_est=resp_est,
                weights=embedding.W,
                alpha=embedding.alpha,
                **cfg["model"]["loss"]
            )
            opter.zero_grad()
            loss.backward()
            opter.step()

            ### update history
            for k, v in losses_all.items():
                history[k].append(v)

        ### save model
        if ep > 0 and ep % 5 == 0:
            torch.save({
                "embedding": embedding.state_dict(),
                "optimizer": opter.state_dict(),
                "history": history,
                "config": cfg,
            }, os.path.join(cfg["save_dir"], f"embedding_latest.pt"), pickle_module=dill)

            if ep % 100 == 0:
                torch.save({
                    "embedding": embedding.state_dict(),
                    "optimizer": opter.state_dict(),
                    "history": history,
                    "config": cfg,
                }, os.path.join(cfg["save_dir"], f"embedding_{ep}.pt"), pickle_module=dill)

            ### plot weights
            plt.figure(figsize=(12, 8))
            for i in range(25):
                plt.subplot(5, 5, i+1)
                plt.imshow(embedding.W[i].detach().cpu().numpy(), cmap="gray")
                plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg["save_dir"], "weights", f"{ep}.png"))
            plt.close()

            ### plot embeddings
            plt.figure(figsize=(10, 7))
            for i in range(min(16, E.shape[0])):
                plt.subplot(4, 4, i+1)
                plt.imshow(E[i].detach().cpu().numpy(), cmap="gray")
                plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg["save_dir"], "embeddings", f"{ep}.png"))
            plt.close()

            ### plot losses
            plt.figure(figsize=(10, 7))
            for k, v in history.items():
                plt.plot(np.convolve(v, np.ones(100)/100, mode="valid"), label=k)
            plt.ylim(0, 800)
            plt.xlabel("Updates")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(cfg["save_dir"], "losses.png"))
            plt.close()