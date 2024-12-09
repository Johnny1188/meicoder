import os

import torch
from data_loader import LatentDataLoader
from models import get_model
from train import Trainer
from utils import get_resp_transform


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = os.path.join(os.environ["DATA_PATH"], "brainreader", "data")
    latent_dataset_dir = os.path.join(
        os.environ["DATA_PATH"], "brainreader", "embeddings"
    )

    epochs = 10
    batch_size = 256
    data = LatentDataLoader(
        dataset_dir,
        latent_dataset_dir,
        get_resp_transform(dataset_dir, DEVICE),
        batch_size,
        DEVICE,
    )

    model = get_model("deconv").to(DEVICE)
    # Test different values for lr (1e-1, 1e-2, 1e-3) and weight_decay (0.1, 1, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    criterion = torch.nn.MSELoss()
    trainer = Trainer(data, model, optimizer, criterion, scheduler, DEVICE)
    trainer.train(epochs)

    torch.save(model.state_dict(), f"trained.pth")
    print(f"Saved model.")


if __name__ == "__main__":
    main()
