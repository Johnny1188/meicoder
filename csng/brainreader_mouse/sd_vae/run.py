import os

import torch
from data_loader import LatentDataLoader
from models import get_model
from train import Trainer
from utils import get_resp_transform, get_result_path, load_json, save_results


def main():
    conf = load_json("setup.json")
    result_path = get_result_path(conf)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = os.path.join(os.environ["DATA_PATH"], "brainreader", "data")
    latent_dataset_dir = os.path.join(
        os.environ["DATA_PATH"], "brainreader", "latent_vectors"
    )

    epochs = conf["epochs"]
    batch_size = conf["batch_size"]
    session_id = conf["session_id"]
    data = LatentDataLoader(
        dataset_dir,
        latent_dataset_dir,
        session_id,
        get_resp_transform,
        batch_size,
        DEVICE,
    )

    inputs, _ = next(iter(data.train_data()))

    model = get_model(conf["model_name"], inputs.shape[1]).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"]
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        if conf["scheduler"]
        else None
    )
    criterion = torch.nn.MSELoss()
    trainer = Trainer(data, model, optimizer, criterion, scheduler, DEVICE)
    trainer.train(epochs)

    torch.save(model.state_dict(), f"trained.pth")
    print(f"Saved model.")

    save_results(trainer, conf, result_path)


if __name__ == "__main__":
    main()
