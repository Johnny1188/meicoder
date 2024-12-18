import datetime
import json
import os

import numpy as np
import torch
import torchvision

from csng.utils.data import Normalize, NumpyToTensor


def get_result_path(setup):
    # create result folder
    run_path = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # root_result_path = './root_result_test/'
    root_result_path = os.path.join("results", setup["model_name"], run_path)
    if not os.path.exists(root_result_path):
        os.makedirs(root_result_path)
    return root_result_path


def save_results(trainer, setup, path):
    train_curve = trainer.get_train_curve()
    val_curve = trainer.get_valid_curve()

    test_loss = trainer.best_test_loss

    # Save curves as .npy files
    np.save(os.path.join(path, "train_curve.npy"), train_curve)
    np.save(os.path.join(path, "val_curve.npy"), val_curve)

    # Save test loss as a text file
    with open(os.path.join(path, "test_loss.txt"), "w") as f:
        f.write(f"{test_loss}\n")

    # Save setup as a JSON file (ensure setup is JSON serializable)
    with open(os.path.join(path, "setup.json"), "w") as f:
        json.dump(setup, f)

    predictions = trainer.get_outputs()

    predictions_np = predictions.cpu().numpy()
    np.save(os.path.join(path, "predictions.npy"), predictions_np)

    print(f"Results with test loss {test_loss:.4f} saved to {path}")


def save_outputs(trainer, setup, path):
    train_curve = trainer.get_train_curve()
    val_curve = trainer.get_valid_curve()

    test_loss = trainer.eval_results()

    # Save curves as .npy files
    np.save(os.path.join(path, "train_curve.npy"), train_curve)
    np.save(os.path.join(path, "val_curve.npy"), val_curve)

    # Save test loss as a text file
    with open(os.path.join(path, "test_loss.txt"), "w") as f:
        f.write(f"{test_loss}\n")

    # Save setup as a JSON file (ensure setup is JSON serializable)
    with open(os.path.join(path, "setup.json"), "w") as f:
        json.dump(setup, f)


def save_json(data, name):
    with open(name, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path):
    if os.path.exists(path) and path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
            return data
    else:
        print("Invalid path")


def get_stim_transform(resize=64, device="cpu"):
    stim_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((36, 64)),
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(
                lambda x: x.to(device, dtype=torch.float16)
            ),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
        ]
    )
    return stim_transform


def get_resp_transform(dataset_dir="train", session_id=1, device="cpu"):
    resp_mean = torch.from_numpy(
        np.load(
            os.path.join(
                dataset_dir, str(session_id), "stats", "responses_mean.npy"
            )
        )
    ).to(device)
    resp_std = torch.from_numpy(
        np.load(
            os.path.join(
                dataset_dir, str(session_id), "stats", "responses_std.npy"
            )
        )
    ).to(device)
    resp_transform = torchvision.transforms.Compose(
        [
            NumpyToTensor(device=device),
            Normalize(mean=resp_mean, std=resp_std),
        ]
    )
    return resp_transform
