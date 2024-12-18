import json
import os
import pickle

import numpy as np
from sklearn.linear_model import Ridge
from utils import get_result_path

dataset_dir = os.path.join(os.environ["DATA_PATH"], "brainreader", "data")
latent_dataset_dir = os.path.join(
    os.environ["DATA_PATH"], "brainreader", "latent_vectors"
)


def get_resp_transform(session_id, subtract_mean=False):
    resp_mean = np.load(
        os.path.join(
            dataset_dir, str(session_id), "stats", "responses_mean.npy"
        )
    )
    resp_std = np.load(
        os.path.join(
            dataset_dir, str(session_id), "stats", "responses_std.npy"
        )
    )
    if subtract_mean:
        return lambda x: (x - resp_mean) / resp_std
    return lambda x: x / resp_std


def get_dataset(dataset="train", session_id=1):
    resp_transform = get_resp_transform(session_id)

    file_names = np.array(
        [
            f_name
            for f_name in os.listdir(
                os.path.join(dataset_dir, str(session_id), dataset)
            )
            if f_name.endswith(".pkl") or f_name.endswith(".pickle")
        ]
    )

    X_list, y_list = [], []
    for f_name in file_names:
        with open(
            os.path.join(dataset_dir, str(session_id), dataset, f_name), "rb"
        ) as f:
            data = pickle.load(f)

        response = data["resp"]
        if dataset == "test":
            response = response.mean(0)

        with open(
            os.path.join(latent_dataset_dir, str(session_id), dataset, f_name),
            "rb",
        ) as f:
            latent_data = pickle.load(f)

        response = resp_transform(response)

        X_list.append(response.reshape(-1))
        y_list.append(latent_data.reshape(-1))

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def mse_loss(X, y, clf):
    return ((clf.predict(X) - y) ** 2).mean()


def main():
    conf = {"model_name": "ridge_regression", "session_id": "6"}

    X_train, y_train = get_dataset("train", conf["session_id"])
    X_val, y_val = get_dataset("val", conf["session_id"])
    X_test, y_test = get_dataset("test", conf["session_id"])

    regularization_params = np.logspace(0, 5, 20)
    for _, reg_param in enumerate(regularization_params):
        conf["reg_param"] = reg_param
        results_path = get_result_path(conf)

        print(
            f"Running ridge regression with regularization parameter set to {reg_param}"
        )

        clf = Ridge(alpha=reg_param)
        clf.fit(X_train, y_train)

        print("Train vae:", mse_loss(X_train, y_train, clf))
        print("Val vae:", mse_loss(X_val, y_val, clf))

        predictions = clf.predict(X_test)
        np.save(os.path.join(results_path, "predictions.npy"), predictions)

        test_loss = mse_loss(X_test, y_test, clf)
        print("Test loss:", test_loss)

        with open(os.path.join(results_path, "test_loss.txt"), "w") as f:
            f.write(f"{test_loss}\n")

        with open(os.path.join(results_path, "setup.json"), "w") as f:
            json.dump(conf, f)


if __name__ == "__main__":
    main()
