import os
import pickle

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, MultiTaskLasso, Ridge
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

dataset_dir = os.path.join(os.environ["DATA_PATH"], "brainreader", "data")
latent_dataset_dir = os.path.join(
    os.environ["DATA_PATH"], "brainreader", "latent_vectors"
)
resp_mean = np.load(
    os.path.join(dataset_dir, str(1), "stats", "responses_mean.npy")
)
resp_std = np.load(
    os.path.join(dataset_dir, str(1), "stats", "responses_std.npy")
)


def get_dataset(dataset):
    file_names = np.array(
        [
            f_name
            for f_name in os.listdir(os.path.join(dataset_dir, "1", dataset))
            if f_name.endswith(".pkl") or f_name.endswith(".pickle")
        ]
    )

    X_list, y_list = [], []
    for f_name in file_names:
        with open(os.path.join(dataset_dir, "1", dataset, f_name), "rb") as f:
            data = pickle.load(f)
        response = data["resp"]

        with open(
            os.path.join(latent_dataset_dir, dataset, f_name), "rb"
        ) as f:
            latent_data = pickle.load(f)

        response = response / resp_std
        X_list.append(response.reshape(-1))
        y_list.append(latent_data.reshape(-1))
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def main():
    X_train, y_train = get_dataset("train")
    X_val, y_val = get_dataset("val")

    X, y = np.vstack((X_train, X_val)), np.vstack((y_train, y_val))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    regularization_params = [100, 1000, 10000]
    for _, reg_param in enumerate(regularization_params):
        clf = MultiTaskLasso(alpha=reg_param)
        (
            train_sizes,
            train_scores_nb,
            test_scores_nb,
            fit_times_nb,
            score_times_nb,
        ) = learning_curve(clf, X, y, train_sizes=[0.9], return_times=True)

        print(train_scores_nb)
        print(test_scores_nb)


if __name__ == "__main__":
    main()
