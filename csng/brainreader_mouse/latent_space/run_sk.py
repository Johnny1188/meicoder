import os
import pickle

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

dataset_dir = os.path.join(os.environ["DATA_PATH"], "brainreader", "data")
embeddings_dataset_dir = os.path.join(
    os.environ["DATA_PATH"], "brainreader", "embeddings"
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

    X_list, y_vae_list, y_clip_list = [], [], []
    for f_name in file_names:
        with open(os.path.join(dataset_dir, "1", dataset, f_name), "rb") as f:
            data = pickle.load(f)
        response = data["resp"]

        with open(
            os.path.join(embeddings_dataset_dir, dataset, f_name), "rb"
        ) as f:
            data = pickle.load(f)
        latent_data, clip_data = data["latent"], data["embeddings"]

        response = response / resp_std
        X_list.append(response.reshape(-1))
        y_vae_list.append(latent_data.reshape(-1))
        y_clip_list.append(clip_data.reshape(-1))
    X = np.array(X_list)
    y_vae = np.array(y_vae_list)
    y_clip = np.array(y_clip_list)
    return X, y_vae, y_clip


def main():
    out_train_file_path = "output_train.pickle"
    out_vae_val_file_path = "output_vae_val.pickle"
    out_clip_val_file_path = "output_clip_val.pickle"

    X_train, y_vae_train, y_clip_train = get_dataset("train")
    X_val, y_vae_val, y_clip_val = get_dataset("val")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    print(y_vae_train.shape, X_val.shape)

    # pca = PCA(2000)
    # X_train = pca.fit_transform(X_train)
    # X_val = pca.transform(X_val)

    regularization_params = [
        # 0.000001,
        # 0.00001,
        # 0.0001,
        # 0.001,
        # 0.01,
        # 0.1,
        1,
        # 10,
        # 100,
        # 1000,
        # 10000,
    ]
    for idx, reg_param in enumerate(regularization_params):
        clf = Ridge(alpha=reg_param)
        clf.fit(X_train, y_vae_train)
        print(clf.coef_.shape)

        print("Train vae:", clf.score(X_train, y_vae_train))
        print("Val vae:", clf.score(X_val, y_vae_val))

        prediction = clf.predict(X_train[1:2])
        with open(out_vae_val_file_path, "wb") as f:
            pickle.dump(
                [prediction[0], y_vae_train[1]], f
            )  # Save the tensor as numpy array

        # clf = Ridge(alpha=reg_param)
        # clf.fit(X_train, y_clip_train)
        # print("Train clip:", clf.score(X_train, y_clip_train))
        # print("Val clip:", clf.score(X_val, y_clip_val))

        # prediction = clf.predict(X_train[:1])
        # with open(out_train_file_path, "wb") as f:
        #     pickle.dump(prediction[0], f)  # Save the tensor as numpy array

        # prediction = clf.predict(X_val[:1])
        # with open(out_clip_val_file_path, "wb") as f:
        #     pickle.dump(
        #         [prediction[0], y_clip_val[1]], f
        #     )  # Save the tensor as numpy array

        print(
            f"Saved model no.{idx+1} with a regularization parameter of {reg_param}."
        )


if __name__ == "__main__":
    main()
