import os
import pickle
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from nnfabrik.builder import get_data
from collections import namedtuple
import csng

# from csng.data import MixedBatchLoader


def get_mouse_v1_data(config):
    ### get dataloaders
    _dataloaders = get_data(config["mouse_v1"]["dataset_fn"], config["mouse_v1"]["dataset_config"])

    if config["mouse_v1"]["average_test_multitrial"]:
        _dataloaders["test"] = average_test_multitrial(_dataloaders["test"], config)

    dataloaders = {
        "mouse_v1": {
            "train": MixedBatchLoader(
                dataloaders=[_dataloaders["train"][data_key] for data_key in _dataloaders["train"].keys()],
                neuron_coords=None,  # added below
                mixing_strategy=config["mixing_strategy"],
                data_keys=list(_dataloaders["train"].keys()),
                return_data_key=True,
                return_pupil_center=True,
                device=config["mouse_v1"]["device"],
            ) if config["mouse_v1"]["skip_train"] is False else MixedBatchLoader(
                dataloaders=[],
                neuron_coords=None,
                mixing_strategy=config["mixing_strategy"],
                data_keys=[],
                return_data_key=True,
                return_pupil_center=True,
                device=config["mouse_v1"]["device"],
            ),
            "val": MixedBatchLoader(
                dataloaders=[_dataloaders["validation"][data_key] for data_key in _dataloaders["validation"].keys()],
                neuron_coords=None,  # added below
                mixing_strategy=config["mixing_strategy"],
                data_keys=list(_dataloaders["train"].keys()),
                return_data_key=True,
                return_pupil_center=True,
                device=config["mouse_v1"]["device"],
            ) if config["mouse_v1"]["skip_val"] is False else MixedBatchLoader(
                dataloaders=[],
                neuron_coords=None,
                mixing_strategy=config["mixing_strategy"],
                data_keys=[],
                return_data_key=True,
                return_pupil_center=True,
                device=config["mouse_v1"]["device"],
            ),
            "test": MixedBatchLoader(
                dataloaders=[_dataloaders["test"][data_key] for data_key in _dataloaders["test"].keys()],
                neuron_coords=None,  # added below
                mixing_strategy=config["mixing_strategy"],
                data_keys=list(_dataloaders["train"].keys()),
                return_data_key=True,
                return_pupil_center=True,
                device=config["mouse_v1"]["device"],
            ) if config["mouse_v1"]["skip_test"] is False else MixedBatchLoader(
                dataloaders=[],
                neuron_coords=None,
                mixing_strategy=config["mixing_strategy"],
                data_keys=[],
                return_data_key=True,
                return_pupil_center=True,
                device=config["mouse_v1"]["device"],
            ),
            "test_no_resp": MixedBatchLoader(
                dataloaders=[_dataloaders["final_test"][data_key] for data_key in _dataloaders["final_test"].keys()],
                neuron_coords=None,  # added below
                mixing_strategy=config["mixing_strategy"],
                data_keys=list(_dataloaders["train"].keys()),
                return_data_key=True,
                return_pupil_center=True,
                device=config["mouse_v1"]["device"],
            )
        }
    }
    
    ### get cell coordinates
    neuron_coords = {
        data_key: torch.tensor(d.neurons.cell_motor_coordinates, dtype=torch.float32, device=config["mouse_v1"]["device"])
        for data_key, d in zip(list(_dataloaders["train"].keys()), [_dl.dataset for _dl in _dataloaders["train"].values()])
    }
    if config["mouse_v1"]["normalize_neuron_coords"]: # normalize coordinates to [-1, 1]
        for data_key in neuron_coords.keys():
            ### normalize x,y,z separately
            for dim_idx in range(neuron_coords[data_key].shape[-1]):
                neuron_coords[data_key][:, dim_idx] = \
                    (neuron_coords[data_key][:, dim_idx] - neuron_coords[data_key][:, dim_idx].min()) \
                    / (neuron_coords[data_key][:, dim_idx].max() - neuron_coords[data_key][:, dim_idx].min()) * 2 - 1

    ### assign neuron_coords to dataloaders
    for dl_type in ["train", "val", "test", "test_no_resp"]:
        dataloaders["mouse_v1"][dl_type].neuron_coords = neuron_coords

    return dataloaders, neuron_coords


def average_test_multitrial(dataloaders, config):
    for data_key, dataloader in dataloaders.items():
        ### check if the averaged test dataset has been already created and saved
        if os.path.exists(os.path.join(dataloader.dataset.dirname, "test_averaged.pkl")):
            # print(f"[INFO] Loading averaged test dataset for {data_key}...")
            with open(os.path.join(dataloader.dataset.dirname, "test_averaged.pkl"), "rb") as f:
                averaged_test_dataset = pickle.load(f)
            dataloaders[data_key] = DataLoader(
                SamplesDataset(
                    stims=averaged_test_dataset["stims"],
                    resps=averaged_test_dataset["resps"],
                    pupil_centers=averaged_test_dataset["pupil_centers"],
                    stim_transform=lambda x: x,
                    resp_transform=lambda x: x,
                ),
                batch_size=config["mouse_v1"].get("test_batch_size", config["mouse_v1"]["dataset_config"]["batch_size"]),
                shuffle=False,
            )
            continue
        else:
            print(f"[INFO] Averaging test dataset for {data_key}...")
            ### get the responses, stimuli, pupil centers and image ids from the test set
            stims, resps, p_centers, img_ids = get_multitrial_info(dataloader.dataset, tier="test")
            repeats_stims = split_on_idx(stims, img_ids)
            repeats_resps = split_on_idx(resps, img_ids)
            repeats_p_centers = split_on_idx(p_centers, img_ids)
            
            ### take a single stim from each multi-trials (they are the same),
            ### average the responses and pupil centers
            for sample_idx in range(len(repeats_stims)):
                for i in range(1, len(repeats_stims[sample_idx]) - 1):
                    assert np.array_equal(repeats_stims[sample_idx][i], repeats_stims[sample_idx][i + 1]), \
                        "Stimuli are not the same across repeats"
            stims = np.stack([repeats_stims[i][0] for i in range(len(repeats_stims))])
            stims = stims[:, np.newaxis, ...]
            resps = np.stack([repeats_resps[i].mean(0) for i in range(len(repeats_resps))])
            p_centers = np.stack([repeats_p_centers[i].mean(0) for i in range(len(repeats_p_centers))])
            dataloaders[data_key] = DataLoader(
                SamplesDataset(
                    stims=stims,
                    resps=resps,
                    pupil_centers=p_centers,
                    stim_transform=lambda x: x,
                    resp_transform=lambda x: x,
                ),
                batch_size=config["mouse_v1"]["dataset_config"]["batch_size"],
                shuffle=False,
            )

            if config["mouse_v1"]["save_test_multitrial"]:
                print(f"[INFO] Saving averaged test dataset for {data_key}...")
                with open(os.path.join(dataloader.dataset.dirname, "test_averaged.pkl"), "wb") as f:
                    pickle.dump({
                        "stims": stims,
                        "resps": resps,
                        "pupil_centers": p_centers,
                    }, f)

    return dataloaders


def split_on_idx(x, ids):
    per_image_repeats = []
    for image_id in np.unique(ids):
        responses_across_repeats = x[ids == image_id]
        per_image_repeats.append(responses_across_repeats)

    return per_image_repeats


def get_multitrial_info(dataset, tier="test"):
    tiers = dataset.trial_info.tiers
    complete_image_ids = dataset.trial_info.frame_image_id

    responses, stimuli, pupil_centers, image_ids = [], [], [], []
    for i, datapoint in enumerate(dataset):
        if tiers[i] != tier:
            continue

        responses.append(datapoint.responses.cpu().numpy().squeeze())
        stimuli.append(datapoint.images.cpu().numpy().squeeze())
        pupil_centers.append(datapoint.pupil_center.cpu().numpy().squeeze())
        image_ids.append(complete_image_ids[i])

    responses = np.stack(responses)
    stimuli = np.stack(stimuli)
    pupil_centers = np.stack(pupil_centers)

    return stimuli, responses, pupil_centers, image_ids


def append_syn_dataloaders(dataloaders, config):
    DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")
    for data_key in config["data_keys"]:
        ### divide by the per neuron std if the std is greater than 1% of the mean std (to avoid division by 0)
        resp_mean = torch.from_numpy(np.load(os.path.join(DATA_PATH, config["dir_name"], data_key, f"responses_mean_original.npy"))).float()
        resp_std = torch.from_numpy(np.load(os.path.join(DATA_PATH, config["dir_name"], data_key, f"responses_std_original.npy"))).float()
        div_by = resp_std.clone()
        thres = 0.01 * resp_std.mean()
        idx = resp_std <= thres
        div_by[idx] = thres

        data_key_to_add = data_key \
            if (config["data_key_prefix"] is None or config["data_key_prefix"] == "") \
            else f"{config['data_key_prefix']}_{data_key}"
        neuron_coords = {
            data_key_to_add: torch.from_numpy(np.load(
                os.path.join(DATA_PATH, config["dir_name"], data_key, f"neuron_coords.npy")
            )).float()
        }

        for data_part in config["append_data_parts"]:
            dataloader = DataLoader(
                PerSampleStoredDataset(
                    dataset_dir=os.path.join(DATA_PATH, config["dir_name"], data_key, data_part),
                    stim_transform=lambda x: x,
                    # resp_transform=lambda x: x,
                    resp_transform=csng.utils.Normalize(
                        # mean=0,
                        mean=resp_mean,
                        std=div_by,
                        center_data=False, # keep the same mean, just scale
                        clip_min=None,
                        clip_max=None,
                    ),
                    device=config.get("device", "cpu"),
                ),
                batch_size=config["batch_size"],
                shuffle=False,
            )
            dataloaders["mouse_v1"][data_part].add_dataloader(
                dataloader,
                neuron_coords=neuron_coords,
                data_key=data_key_to_add,
            )
    
    return dataloaders


def append_data_aug_dataloaders(dataloaders, config):
    for data_transform in config["data_transforms"]:
        for data_part in config["append_data_parts"]:
            ### copy all base dataloaders
            curr_dl = dataloaders["mouse_v1"][data_part]
            new_dls = deepcopy(curr_dl.dataloaders)
            new_neuron_coords = deepcopy(curr_dl.neuron_coords)
            new_data_keys = deepcopy(curr_dl.data_keys)
            assert len(new_dls) == len(new_data_keys)

            if config["force_same_order"]:
                ### set the same generator state for samplers in the old and new dataloaders
                for old_dl, new_dl in zip(curr_dl.dataloaders, new_dls):
                    old_dl.sampler.generator = torch.Generator().manual_seed(config["seed"])
                    new_dl.sampler.generator = torch.Generator().set_state(old_dl.sampler.generator.get_state())

            ### wrap base dataloaders with dataloaders applying the transforms
            for dl_i, (dl, data_key) in enumerate(zip(new_dls, new_data_keys)):
                new_dl = DataloaderWrapper(
                    dataloader=dl,
                    transform=data_transform[data_key] if type(data_transform) == dict else data_transform[dl_i],
                )
                curr_dl.add_dataloader(
                    new_dl,
                    neuron_coords={data_key: new_neuron_coords[data_key]},
                    data_key=data_key,
                )

    return dataloaders


class DataloaderWrapper:
    def __init__(self, dataloader, transform):
        self.dataloader = dataloader
        self.transform = transform

    @property
    def dataset(self):
        return self.dataloader.dataset

    def __iter__(self):
        for data in self.dataloader:
            yield self.transform(data)

    def __len__(self):
        return len(self.dataloader)


class RespGaussianNoise:
    def __init__(self, noise_std, clip_min=None, dynamic_mul_factor=0., resp_fn="identity"):
        self.noise_std = noise_std
        self.clip_min = clip_min
        self.dynamic_mul_factor = dynamic_mul_factor  # higher response of neuron => higher std noise
        self.resp_fn = resp_fn  # "identity", "squared", "shifted_exp", "log", "shifted_exp_sqrt"

    def _transform_resp(self, resp):
        if self.resp_fn == "identity":
            return resp
        elif self.resp_fn == "squared":
            return (resp ** 2) / 5
        elif self.resp_fn == "shifted_exp":
            return torch.exp(resp) - 1
        elif self.resp_fn == "shifted_exp_sqrt":
            return torch.exp(torch.sqrt(resp)) - 1
        elif self.resp_fn == "log":
            return torch.log(resp)
        else:
            raise NotImplementedError

    def __call__(self, data):
        noise_std = self.noise_std
        if self.dynamic_mul_factor > 0:
            noise_std = noise_std.unsqueeze(0).repeat(data.responses.size(0), 1)
            noise_std += self.dynamic_mul_factor * self._transform_resp(data.responses)
        noise = torch.randn_like(data.responses) * noise_std

        return namedtuple("Datapoint", ["images", "responses", "pupil_center"])(
            data.images,
            data.responses + noise \
             if self.clip_min is None \
             else torch.clamp(data.responses + noise, min=self.clip_min),
            data.pupil_center,
        )


class MixedBatchLoader:
    """
    A dataloader that interleaves or mixes batches from multiple dataloaders.

    Args:
        dataloaders (list): A list of dataloaders to interleave or mix batches from.
        neuron_coords (dict): A dictionary containing the neuron coordinates for each dataloader (data_key as the key).
        mixing_strategy (str): The strategy to use for mixing batches. Must be one of ["sequential", "parallel_min", "parallel_max"].
        device (str): The device to load the batches onto. Default is "cpu".

    Raises:
        AssertionError: If mixing_strategy is not one of ["sequential", "parallel_min", "parallel_max"].

    Attributes:
        dataloader_iters (list): A list of iterators for each dataloader.
        neuron_coords (dict): A dictionary containing the neuron coordinates for each dataloader (data_key as the key).
        n_dataloaders (int): The number of dataloaders.
        mixing_strategy (str): The strategy used for mixing batches.
        device (str): The device to load the batches onto.
        batch_idx (int): The current batch index.
        n_batches (int): The total number of batches.

    Methods:
        _get_sequential(): Interleaves batches from multiple dataloaders.
        _get_parallel(): Mixes batches from multiple dataloaders.
        __len__(): Returns the total number of batches.
        __iter__(): Returns the iterator object.
        __next__(): Returns the next batch of data.

    """
    def __init__(
        self,
        dataloaders,
        neuron_coords=None,
        mixing_strategy="sequential",
        data_keys=None,
        return_data_key=True,
        return_pupil_center=True,
        return_neuron_coords=True,
        device="cpu",
    ):
        assert mixing_strategy in ["sequential", "parallel_min", "parallel_max"], \
            f"mixing_strategy must be one of ['sequential', 'parallel_min', 'parallel_max'], but got {mixing_strategy}"

        self.dataloaders = dataloaders
        self.neuron_coords = neuron_coords
        self.data_keys = data_keys
        self.dataloader_iters = {dl_idx: {"dl": iter(dataloader)} for dl_idx, dataloader in enumerate(dataloaders)}
        if data_keys is not None:
            assert len(data_keys) == len(dataloaders), f"len(data_keys) must be equal to len(dataloaders), but got {len(data_keys)} and {len(dataloaders)}"
            for dl_idx, data_key in zip(self.dataloader_iters.keys(), data_keys):
                self.dataloader_iters[dl_idx]["data_key"] = data_key
        self.dataloaders_left = list(self.dataloader_iters.keys())
        self.n_dataloaders = len(self.dataloader_iters)
        self.mixing_strategy = mixing_strategy
        
        self.return_data_key = return_data_key
        self.return_pupil_center = return_pupil_center
        self.return_neuron_coords = return_neuron_coords
        
        self.device = device
        self.batch_idx = 0
        
        if self.mixing_strategy == "sequential":
            self.n_batches = sum([len(dataloader) for dataloader in dataloaders]) if len(dataloaders) > 0 else 0
        elif self.mixing_strategy == "parallel_max":
            self.n_batches = max([len(dataloader) for dataloader in dataloaders]) if len(dataloaders) > 0 else 0
        elif self.mixing_strategy == "parallel_min":
            self.n_batches = min([len(dataloader) for dataloader in dataloaders]) if len(dataloaders) > 0 else 0

        self.datasets = []
        for dl in dataloaders:
            if hasattr(dl, "dataset"):
                self.datasets.append(dl.dataset)
            else:
                self.datasets.append(dl)

    def add_dataloader(self, dataloader, neuron_coords=None, data_key=None):
        """
        Adds a dataloader to the list of dataloaders.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to add.
            data_key (str): The key to use for the dataloader. If None, the dataloader will be added with a numeric key.
        """
        self.dataloaders.append(dataloader)
        dl_idx = len(self.dataloaders)
        self.dataloader_iters[dl_idx] = {"dl": iter(dataloader)}
        if data_key is not None:
            self.dataloader_iters[dl_idx]["data_key"] = data_key
            if type(self.data_keys) == list:
                self.data_keys.append(data_key)
        self.dataloaders_left.append(dl_idx)
        self.n_dataloaders += 1
        if self.mixing_strategy == "sequential":
            self.n_batches += len(dataloader)
        elif self.mixing_strategy == "parallel_max":
            self.n_batches = max(self.n_batches, len(dataloader))
        elif self.mixing_strategy == "parallel_min":
            self.n_batches = min(self.n_batches, len(dataloader)) \
                if self.n_batches > 0 else len(dataloader)
        if hasattr(dataloader, "dataset"):
            self.datasets.append(dataloader.dataset)
        else:
            self.datasets.append(dataloader)

        if neuron_coords is not None:
            for _data_key, coords in neuron_coords.items():
                if _data_key in self.neuron_coords.keys() \
                    and not torch.equal(self.neuron_coords[_data_key], coords.to(self.device)):
                    print(f"[WARNING]: neuron_coords for data_key {_data_key} already exists and are not the same. Overwriting.")
                self.neuron_coords[_data_key] = coords.to(self.device)

    def _get_sequential(self):
        """
        Interleaves batches from multiple dataloaders.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        to_return = []
        while True:
            dl_idx = self.dataloaders_left[self.batch_idx % self.n_dataloaders]
            try:
                datapoint = next(self.dataloader_iters[dl_idx]["dl"])
                stim, resp = datapoint.images, datapoint.responses
                # to_return[self.dataloader_iters[dl_idx]["data_key"]] = [stim.to(self.device), resp.to(self.device)]
                to_return.append([self.dataloader_iters[dl_idx]["data_key"], stim.to(self.device), resp.to(self.device)])
                if self.return_neuron_coords:
                    _neuron_coords = self.neuron_coords[self.dataloader_iters[dl_idx]["data_key"]]
                    # to_return[self.dataloader_iters[dl_idx]["data_key"]].append(_neuron_coords.to(self.device))
                    to_return[-1].append(_neuron_coords.to(self.device))
                if self.return_pupil_center:
                    _pupil_center = datapoint.pupil_center
                    # to_return[self.dataloader_iters[dl_idx]["data_key"]].append(_pupil_center.to(self.device))
                    to_return[-1].append(_pupil_center.to(self.device))
                break
            except StopIteration:
                ### no more data in this dataloader
                self.dataloaders_left = [_dl_idx for _dl_idx in self.dataloaders_left if _dl_idx != dl_idx]
                del self.dataloader_iters[dl_idx]
                self.n_dataloaders -= 1
                if self.n_dataloaders == 0:  # no more data
                    break
                else:
                    continue

        return to_return
    
    def _get_parallel(self):
        """
        Mixes batches from multiple dataloaders.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        empty_dataloader_idxs = set()
        to_return = []
        for dl_idx, dataloader_iter in self.dataloader_iters.items():
            try:
                datapoint = next(dataloader_iter["dl"])
                to_return.append([dataloader_iter["data_key"], datapoint.images.to(self.device), datapoint.responses.to(self.device)])
                if self.return_neuron_coords:
                    to_return[-1].append(self.neuron_coords[dataloader_iter["data_key"]].to(self.device))
                if self.return_pupil_center and "pupil_center" in datapoint._fields:
                    to_return[-1].append(datapoint.pupil_center.to(self.device))
            except StopIteration:
                ### no more data in this dataloader
                if self.mixing_strategy == "parallel_min":
                    ### end the whole loop
                    empty_dataloader_idxs = set(self.dataloader_iters.keys())
                elif self.mixing_strategy == "parallel_max":
                    ### continue with the remaining ones
                    empty_dataloader_idxs.add(dl_idx)
                else:
                    raise NotImplementedError

        ### remove empty dataloaders
        if len(empty_dataloader_idxs) > 0:
            # new_dataloader_iters = []
            # for dl_idx, dataloader_iter in enumerate(self.dataloader_iters):
            #     if dl_idx not in empty_dataloader_idxs:
            #         new_dataloader_iters.append(dataloader_iter)
            # self.dataloader_iters = new_dataloader_iters
            for dl_idx_to_remove in empty_dataloader_idxs:
                del self.dataloader_iters[dl_idx_to_remove]
            self.n_dataloaders = len(self.dataloader_iters)

        return to_return

        # ### concatenate
        # stim = torch.cat(stim, dim=0)
        # resp = torch.cat(resp, dim=0)
        # if self.return_data_key:
        #     return stim, resp, None
        # return stim, resp

    def __len__(self):
        """
        Returns the total number of batches.

        Returns:
            int: The total number of batches.
        """
        return self.n_batches

    def __iter__(self):
        """
        Returns the iterator object.

        Returns:
            MixedBatchLoader: The iterator object.
        """
        return self

    def __next__(self):
        """
        Returns the next batch of data.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        self.batch_idx += 1
        if self.mixing_strategy == "sequential":
            out = self._get_sequential()
        elif self.mixing_strategy in ("parallel_min", "parallel_max"):
            out = self._get_parallel()
        else:
            raise NotImplementedError

        if len(out) == 0: # no more data
            raise StopIteration

        return out


class SamplesDataset(Dataset):
    def __init__(self, stims, resps, pupil_centers=None, stim_transform=None, resp_transform=None):
        self.stims = stims
        self.resps = resps
        self.pupil_centers = pupil_centers
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor()
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor()

    def __len__(self):
        return len(self.stims)

    def __getitem__(self, idx):
        stimuli = self.stims[idx]
        responses = self.resps[idx]
        if self.stim_transform is not None:
            stimuli = self.stim_transform(stimuli)
        if self.resp_transform is not None:
            responses = self.resp_transform(responses)

        if self.pupil_centers is None:
            return namedtuple("Datapoint", ["images", "responses"])(stimuli, responses)
        else:
            return namedtuple("Datapoint", ["images", "responses", "pupil_center"])(stimuli, responses, self.pupil_centers[idx])


class PerSampleStoredDataset(Dataset):
    def __init__(self, dataset_dir, stim_transform=None, resp_transform=None, device="cpu"):
        self.dirname = dataset_dir
        self.stim_transform = stim_transform if stim_transform is not None else NumpyToTensor()
        self.resp_transform = resp_transform if resp_transform is not None else NumpyToTensor()
        self.file_names = [
            f_name for f_name in os.listdir(self.dirname)
            if f_name.endswith(".pkl") or f_name.endswith(".pickle")
        ]
        self.device = device

    @property
    def n_neurons(self):
        return self[0][1].shape[-1]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        with open(os.path.join(self.dirname, f_name), "rb") as f:
            data = pickle.load(f)
            stimuli = data["stim"]
            responses = data["resp"]
            if self.stim_transform is not None:
                stimuli = self.stim_transform(stimuli)
            if self.resp_transform is not None:
                responses = self.resp_transform(responses)

            if "pupil_center" in data:
                return namedtuple("Datapoint", ["images", "responses", "pupil_center"])(
                    stimuli.to(self.device),
                    responses.to(self.device),
                    data["pupil_center"].to(self.device)
                )
            else:
                return namedtuple("Datapoint", ["images", "responses"])(
                    stimuli.to(self.device),
                    responses.to(self.device)
                )


class NumpyToTensor:
    def __init__(self, device="cpu", unsqueeze_dims=None):
        self.__unsqueeze_dims = unsqueeze_dims
        self.__device = device

    def __call__(self, x, *args, **kwargs):
        if self.__unsqueeze_dims is not None:
            x = np.expand_dims(x, self.__unsqueeze_dims)
        return torch.from_numpy(x).float().to(self.__device)
