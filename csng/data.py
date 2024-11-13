import os
from collections import namedtuple
import skimage.transform
import torch
import torch.nn.functional as F
from nnfabrik.builder import get_data

from csng.brainreader_mouse.data import get_brainreader_data
from csng.mouse_v1.data_utils import get_mouse_v1_data, append_syn_dataloaders, append_data_aug_dataloaders, average_test_multitrial
from csng.cat_v1.data import prepare_v1_dataloaders


def get_sample_data(dls, config):
    s = {"stim": None, "resp": None, "sample_data_key": None, "sample_dataset": None}

    if "brainreader_mouse" in config["data"]:
        s["b_sample_dataset"] = "brainreader_mouse"
        b_dp = next(iter(dls["val"][s["b_sample_dataset"]]))
        s["b_stim"], s["b_resp"], s["b_sample_data_key"] = b_dp[0]["stim"], b_dp[0]["resp"], b_dp[0]["data_key"]
        s["stim"], s["resp"], s["sample_data_key"], s["sample_dataset"] = s["b_stim"], s["b_resp"], s["b_sample_data_key"], s["b_sample_dataset"]
    if "cat_v1" in config["data"]:
        s["c_sample_dataset"] = "cat_v1"
        c_dp = next(iter(dls["val"][s["c_sample_dataset"]]))
        s["c_stim"], s["c_resp"], s["c_sample_data_key"] = c_dp[0]["stim"], c_dp[0]["resp"], c_dp[0]["data_key"]
        s["stim"], s["resp"], s["sample_data_key"], s["sample_dataset"] = s["c_stim"], s["c_resp"], s["c_sample_data_key"], s["c_sample_dataset"]
    if "mouse_v1" in config["data"]:
        s["m_sample_dataset"] = "mouse_v1"
        m_dp = next(iter(dls["val"][s["m_sample_dataset"]]))
        s["m_stim"], s["m_resp"], s["m_sample_data_key"], s["m_pupil_center"] = m_dp[0]["stim"], m_dp[0]["resp"], m_dp[0]["data_key"], m_dp[0]["pupil_center"]
        s["stim"], s["resp"], s["sample_data_key"], s["sample_dataset"] = s["m_stim"], s["m_resp"], s["m_sample_data_key"], s["m_sample_dataset"]

    return s


def get_dataloaders(config):
    dls = dict(train=dict(), val=dict(), test=dict())
    neuron_coords = dict()

    ### brainreader_mouse
    if "brainreader_mouse" in config["data"]:
        _dls = get_brainreader_data(config=config["data"]["brainreader_mouse"])
        for tier in ("train", "val", "test"):
            dls[tier]["brainreader_mouse"] = _dls["brainreader_mouse"][tier]
        neuron_coords["brainreader_mouse"] = {data_key: None for data_key in _dls["brainreader_mouse"]["train"].data_keys}

        # data_keys = _dataloaders["brainreader_mouse"]["train"].data_keys
        # dataloaders = dict(
        #     train=dict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["train"].dataloaders)}),
        #     val=dict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["val"].dataloaders)}),
        #     test=dict({data_key: dl for data_key, dl in zip(data_keys, _dataloaders["brainreader_mouse"]["test"].dataloaders)}),
        # )

        # for tier in ("train", "val", "test"):
        #     dls[tier]["brainreader_mouse"] = MixedBatchLoaderV2(
        #         dataloaders=dataloaders[tier],
        #         neuron_coords=None,
        #         mixing_strategy=config["data"]["mixing_strategy"],
        #         max_batches=config["data"].get("max_training_batches"),
        #         data_keys=data_keys,
        #         return_data_key=True,
        #         return_pupil_center=False,
        #         device=config["device"],
        #     )

    ### mouse v1 - base
    if "mouse_v1" in config["data"] and config["data"]["mouse_v1"] is not None:
        ### get dataloaders
        _dataloaders = get_data(config["data"]["mouse_v1"]["dataset_fn"], config["data"]["mouse_v1"]["dataset_config"])

        if config["data"]["mouse_v1"]["average_test_multitrial"]:
            _dataloaders["test"] = average_test_multitrial(_dataloaders["test"], config["data"])

        m_dls = {
            "mouse_v1": {
                "train": MixedBatchLoaderV2(
                    dataloaders=[_dataloaders["train"][data_key] for data_key in _dataloaders["train"].keys()],
                    neuron_coords=None,  # added below
                    mixing_strategy=config["data"]["mixing_strategy"],
                    max_batches=config["data"].get("max_training_batches"),
                    data_keys=list(_dataloaders["train"].keys()),
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ) if config["data"]["mouse_v1"]["skip_train"] is False else MixedBatchLoader(
                    dataloaders=[],
                    neuron_coords=None,
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=[],
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ),
                "val": MixedBatchLoaderV2(
                    dataloaders=[_dataloaders["validation"][data_key] for data_key in _dataloaders["validation"].keys()],
                    neuron_coords=None,  # added below
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=list(_dataloaders["train"].keys()),
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ) if config["data"]["mouse_v1"]["skip_val"] is False else MixedBatchLoader(
                    dataloaders=[],
                    neuron_coords=None,
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=[],
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ),
                "test": MixedBatchLoaderV2(
                    dataloaders=[_dataloaders["test"][data_key] for data_key in _dataloaders["test"].keys()],
                    neuron_coords=None,  # added below
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=list(_dataloaders["train"].keys()),
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ) if config["data"]["mouse_v1"]["skip_test"] is False else MixedBatchLoader(
                    dataloaders=[],
                    neuron_coords=None,
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=[],
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                ),
                "test_no_resp": MixedBatchLoaderV2(
                    dataloaders=[_dataloaders["final_test"][data_key] for data_key in _dataloaders["final_test"].keys()],
                    neuron_coords=None,  # added below
                    mixing_strategy=config["data"]["mixing_strategy"],
                    data_keys=list(_dataloaders["train"].keys()),
                    return_data_key=True,
                    return_pupil_center=True,
                    device=config["data"]["mouse_v1"]["device"],
                )
            }
        }
        
        ### get cell coordinates
        _neuron_coords = {
            data_key: torch.tensor(d.neurons.cell_motor_coordinates, dtype=torch.float32, device=config["data"]["mouse_v1"]["device"])
            for data_key, d in zip(list(_dataloaders["train"].keys()), [_dl.dataset for _dl in _dataloaders["train"].values()])
        }
        if config["data"]["mouse_v1"]["normalize_neuron_coords"]: # normalize coordinates to [-1, 1]
            for data_key in _neuron_coords.keys():
                ### normalize x,y,z separately
                for dim_idx in range(_neuron_coords[data_key].shape[-1]):
                    _neuron_coords[data_key][:, dim_idx] = \
                        (_neuron_coords[data_key][:, dim_idx] - _neuron_coords[data_key][:, dim_idx].min()) \
                        / (_neuron_coords[data_key][:, dim_idx].max() - _neuron_coords[data_key][:, dim_idx].min()) * 2 - 1

        ### assign neuron_coords to dataloaders
        for dl_type in ["train", "val", "test", "test_no_resp"]:
            m_dls["mouse_v1"][dl_type].neuron_coords = _neuron_coords
        neuron_coords["mouse_v1"] = _neuron_coords

        ### mouse v1 - synthetic data
        if "syn_dataset_config" in config["data"] and config["data"]["syn_dataset_config"] is not None:
            raise NotImplementedError
            m_dls = append_syn_dataloaders(
                dataloaders=m_dls,
                config=config["data"]["syn_dataset_config"]
            )

        ### mouse v1 - data augmentation
        if "data_augmentation" in config["data"] and config["data"]["data_augmentation"] is not None:
            raise NotImplementedError
            m_dls = append_data_aug_dataloaders(
                dataloaders=m_dls,
                config=config["data"]["data_augmentation"],
            )

        ### add to dls
        for tier in ("train", "val", "test"):
            dls[tier]["mouse_v1"] = m_dls["mouse_v1"][tier]

    ### cat v1
    if "cat_v1" in config["data"]:
        c_dls = prepare_v1_dataloaders(**config["data"]["cat_v1"]["dataset_config"])

        ### get neuron coordinates
        torch.allclose(c_dls["train"].dataset[0].neuron_coords, c_dls["train"].dataset[-1].neuron_coords) and \
        torch.allclose(c_dls["train"].dataset[-1].neuron_coords, c_dls["val"].dataset[0].neuron_coords) and \
        torch.allclose(c_dls["val"].dataset[0].neuron_coords, c_dls["val"].dataset[-1].neuron_coords) and \
        torch.allclose(c_dls["val"].dataset[-1].neuron_coords, c_dls["test"].dataset[0].neuron_coords) and \
        torch.allclose(c_dls["test"].dataset[0].neuron_coords, c_dls["test"].dataset[-1].neuron_coords), \
            "Neuron coordinates must be the same for all samples in the dataset"
        neuron_coords["cat_v1"] = c_dls["train"].dataset[0].neuron_coords.float().to(config["device"])

        ### add to dls
        for tier in ("train", "val", "test"):
            dls[tier]["cat_v1"] = MixedBatchLoaderV2(
                dataloaders=[c_dls[tier]],
                neuron_coords={"cat_v1": neuron_coords["cat_v1"]},
                mixing_strategy=config["data"]["mixing_strategy"],
                max_batches=config["data"].get("max_training_batches"),
                data_keys=["cat_v1"],
                return_data_key=True,
                return_pupil_center=False,
                device=config["device"],
            )

    return dls, neuron_coords


class MixedBatchLoaderV2:
    def __init__(
        self,
        dataloaders,
        neuron_coords=None,
        mixing_strategy="sequential",
        max_batches=None,
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
        self.max_batches = max_batches
        
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
        if self.max_batches is not None:
            self.n_batches = min(self.n_batches, self.max_batches)

        self.datasets = []
        for dl in dataloaders:
            if hasattr(dl, "dataset"):
                self.datasets.append(dl.dataset)
            else:
                self.datasets.append(dl)

    def add_dataloader(self, dataloader, neuron_coords=None, data_key=None):
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
        to_return = []
        while True:
            dl_idx = self.dataloaders_left[self.batch_idx % self.n_dataloaders]
            try:
                datapoint = next(self.dataloader_iters[dl_idx]["dl"])
                to_return.append(dict(data_key=None, stim=None, resp=None, neuron_coords=None, pupil_center=None))
                to_return[-1]["stim"] = datapoint.images.to(self.device)
                to_return[-1]["resp"] = datapoint.responses.to(self.device)
                if self.return_data_key:
                    to_return[-1]["data_key"] = self.dataloader_iters[dl_idx]["data_key"]
                if self.return_neuron_coords:
                    _neuron_coords = self.neuron_coords[self.dataloader_iters[dl_idx]["data_key"]]
                    to_return[-1]["neuron_coords"] = _neuron_coords.to(self.device)
                if self.return_pupil_center:
                    to_return[-1]["pupil_center"] = datapoint.pupil_center.to(self.device)
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
        empty_dataloader_idxs = set()
        to_return = []
        for dl_idx, dataloader_iter in self.dataloader_iters.items():
            try:
                datapoint = next(dataloader_iter["dl"])
                to_return.append(dict(data_key=None, stim=None, resp=None, neuron_coords=None, pupil_center=None))
                to_return[-1]["stim"] = datapoint.images.to(self.device)
                to_return[-1]["resp"] = datapoint.responses.to(self.device)
                if self.return_data_key:
                    to_return[-1]["data_key"] = dataloader_iter["data_key"]
                if self.return_neuron_coords:
                    to_return[-1]["neuron_coords"] = self.neuron_coords[dataloader_iter["data_key"]].to(self.device)
                if self.return_pupil_center and "pupil_center" in datapoint._fields:
                    to_return[-1]["pupil_center"] = datapoint.pupil_center.to(self.device)
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
            for dl_idx_to_remove in empty_dataloader_idxs:
                del self.dataloader_iters[dl_idx_to_remove]
            self.n_dataloaders = len(self.dataloader_iters)

        return to_return

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_idx += 1
        if self.mixing_strategy == "sequential":
            out = self._get_sequential()
        elif self.mixing_strategy in ("parallel_min", "parallel_max"):
            out = self._get_parallel()
        else:
            raise NotImplementedError

        if len(out) == 0 or (self.max_batches is not None and self.batch_idx > self.max_batches):
            raise StopIteration

        return out


class MixedBatchLoader:
    """
    A dataloader that interleaves or mixes batches from multiple dataloaders.

    Args:
        dataloaders (list): A list of dataloaders to interleave or mix batches from.
        mixing_strategy (str): The strategy to use for mixing batches. Must be one of ["sequential", "parallel_min", "parallel_max"].
        device (str): The device to load the batches onto. Default is "cpu".

    Raises:
        AssertionError: If mixing_strategy is not one of ["sequential", "parallel_min", "parallel_max"].

    Attributes:
        dataloader_iters (list): A list of iterators for each dataloader.
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
    
    def __init__(self, dataloaders, mixing_strategy="sequential", data_keys=None, return_data_key=False, device="cpu"):
        assert mixing_strategy in ["sequential", "parallel_min", "parallel_max"], \
            f"mixing_strategy must be one of ['sequential', 'parallel'], but got {mixing_strategy}"

        self.dataloaders = dataloaders
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
        self.device = device
        self.batch_idx = 0
        
        if self.mixing_strategy == "sequential":
            self.n_batches = sum([len(dataloader) for dataloader in dataloaders])
        elif self.mixing_strategy == "parallel_max":
            self.n_batches = max([len(dataloader) for dataloader in dataloaders])
        elif self.mixing_strategy == "parallel_min":
            self.n_batches = min([len(dataloader) for dataloader in dataloaders])

        self.datasets = []
        for dl in dataloaders:
            if hasattr(dl, "dataset"):
                self.datasets.append(dl.dataset)
            else:
                self.datasets.append(dl)


    def add_dataloader(self, dataloader, data_key=None):
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
        self.n_batches += len(dataloader)
        if hasattr(dataloader, "dataset"):
            self.datasets.append(dataloader.dataset)
        else:
            self.datasets.append(dataloader)


    def _get_sequential(self):
        """
        Interleaves batches from multiple dataloaders.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        while True:
            dl_idx = self.dataloaders_left[self.batch_idx % self.n_dataloaders]
            try:
                stim, resp = next(self.dataloader_iters[dl_idx]["dl"])
                break
            except StopIteration:
                self.dataloaders_left = [_dl_idx for _dl_idx in self.dataloaders_left if _dl_idx != dl_idx]
                del self.dataloader_iters[dl_idx]
                self.n_dataloaders -= 1
                if self.n_dataloaders == 0:
                    if self.return_data_key:
                        return None, None, None
                    return None, None
                else:
                    continue

        if self.return_data_key:
            return stim.to(self.device), resp.to(self.device), self.dataloader_iters[dl_idx]["data_key"]
        return stim.to(self.device), resp.to(self.device)
    
    def _get_parallel(self):
        """
        Mixes batches from multiple dataloaders.

        Returns:
            tuple: A tuple of tensors containing the stimulus and response data for the next batch.
        """
        stim, resp = [], []
        empty_dataloader_idxs = set()
        for dl_idx, dataloader_iter in self.dataloader_iters.items():
            try:
                _stim, _resp = next(dataloader_iter["dl"])
                stim.append(_stim.to(self.device))
                resp.append(_resp.to(self.device))
            except StopIteration:
                if self.mixing_strategy == "parallel_min":
                    ### if a single dataloader ends, end the whole loop
                    # _ = [empty_dataloader_idxs.add(_dl_idx) for _dl_idx in range(0, self.n_dataloaders)]
                    empty_dataloader_idxs = set(self.dataloader_iters.keys())
                elif self.mixing_strategy == "parallel_max":
                    ### if a single dataloader ends, continue with the remaining ones
                    # empty_dataloader_idxs.add(dl_idx)
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

        if len(stim) == 0:
            if self.return_data_key:
                return None, None, None
            return None, None

        ### concatenate
        stim = torch.cat(stim, dim=0)
        resp = torch.cat(resp, dim=0)
        if self.return_data_key:
            return stim, resp, None
        return stim, resp

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
        
        if out[0] is None: # no more data
            raise StopIteration
        return out


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

    def _add_noise(self, responses):
        noise_std = self.noise_std
        if self.dynamic_mul_factor > 0:
            noise_std = noise_std.unsqueeze(0).repeat(responses.size(0), 1)
            noise_std += self.dynamic_mul_factor * self._transform_resp(responses)
        noise = torch.randn_like(responses) * noise_std

        resp_out = responses + noise
        if self.clip_min is not None:
            resp_out = torch.clamp(resp_out, min=self.clip_min)

        return resp_out

    def __call__(self, data):
        return namedtuple("Datapoint", ["images", "responses", "pupil_center"])(
            data.images,
            self._add_noise(data.responses),
            data.pupil_center,
        )
