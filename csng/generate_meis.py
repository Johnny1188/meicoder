import os
import json
from tqdm import tqdm
import torch
from torch import nn
import dill
import featurevis
import featurevis.ops as ops

from csng.brainreader_mouse.encoder import get_encoder as get_encoder_brainreader
from csng.mouse_v1.encoder import get_encoder as get_encoder_mouse_v1
from csng.cat_v1.encoder import get_encoder as get_encoder_cat_v1
from csng.allen.encoder import get_encoder as get_encoder_allen

get_encoder_fns = {
    "brainreader_mouse": get_encoder_brainreader,
    "mouse_v1": get_encoder_mouse_v1,
    "cat_v1": get_encoder_cat_v1,
    "allen": get_encoder_allen,
}

### set paths
DATA_PATH = os.environ["DATA_PATH"]
DATA_PATH_CAE = os.path.join(DATA_PATH, "cae")
DATA_PATH_CAT_V1 = os.path.join(DATA_PATH, "cat_V1_spiking_model", "50K_single_trial_dataset")
DATA_PATH_MOUSE_V1 = os.path.join(DATA_PATH, "mouse_v1_sensorium22")
DATA_PATH_BRAINREADER = os.path.join(DATA_PATH, "brainreader")


### config
# config = {
#     "data_name": (data_name := "allen"),
#     "encoder_path": os.path.join(DATA_PATH, "models", "encoder_allen_6l96ch.pt"), # pre-trained encoder path
#     "data_key": "allen",
#     "save_path": os.path.join({
#         "cat_v1": DATA_PATH_CAT_V1,
#         "mouse_v1": DATA_PATH_MOUSE_V1,
#         "brainreader_mouse": DATA_PATH_BRAINREADER,
#         "allen": DATA_PATH_CAE,
#     }[data_name], "meis_0-05std_gauss2_encoder_allen_6l96ch"),
#     "chunk_size": 30, # number of cells to process at once
#     "mei": {
#         "mean": 0,
#         # "std": 0.15, # everything from 0.10 to 0.25 works here
#         "std": 0.05, # everything from 0.10 to 0.25 works here
#         "pixel_min": -1,
#         "pixel_max": 1,
#         # "img_res": (36, 64), # should be the size of the input image to the encoder
#         "img_res": (256, 256), # should be the size of the input image to the encoder
#         "step_size": 1,
#         "num_iterations": 1000,
#         "gradient_f": ops.GaussianBlur(2.), # blur the gradient to avoid artifacts
#         "print_iters": 1e10,
#     },
#     "device": os.environ["DEVICE"],
# }
config = {
    "data_name": (data_name := "brainreader_mouse"),
    "encoder_path": os.path.join(DATA_PATH, "models", (encoder_filename := "encoder_b6_8l128ch.pt")), # pre-trained encoder path
    "data_key": "6",
    "save_path": os.path.join({
        "cat_v1": DATA_PATH_CAT_V1,
        "mouse_v1": DATA_PATH_MOUSE_V1,
        "brainreader_mouse": DATA_PATH_BRAINREADER,
    }[data_name], f"meis__36-64_{encoder_filename.split('.')[0]}"),
    "chunk_size": 500, # number of cells to process at once
    "mei": {
        "mean": 0,
        "std": 0.15, # everything from 0.10 to 0.25 works here
        "pixel_min": -1,
        "pixel_max": 1,
        "img_res": (36, 64), # should be the size of the input image to the encoder
        # "img_res": (72, 128), # should be the size of the input image to the encoder
        # "img_res": (144, 256), # should be the size of the input image to the encoder
        "step_size": 1,
        "num_iterations": 1000,
        "gradient_f": ops.GaussianBlur(1.), # blur the gradient to avoid artifacts
        "print_iters": 1e10,
    },
    "device": os.environ["DEVICE"],
}


class BatchedEncoder(nn.Module):
    def __init__(self, model, cell_idx_start=None, cell_idx_end=None):
        super().__init__()
        self.model = model
        self.cell_idx_start = cell_idx_start
        self.cell_idx_end = cell_idx_end
    
    def forward(self, x, **kwargs):
        start = 0 if self.cell_idx_start is None else self.cell_idx_start
        end = x.shape[0] if self.cell_idx_end is None else self.cell_idx_end
        return self.model(x, **kwargs)[
            torch.arange(x.shape[0]),
            torch.arange(start, end),
        ].sum()


def generate_meis():
    """ generates MEIs for all cells in the model and saves them to disk """

    ### prepare save dir
    save_dir = os.path.join(config["save_path"], config["data_key"])
    print(f"[INFO] Saving MEIs to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "chunked"), exist_ok=True)

    ### load encoder
    encoder = get_encoder_fns[config["data_name"]](
        ckpt_path=config["encoder_path"],
        device=config["device"],
        eval_mode=True,
    )
    config["n_cells"] = encoder.readout[config["data_key"]].outdims
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    ### generate MEIs
    for cell_idx_start, cell_idx_end in zip(
        range(0, config["n_cells"], config["chunk_size"]),
        range(config["chunk_size"], config["n_cells"] + config["chunk_size"], config["chunk_size"]),
    ):
        ### featurevis requires the model to return response for only one neuron
        batched_encoder = BatchedEncoder(model=encoder, cell_idx_start=cell_idx_start, cell_idx_end=min(cell_idx_end, config["n_cells"]))

        ### operation to perform on the image before passing it to the model
        transforms = featurevis.utils.Compose([
            ops.ChangeStats(std=config["mei"]["std"], mean=config["mei"]["mean"]),
            ops.ClipRange(config["mei"]["pixel_min"], config["mei"]["pixel_max"])
        ])

        ### set random initial image to optimize with correct amount of std and mean around midgrey
        initial_image = torch.randn(
            min(cell_idx_end, config["n_cells"]) - cell_idx_start,
            1,
            *config["mei"]["img_res"],
            dtype=torch.float32,
        ) * config["mei"]["std"] + config["mei"]["mean"]
        initial_image = transforms(initial_image).to(config["device"])
        single_mei, vals, reg_vals = featurevis.gradient_ascent(
            batched_encoder,
            initial_image, 
            step_size=config["mei"]["step_size"],
            num_iterations=config["mei"]["num_iterations"],
            post_update=transforms, 
            gradient_f=config["mei"]["gradient_f"],
            print_iters=config["mei"]["print_iters"],
            additional_f_kwargs={"data_key": config["data_key"]}, # additional input for the base encoder
        )

        ### save the results
        file_name = f"{cell_idx_start}-{cell_idx_end}.pt"
        torch.save({
            "mei": single_mei.cpu(),
            "vals": vals,
            "reg_vals": reg_vals,
        }, os.path.join(save_dir, "chunked", file_name), pickle_module=dill)
        print(f"[INFO] chunk {cell_idx_start}-{cell_idx_end} processed.")
        
    print("[INFO] Chunked MEIs generated.")

    ### combine chunked MEIs and save
    meis = []
    vals = []
    reg_vals = []
    for file_name in tqdm(os.listdir(os.path.join(save_dir, "chunked"))):
        data = torch.load(os.path.join(save_dir, "chunked", file_name), pickle_module=dill)
        meis.append(data["mei"].cpu())
        vals.append(data["vals"])
        reg_vals.append(data["reg_vals"])

    torch.save({
        "meis": torch.cat(meis, dim=0),
        "vals": vals,
        "reg_vals": reg_vals,
    }, os.path.join(save_dir, "meis.pt"), pickle_module=dill)
    print("[INFO] MEIs generated.")


if __name__ == "__main__":
    generate_meis()
