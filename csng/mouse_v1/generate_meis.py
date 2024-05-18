import os
import json
from tqdm import tqdm
import torch
from torch import nn
import dill
import featurevis
import featurevis.ops as ops

from encoder import get_encoder

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "mouse_v1_sensorium22")


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
    
    ### config
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_key": "21067-10-18",
        # "data_key": "22846-10-16",
        # "data_key": "23343-5-17",
        # "data_key": "23656-14-22",
        # "data_key": "23964-4-22",
        "save_path": os.path.join(DATA_PATH, "meis"),
        "chunk_size": 500, # number of cells to process at once
        "mei": {
            "mean": 0, # (here people often uses 0. I think that the midgreyscale value is a better choice though (pixel_min + pixel_max)/2 )
            "std": 0.15, # (everything from 0.10 to 0.25 works here)
            "pixel_min": -1,
            "pixel_max": 1,
            "img_res": (36, 64),
            "step_size": 1,
            "num_iterations": 1000,
            "gradient_f": ops.GaussianBlur(1.), # this blurs the gradient to avoid artifacts.
            "print_iters": 1e10,
        }
    }

    ### prepare save dir
    save_dir = os.path.join(config["save_path"], config["data_key"])
    print(f"[INFO] Saving MEIs to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "chunked"), exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    ### load encoder
    encoder = get_encoder(
        ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_mall_no_shifter.pth"),
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
        # featurevis requires the model to return response for only one neuron
        batched_encoder = BatchedEncoder(model=encoder, cell_idx_start=cell_idx_start, cell_idx_end=min(cell_idx_end, config["n_cells"]))

        # operation to perform on the image before passing it to the model
        transforms = featurevis.utils.Compose([
            ops.ChangeStats(std=config["mei"]["std"], mean=config["mei"]["mean"]),
            ops.ClipRange(config["mei"]["pixel_min"], config["mei"]["pixel_max"])
        ])

        # set random initial image to optimise with correct amount of std and mean around midgrey
        initial_image = torch.randn(min(cell_idx_end, config["n_cells"]) - cell_idx_start, 1, *config["mei"]["img_res"], dtype=torch.float32) * config["mei"]["std"] + config["mei"]["mean"]
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

    ### combine chunked MEIs
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
