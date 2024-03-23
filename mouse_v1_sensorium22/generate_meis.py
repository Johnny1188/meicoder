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


class SingleCellModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx
    
    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)[:, self.idx]


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

    encoder = get_encoder(
        device=config["device"],
        eval_mode=True,
        use_shifter=False,
        ckpt_path=os.path.join(DATA_PATH, "models", "encoder_sens22_no_shifter.pth"),
    )
    all_meis, all_vals, all_reg_vals = [], [], []
    for cell_idx in tqdm(range(encoder.readout[config["data_key"]].outdims), ncols=50, position=0, leave=True):
        single_cell_encoder = SingleCellModel(model=encoder, idx=cell_idx) # featurevis requires the model to return response for only one neuron.

        # operation to perform on the image before passing it to the model
        transforms = featurevis.utils.Compose([
            ops.ChangeStats(std=config["mei"]["std"], mean=config["mei"]["mean"]),
            ops.ClipRange(config["mei"]["pixel_min"], config["mei"]["pixel_max"])
        ])

        # set random initial image to optimise with correct amount of std and mean around midgrey
        initial_image = torch.randn(1, 1, *config["mei"]["img_res"], dtype=torch.float32) * config["mei"]["std"] + config["mei"]["mean"]
        initial_image = transforms(initial_image).to(config["device"])
        single_mei, vals, reg_vals = featurevis.gradient_ascent(
            single_cell_encoder,
            initial_image, 
            step_size=config["mei"]["step_size"],
            num_iterations=config["mei"]["num_iterations"],
            post_update=transforms, 
            gradient_f=config["mei"]["gradient_f"],
            print_iters=config["mei"]["print_iters"],
            additional_f_kwargs={"data_key": config["data_key"]}, # additional input for the base encoder
        )

        all_meis.append(single_mei.cpu())
        all_vals.append(vals)
        all_reg_vals.append(reg_vals)

    ### save the results
    save_dir = os.path.join(config["save_path"], config["data_key"])
    print(f"[INFO] Saving MEIs to {save_dir} [meis.pt, vals.pt, reg_vals.pt]")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)
    torch.save(torch.cat(all_meis, dim=0), os.path.join(save_dir, "meis.pt"), pickle_module=dill)
    torch.save(all_vals, os.path.join(save_dir, "vals.pt"))
    torch.save(all_reg_vals, os.path.join(save_dir, "reg_vals.pt"))
    print("[INFO] Done.")


if __name__ == "__main__":
    generate_meis()
