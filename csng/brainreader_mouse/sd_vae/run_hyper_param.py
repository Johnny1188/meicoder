from itertools import product

import numpy as np
import run
from utils import load_json, save_json


def run_program(
    lr,
    weight_decay,
    model_name,
):
    setup = load_json("setup.json")
    setup["lr"] = lr
    setup["weight_decay"] = weight_decay
    setup["model_name"] = model_name
    save_json(setup, "setup.json")
    run.main()


def main():
    models = ["fully_connected", "cnn", "deconv"]
    rates = np.logspace(-5, 0, 6)
    decays = np.logspace(-4, 0, 5)

    for model_name, lr, weight_decay in product(models, rates, decays):
        run_program(lr, weight_decay, model_name)


if __name__ == "__main__":
    main()
