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
    rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    decays = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    for model_name in models:
        for lr in rates:
            for weight_decay in decays:
                run_program(lr, weight_decay, model_name)


if __name__ == "__main__":
    main()
