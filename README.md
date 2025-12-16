# MEIcoder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs

[Jan Sobotka](https://johnny1188.github.io), [Luca Baroni](https://lucabaroni.github.io), [Ján Antolík](http://antolik.net)

---

This project focuses on decoding visual scenes from population neural activity recorded in the early visual system. The repository will be cleaned in the coming weeks, but the main code is fully functional and can be used to reproduce the results from our [NeurIPS 2025 paper](https://arxiv.org/abs/2510.20762) as well as to run your own experiments.


## Data
For instructions on getting the `cat_v1` and `mouse_v1` datasets, please refer to the README files in the respective directories `csng/cat_v1/` and `csng/mouse_v1/`.

## Environment setup
Setup an environment from the `environment.yaml` file and activate it ([Miniconda](https://docs.anaconda.com/free/miniconda/index.html)):
```bash
conda env create -f environment.yaml
conda activate csng
```

Install the main `csng` package:
```bash
pip install -e .
```

Install the modified packages [neuralpredictors](https://github.com/sinzlab/neuralpredictors), [nnfabrik](https://github.com/sinzlab/nnfabrik), [featurevis](https://github.com/ecobost/featurevis), [sensorium](https://github.com/sinzlab/sensorium) (modified for Python 3.10 compatibility and additional features), and the packages for the [CAE decoder](https://doi.org/10.1371/journal.pcbi.1012297), [MonkeySee](https://openreview.net/forum?id=OWwdlxwnFN), and [Energy-Guided Diffusion (EGG)](https://openreview.net/forum?id=1moStpWGUj) in the `pkgs` directory:
```bash
pip install -e pkgs/neuralpredictors pkgs/nnfabrik pkgs/featurevis pkgs/sensorium pkgs/CAE pkgs/MonkeySee pkgs/energy-guided-diffusion
```

Create `.env` file in the root directory according to `.env.example` file and make sure to set the path to an existing directory where the data will reside (`DATA_PATH`). You might need to load the environment variable(s) from the `.env` file manually in the terminal: `export $(cat .env | xargs)`


## Directory structure
- `README.md` - This file
- `setup.py` - Setup file for the `csng` package
- `environment.yaml` - Environment file with all the dependencies
- `.env.example` - Example of the `.env` file. Important to setup your own .env file in the same directory to be able to run the scripts
- `.gitignore` - Git ignore file
- `pkgs` - Directory containing modified packages `neuralpredictors`, `nnfabrik`, `featurevis`, `sensorium`. Directories `pkgs/CAE`, `pkgs/MindEye2`, `pkgs/MonkeySee`, and `pkgs/energy-guided-diffusion` contain code for the [CAE decoder](https://doi.org/10.1371/journal.pcbi.1012297), [MindEye2](https://arxiv.org/abs/2403.11207), [MonkeySee](https://openreview.net/forum?id=OWwdlxwnFN), and [Energy-Guided Diffusion (EGG)](https://openreview.net/forum?id=1moStpWGUj), respectively.
- `csng` - Directory containing the main code for the project (see `csng/README.md` for details):
  - `cat_v1/` - Directory with code specific to the cat V1 data (**C**)
  - `mouse_v1/` - Directory with code specific to the SENSORIUM 2022 mouse V1 data (datasets **M-\<mouse id\>** and **M-All**)
  - `brainreader_mouse/` - Directory with code specific to the mouse V1 data from [Cobos E. et al. 2022](https://doi.org/10.1101/2022.12.09.519708) (datasets **B-\<mouse id\>** and **B-All**)
  - `<your-data>/` - Directory with code specific to your data (e.g., `cat_v1/`). This folder should include a dataloading utility that could be then combined with other datasets using the code in `csng/data.py`.
- `notebooks/` - Directory with Jupyter notebooks for plotting, inspecting data and model performance, and for demonstration purposes. Notebook `notebooks/train.ipynb` is a minimal example of how to train a model using the `csng` package on one of the provided datasets, serving as a good starting point for your own experiments.


## Running experiments
The main training script is `csng/run_gan_decoder.py`. The script is highly configurable via the `config` dictionary defined at the top of the file. Below are the steps to run an experiment:

1) Prepare data and MEIs: download/generate datasets as described in `csng/<dataset>/README.md` files and produce MEIs with `csng/generate_meis.py`. Place the resulting `meis.pt` under `DATA_PATH/.../meis/<data_key>/` so it matches the `meis_path` entries in the config (cat V1 MEIs are linked in `csng/README.md`).
2) Activate the environment: `conda activate csng`.
3) Choose the dataset block in `csng/run_gan_decoder.py`: uncomment and edit the relevant `config["data"]["<dataset>"]` entry (e.g., `brainreader_mouse`, `cat_v1`, or `mouse_v1`). Verify batch sizes, resize targets, and any neuron coordinate settings.
4) Launch training: run `python csng/run_gan_decoder.py`. Checkpoints and logs are written to the run directory configured by `setup_run_dir` (defaults to `DATA_PATH/models/gan/<timestamp>`). Resume or fine-tune by filling the `config["decoder"]["load_ckpt"]` block.

### Configuration essentials
The top-level `config` dictionary in `csng/run_gan_decoder.py` controls the full experiment:
- `config["device"]`, `["seed"]`, `["save_run"]`, and `["wandb"]` handle reproducibility, checkpointing, and logging.
- `config["data"]` holds per-dataset dataloader settings (paths, batch sizes, normalization, cropping, optional neuron coords). Only keep blocks for the datasets you want to train on, and comment out or remove others.
- `config["decoder"]["readin_type"]` should stay `mei` for MEIcoder. However, you can create your own readin modules by inheriting from `csng.models.readins.ReadIn` (see `csng.models.readins.MEIReadIn` for example) and then specifying the new class name here.
- `config["decoder"]["model"]` defines the GAN core (generator/discriminator shapes) and appends a `readins_config` entry per dataset. Each MEI entry specifies `meis_path`, `mei_target_shape`, `mei_resize_method`, whether MEIs are trainable, contextual modulation (`ctx_net_config`), and pointwise convolution settings. Shapes must match the crop windows in `config["crop_wins"]`.
- `config["decoder"]["loss"]`, optimizer settings (`G_opter_kwargs`/`D_opter_kwargs`), adversarial/Stim loss weights, and `n_epochs` control training dynamics. `eval_loss_name` selects the validation metric used for checkpointing.

After edits, rerun `python csng/run_gan_decoder.py` to train with the updated configuration.


## Final evaluation on test sets
Use `csng/run_comparison.py` to evaluate MEIcoder (and baselines) on held-out test sets and plot metrics/reconstructions.

1) Pick datasets: enable the needed `config["data"]["<dataset>"]` blocks (e.g., `brainreader_mouse`, `cat_v1`, `mouse_v1`) at the top of `csng/run_comparison.py`; crop windows are inferred there.
2) Point to checkpoints: in `config["comparison"]["to_compare"]`, set each model’s `ckpt_path` (or provide a `decoder` object). `run_name` is used just for labeling on figures. `load_best=True` loads the best-val checkpoint; `eval_all_ckpts` or `find_best_ckpt_according_to` lets you sweep/auto-pick checkpoints.
3) Configure evaluation: choose `eval_tier` (default `test`), metrics in `losses_to_plot`, and output directory via `save_dir`. Keep `save_all_preds_and_targets=True` if you need full tensors alongside plots. Optional `load_ckpt` lets you reload prior comparison results to re-plot without recompute.
4) Run: `python csng/run_comparison.py`. Results are saved under to `config["comparison"]["save_dir"]`.


---
## Citing
If you find our repository useful, please consider citing:
```
@inproceedings{
  sobotka2025meicoder,
  title={{MEI}coder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs},
  author={Jan Sobotka and Luca Baroni and J{\'a}n Antol{\'\i}k},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=V3WQoshcZe}
}
```
