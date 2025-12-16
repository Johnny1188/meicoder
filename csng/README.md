## In this folder

- `run_gan_decoder.py`: Main MEIcoder training pipeline; config at the top selects dataset(s), MEI read-in, and GAN core, then runs the full training loop with checkpointing and optional Weights & Biases logging.
- `models/readins.py`: Readin modules used by decoders; `MEIReadIn` contextually modulates stored MEIs.
- `models/utils/gan.py`: Defines the GAN decoder core (generator/discriminator stacks), training utilities (`train`, `init_decoder`, `setup_run_dir`), and checkpoint helpers used by MEIcoder.
- `generate_meis.py`: Generate MEIs for each neuron from a pretrained encoder; save `meis.pt` files consumed by `MEIReadIn` (cat V1 MEIs are linked [here](https://drive.google.com/file/d/1NXwTU-056WSwq7Qy6uCnf3J3XFPuWstB/view?usp=sharing)).
- `data.py`: Dataset utilities and loader factories shared across decoders; handles normalization, cropping, mixing strategies, and neuron coordinate handling.
- `losses.py`: Custom metrics/losses (e.g., SSIM variants, Alex/CLIP/SwAV scores) used during training and evaluation.
- `run_comparison.py`: Final test-set evaluation/plotting script; loads checkpoints (MEIcoder or baselines), computes metrics, and saves reconstructions.
- `utils/`: Misc utilities for seeding, plotting, model inspection, training helpers, etc.
- `<dataset-name>/train_encoder.py`: Trains an encoder (images -> responses) for a specific dataset.
- `<dataset-name>/encoder_inversion.py`: Hyperparameter search for encoder-inversion decoding on that dataset.
- `<dataset-name>/data.py`: Dataset-specific loading/preprocessing code (paths, transforms, coordinates).
- `<dataset-name>/encoder.py`: Helper to load the pretrained encoder for that dataset.

---

## Usage

For each of the training scripts (`run_<model>_decoder.py`), first specify the configuration at the beginning of the script, and then run the script as follows:
```bash
python run_<model>_decoder.py
```

The training script will (by default) save the trained model and its checkpoints. To evaluate and compare multiple trained models, you can use the `run_comparison.py` script:
```bash
python run_comparison.py
```
In this file, you need to specify the paths to the trained models you want to compare, as well as the dataset and other parameters (see configuration at the top of the script).
