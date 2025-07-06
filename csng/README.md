## In this folder

`data.py`:
Provides utilities for data loading and preprocessing.

`losses.py`:
Implements custom loss functions and evaluation metrics.

`generate_meis.py`:
Script for generating most exciting inputs (MEIs) using a pre-trained encoder model.

`run_cnn_decoder.py`:
Pipeline for training convolutional neural network (CNN)-based decoders.

`run_naive_decoder.py`:
Pipeline for training CNN-based decoders with fully-connected readins and MSE training loss.

`run_gan_decoder.py`:
Pipeline for training GAN-based decoders.

`run_comparison.py`:
Script for comparing decoding models.

`utils/`:
Contains utility functions for various tasks, such as data loading, model evaluation, plotting, etc.

`models/`:
Contains model definitions and training scripts.

`<dataset-name>/train_encoder.py`:
Script for training the encoder model (images -> responses).

`<dataset-name>/encoder_inversion.py`:
Script for performing hyperparameter search with the encoder inversion decoding method.

`<dataset-name>/data.py`:
Utility functions for loading and preprocessing the dataset.

`<dataset-name>/encoder.py`:
Utility function for loading the encoder model.

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
