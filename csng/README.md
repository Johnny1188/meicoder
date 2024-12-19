## File Descriptions

### 1. `data.py`
Provides utilities for data loading and preprocessing.

### 2. `inspect_data.ipynb`
Interactive notebook to:
- Visualize neural data (stimuli and responses).
- Debug data loading pipelines.

### 3. `losses.py`
Implements custom loss functions and evaluation metrics

### 4. `run_cnn_decoder.py`
Pipeline for training convolutional neural network (CNN)-based decoders.

### 5. `run_gan_decoder.py`
Pipeline for training GAN-based decoders

### 6. `run_comparison.py`
Script for comparing various decoding models.

### 7. `plot.ipynb`
Interactive notebook to make plots in the report.

### 8. `brainreader_mouse/train_encoder.py`
Script for training the encoder model (images -> responses).

### 9. `brainreader_mouse/generate_meis.py`
Script for generating MEIs using the encoder model.

### 10. `brainreader_mouse/encoder_inversion.py`
Script for performing hyperparameter search for encoder inversion.

### 11. `brainreader_mouse/data.py`
Utility functions for loading and preprocessing the dataset.

### 12. `brainreader_mouse/encoder.py`
Utility function for loading the encoder model.

---

## Usage

### 1. Train a CNN Decoder
Run the CNN decoder training pipeline:
```bash
python run_cnn_decoder.py
```

### 2. Train a GAN Decoder
Run the GAN decoder training pipeline:
```bash
python run_gan_decoder.py
```

### 3. Compare Models
Evaluate and compare multiple trained models:
```bash
python run_comparison.py
```

### 4. Inspect Data
Open the Jupyter notebook to visualize datasets:
```bash
jupyter notebook inspect_data.ipynb
```
