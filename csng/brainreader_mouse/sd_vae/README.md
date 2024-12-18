# Latent Space Image Representation Project

## Data Preparation

### Generate Latent Space Data
To generate latent space representation for your images, run the `generate_latent_space_data.py` script with the following arguments:
```bash
python generate_latent_space_data.py --dataset [train/test/val] --session_id [session_number]
```

## Model Configuration

### Setup Configuration
Configure your model parameters in `setup.json`. Example configuration:
```json
{
    "lr": 1e-05,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "epochs": 5,
    "model_name": "fully_connected",
    "scheduler": false,
    "session_id": 6
}
```

## Running Models

### Deep Learning Models
1. Modify `setup.json` as needed
2. Run the training script:
```bash
python run.py
```
- Available models are listed in `models.py`

### Ridge Regression
To run ridge regression:
```bash
python run_ridge_regression.py
```
- Results will be saved in the `results` folder

## Generating Image Outputs

### Test Model Visualization
1. Open `test_model.py`
2. Update `results_path` to the desired latent space results
```python
results_path = "results/fully_connected/2024-12-14-16-56-50"
```
3. Run the script to generate final image outputs
```bash
python test_model.py
```
- Results will be saved in the `results` folder

## Notes
- Ensure all dependencies are installed
- Adjust paths and configurations according to your specific setup
