# Notebooks

Short guide to the notebooks in this folder and what they cover.

- `analysis.ipynb` – End-to-end analysis workbook: loads the prepared datasets, runs representational-similarity checks, and evaluates MindEye2/GAN decoders on held-out data. Includes error breakdowns, ablation and sensitivity studies, and interpretability probes (e.g., MEI-driven neuron embeddings, NMF/MDS analyses).
- `infer.ipynb` – Minimal inference pipeline that wires up dataloaders across cat V1, mouse V1, and brainreader datasets. Demonstrates how to point a pretrained GAN decoder at the loaders and visualize reconstructed samples.
- `nli.ipynb` – Collects brainreader encoder responses, caches them, and trains a lightweight linearized encoder on stimuli. Evaluates correlations between the linear model, the pretrained encoder, and ground truth to gauge how much structure is captured by a linear readout.
- `plot_neuron_data_vs_performance.ipynb` – Loads saved comparison results and plots how reconstruction metrics scale with neuron count and dataset size. Provides line and bar plots to visualize performance trends.

## Archived notebooks
- `.archive/metrics.ipynb` – Older evaluation notebook computing reconstruction/brain-similarity metrics for the inverted encoder and GAN decoders.
- `.archive/plot.ipynb` – Legacy plotting helper that loads saved comparison runs and renders metric bar charts (SSIM, pixel correlation, AlexNet layers, MAE).
