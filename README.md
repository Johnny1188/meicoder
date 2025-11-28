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

Install the modified packages [neuralpredictors](https://github.com/sinzlab/neuralpredictors), [nnfabrik](https://github.com/sinzlab/nnfabrik), [featurevis](https://github.com/ecobost/featurevis), [sensorium](https://github.com/sinzlab/sensorium) (modified for Python 3.10 compatibility and additional features), and the package for the CAE decoder ([YeChen, et al. 2024](https://doi.org/10.1371/journal.pcbi.1012297)) in the `pkgs` directory:
```bash
pip install -e pkgs/neuralpredictors pkgs/nnfabrik pkgs/featurevis pkgs/sensorium pkgs/CAE
```

Create `.env` file in the root directory according to `.env.example` file and make sure to set the path to an existing directory where the data will reside (`DATA_PATH`). You might need to load the environment variable(s) from the `.env` file manually in the terminal: `export $(cat .env | xargs)`


## Directory structure
- `README.md` - This file
- `setup.py` - Setup file for the `csng` package
- `environment.yaml` - Environment file with all the dependencies
- `.env.example` - Example of the `.env` file. Important to setup your own .env file in the same directory to be able to run the scripts
- `.gitignore` - Git ignore file
- `pkgs` - Directory containing modified packages `neuralpredictors`, `nnfabrik`, `featurevis`, and `sensorium`. Directories `pkgs/MindEye2` and `pkgs/MonkeySee` contain code for the [MindEye2](https://arxiv.org/abs/2403.11207) and [MonkeySee](https://openreview.net/forum?id=OWwdlxwnFN) decoders.
- `csng` - Directory containing the main code for the project (see `csng/README.md` for details):
  - `cat_v1/` - Directory with code specific to the cat V1 data (**C**)
  - `mouse_v1/` - Directory with code specific to the SENSORIUM 2022 mouse V1 data (datasets **M-\<mouse id\>** and **M-All**)
  - `brainreader_mouse/` - Directory with code specific to the mouse V1 data from [Cobos E. et al. 2022](https://doi.org/10.1101/2022.12.09.519708) (datasets **B-\<mouse id\>** and **B-All**)
  - `<your-data>/` - Directory with code specific to your data (e.g., `cat_v1/`). This folder should include a dataloading utility that could be then combined with other datasets using the code in `csng/data.py`.
- `notebooks/` - Directory with Jupyter notebooks for plotting, inspecting data and model performance, and for demonstration purposes. Notebook `notebooks/train.ipynb` is a minimal example of how to train a model using the `csng` package on one of the provided datasets, serving as a good starting point for your own experiments.

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