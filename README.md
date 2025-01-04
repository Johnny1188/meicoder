# Decoding Brain Activity

This project focuses on decoding visual scenes from population neural activity recorded in the early visual system.


## Data
For instructions on getting the `cat_v1` and `mouse_v1` datasets, please refer to the README files in the respective directories `csng/cat_v1/` and `csng/mouse_v1/`. The `brainreader_mouse` data was provided to us directly by the original authors from the Andreas Tolias lab, but we did not get the permission to share it with third parties. You can email the original authors for access and use the notebook `csng/brainreader_mouse/preprocess_data.ipynb` to preprocess the raw data.


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

Install the modified packages [neuralpredictors](https://github.com/sinzlab/neuralpredictors), [nnfabrik](https://github.com/sinzlab/nnfabrik), [featurevis](https://github.com/ecobost/featurevis), and [sensorium](https://github.com/sinzlab/sensorium) in the `pkgs` directory (modified for Python 3.10 compatibility and additional features):
```bash
pip install -e pkgs/neuralpredictors pkgs/nnfabrik pkgs/featurevis pkgs/sensorium
```

Create `.env` file in the root directory according to `.env.example` file and make sure to set the path to an existing directory where the data will reside (`DATA_PATH`). You might need to load the environment variable(s) from the `.env` file manually in the terminal: `export $(cat .env | xargs)`

### Potential issues
If you encounter the error `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`, run the following within the activated `csng` environment:
```bash
pip install --force-reinstall -v "numpy==1.25.2"
```


## Directory structure
- `README.md` - This file
- `setup.py` - Setup file for the `csng` package
- `environment.yaml` - Environment file with all the dependencies
- `.env.example` - Example of the `.env` file. Important to setup your own .env file in the same directory to be able to run the scripts
- `.gitignore` - Git ignore file
- `pkgs` - Directory containing modified packages `neuralpredictors`, `nnfabrik`, `featurevis`, and `sensorium`
- `csng` - Directory containing the main code for the project
  - `cat_v1/` - Directory with code specific to the cat V1 data (**C**)
  - `mouse_v1/` - Directory with code specific to the SENSORIUM 2022 mouse V1 data (datasets **M-\<mouse id\>** and **M-All**)
  - `brainreader_mouse/` - Directory with code specific to the mouse V1 data from [Cobos E. et al. 2022](https://doi.org/10.1101/2022.12.09.519708) (datasets **B-\<mouse id\>** and **B-All**)
  - `<your-data>/` - Directory with code specific to your data (e.g., `cat_v1/`). This folder should include a dataloading utility that could be then combined with other datasets using the code in `csng/data.py`.
