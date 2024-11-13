# Directory structure
- `README.md` - This file
- `LICENSE` - License file
- `setup.py` - Setup file for the `csng` package
- `environment.yaml` - Environment file with all the dependencies
- `.env.example` - Example of the `.env` file
- `.gitignore` - Git ignore file
- `pkgs` - Directory containing modified packages `neuralpredictors`, `nnfabrik`, `featurevis`, and `sensorium`
- `csng` - Directory containing the main code for the project
  - `cat_v1/` - Directory for experiments on the cat V1 data (dataset **C**)
  - `mouse_v1/` - Directory for experiments on the SENSORIUM 2022 mouse V1 data (datasets **M-1** and **M-All**)
  - `brainreader_mouse_v1/` - Directory for experiments on the mouse V1 data from [Cobos E. et al. 2022](https://doi.org/10.1101/2022.12.09.519708) (datasets **M-1** and **M-All**)

# Environment
- The code is only tested on Linux (Ubuntu 22.04) with Nvidia GPU (CUDA 12.2) and with the provided environment setup. Other setups, for example CPU-only, is not guaranteed to work. Depending on the selected batch size, most of the experiments can be run with under 4GB of GPU memory.
- Setup an environment from the `environment.yaml` file:
  - `conda env create -f environment.yaml` ([Miniconda](https://docs.anaconda.com/free/miniconda/index.html) is recommended for managing environments)
  - And activate it: `conda activate csng` (`csng` is the name of the environment)
- Pip install the main `csng` package and all the modified packages in the `pkgs` directory:
  - `pip install -e .` (`csng` package)
  - `pip install -e pkgs/neuralpredictors pkgs/nnfabrik pkgs/featurevis pkgs/sensorium`
  - The `csng` package is the developed code for this project, the other are modified versions of existing packages to work with Python 3.10 or to include additional features. Specifically:
    - Modified version of [neuralpredictors](https://github.com/sinzlab/neuralpredictors) (used only by the encoder implementation from SENSORIUM 2022)
    - Modified version of [nnfabrik](https://github.com/sinzlab/nnfabrik) (used only by the encoder implementation from SENSORIUM 2022)
    - Modified version of [featurevis](https://github.com/ecobost/featurevis) (used for MEI generation)
    - Modified version of [sensorium](https://github.com/sinzlab/sensorium) (used for its encoder implementation and utilities for getting the mouse V1 data)
- Create `.env` file in the root directory according to `.env.example` file and make sure to set the path to an existing directory where the data will reside (`DATA_PATH`)
- You might need to load the environment variable(s) from the `.env` file manually in the terminal: `export $(cat .env | xargs)`
- Instructions for obtaining the data and running the code for the two experiments are in the respective README files in the `cat_v1_spiking_model/` and `mouse_v1_sensorium22/` directories
