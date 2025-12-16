## Brainreader mouse data
We do not have permission to redistribute the Brainreader data. Please contact the [Andreas Tolias Lab](https://toliaslab.org/contact/).

Place the obtained data (files named `<session>_<data_part>_images` and `<session>_<data_part>_resp`) in the main data folder `<DATA_PATH>`. Finaly, run the `notebooks/preprocess_data.ipynb` notebook to rearrange the data and to compute the data statistics. These steps will create a folder structure like this:
```
<DATA_PATH>/brainreader
|   ├── data
|   │   ├── 1
|   │   │   ├── stats
|   |   │   │   ├── responses_iqr.npy
|   |   │   │   ├── responses_median.npy
|   |   │   │   ├── responses_mean.npy
|   |   │   │   ├── responses_std.npy
|   |   │   │   ├── stimuli_mean.npy
|   |   │   │   └── stimuli_std.npy
|   │   │   ├── train  ## contains individual data points as .pickle files
|   │   │   ├── test
|   │   │   └── val
|   │   ├── 2  ## same structure as session 1
|   │   ├── ...
|   │   └── 22 ## same structure as session 1
```
