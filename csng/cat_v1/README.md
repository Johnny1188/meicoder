## Data
The data can be obtained from [this data repository](https://gin.g-node.org/lucabaroni/LSV1M_Dec23/src/master) (from [Computational Systems Neuroscience Group](https://csng.mff.cuni.cz)).

Download the `.zip` files called `50K_single_trial_dataset.zip` and `Dataset_multitrial.zip` to `<DATA_PATH>/cat_V1_spiking_model` and perform the following steps in this directory:
  - Extract the training and validation data: `unzip 50K_single_trial_dataset.zip`
  - Move the extracted files: `mkdir -p 50K_single_trial_dataset && mv CSNG/baroni/Dic23data/* 50K_single_trial_dataset/`
  - Extract the test data: `unzip Dataset_multitrial.zip -d 50K_single_trial_dataset/Dataset_multitrial`
  - Run the Jupyter notebook `data_50k_prep.ipynb` to prepare the dataset
    - The first two sections of the notebook will prepare the data and place them in `<DATA_PATH>/cat_V1_spiking_model/50K_single_trial_dataset/datasets/<train|val|test>` directories
    - The last section of the notebook titled "Get data statistics" will generate the statistics of the dataset and save them.
  