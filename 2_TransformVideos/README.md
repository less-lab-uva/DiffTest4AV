# Transforming data
In order for the AVs and SUTs set up in `0_Setup` to run on the datasets from `1_Datasets`, all of the input videos must be preprocessed into a common format.
This folder contains the scripts to handle this process.
Each script must be run separately after the datasets are set up.
See the README in `1_Datasets` for more information on the separate datasets.

This will process the original data into the following folders `1_Datasets/Data`:
* `OpenPilot_2016/1_ProcessedData/`
* `OpenPilot_2k19/1_ProcessedData/`
* `External_jutah/1_ProcessedData/`

These folders have been added to the git repository by including a `delete_me.md` placeholder.
Please delete this file in each folder before continuing.