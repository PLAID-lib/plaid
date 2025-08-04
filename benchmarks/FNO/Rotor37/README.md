To run this benchmark, you need to download and untar the dataset from Zenodo, then edit in the files "prepare_[dataset].py", "train.py" and "construct_prediction.py" the following variables at the top of the files:

- `plaid_location`: the location where the plaid dataset has been untarred
- `prepared_data_dir`: temp folder used by the scripts, for the dataset projected onto a regular grid
- `predicted_data_dir`: temp folder used by the scripts, for the prediction of the test set onto the regular grid

Run in the order: "prepare_[dataset].py", "train.py" and "construct_prediction.py" to generate the prediction on the testing set.