To run this benchmark, you need to download and untar the dataset from Zenodo, then edit in the files "convert_to_rectilinear_grid.py", "train.py" and "build_pred.py" the following variables at the top of the files:

- `dataset_path`: the location where the plaid dataset has been untarred
- `rect_dataset_path`: temp folder used by the scripts, for the dataset projected onto a regular grid

Run in the order: "convert_to_rectilinear_grid.py", "train.py" and "build_pred.py" to generate the prediction on the testing set.
