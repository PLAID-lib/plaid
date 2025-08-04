To run this benchmark, you need to download and untar the dataset from Zenodo, then edit in the files "morphing_script.py" and "train_and_predict.py" the following variables at the top of the files:

- `plaid_location`: the location where the plaid dataset has been untarred

Run in the order: "create_coarse_common_mesh.py", "launch_morphings.py", and "train_and_predict.py" to generate the prediction on the testing set.

Remark: the loop in "launch_morphings.py" is embarrasingly parallel, and the calls to "morphing_script.py" can be efficiently handled by any job scheduler.
