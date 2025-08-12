The code to run this benchmark can be retrieved at https://gitlab.com/drti/mmgp. More precisely, it is located in examples/Tensile2d.

The file `configuration.yml` must be eddited to indicate the location of the untarred dataset at `init_dataset_location`, and `number_Monte_Carlo_samples` can be lowered to 2, since predictive uncertainty is not of interest for this benchmark. Beside, both `number_of_modes` are set to 16. Then run:

```python
python run.py --preprocess --train --infer --export_predictions
```

The prediction will be generated in a folder named `Tensile2d_predicted` located in a folder configurated under `generated_data_folder`. Finally, the prediction file can be generated using `construct_prediction.py` (locations at the top of the file must be set).

### List of dependencies

- [PLAID=0.1.6](https://github.com/PLAID-lib/plaid)
- [MMGP=0.0.9](https://gitlab.com/drti/mmgp)