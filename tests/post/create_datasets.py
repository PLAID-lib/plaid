import numpy as np

from plaid import Dataset, ProblemDefinition, Sample
from plaid.types import FeatureIdentifier
from plaid.utils.split import split_dataset

ins = []
outs = []
for i in range(30):
    ins.append(np.random.rand())
    outs.append(np.random.rand())

samples = []
for i in range(30):
    sample = Sample()
    sample.add_scalar("feature_1", ins[i])
    sample.add_scalar("feature_2", outs[i])
    samples.append(sample)

dataset = Dataset(samples=samples)
dataset._save_to_dir_(path="dataset_ref", verbose=True)


samples = []
for i in range(30):
    sample = Sample()
    sample.add_scalar("feature_1", 1.00001 * ins[i])
    sample.add_scalar("feature_2", 1.00001 * outs[i])
    samples.append(sample)

dataset = Dataset(samples=samples)
dataset._save_to_dir_(path="dataset_near_pred", verbose=True)


samples = []
for i in range(30):
    sample = Sample()
    sample.add_scalar("feature_1", 0.5 * ins[i])
    sample.add_scalar("feature_2", 0.5 * outs[i])
    samples.append(sample)

dataset = Dataset(samples=samples)
dataset._save_to_dir_(path="dataset_pred", verbose=True)


print("dataset =", dataset)
print(dataset[0].get_scalar("feature_1"))


pb_def = ProblemDefinition()

scalar_1_feat_id = FeatureIdentifier({"type": "scalar", "name": "feature_1"})
scalar_2_feat_id = FeatureIdentifier({"type": "scalar", "name": "feature_2"})

pb_def.add_in_feature_identifier(scalar_1_feat_id)
pb_def.add_out_feature_identifier(scalar_2_feat_id)

pb_def.add_input_scalar_name("feature_1")
pb_def.add_output_scalar_name("feature_2")

pb_def.set_task("regression")

options = {
    "shuffle": False,
    "split_sizes": {
        "train": 20,
        "test": 10,
    },
}

split = split_dataset(dataset, options)
print(f"{split = }")

pb_def.set_split(split)

pb_def._save_to_dir_("problem_definition")
