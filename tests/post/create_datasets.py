from plaid import Sample, Dataset
import numpy as np


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
    sample.add_scalar("feature_1", 1.00001*ins[i])
    sample.add_scalar("feature_2", 1.00001*outs[i])
    samples.append(sample)

dataset = Dataset(samples=samples)
dataset._save_to_dir_(path="dataset_near_pred", verbose=True)


samples = []
for i in range(30):
    sample = Sample()
    sample.add_scalar("feature_1", 0.5*ins[i])
    sample.add_scalar("feature_2", 0.5*outs[i])
    samples.append(sample)

dataset = Dataset(samples=samples)
dataset._save_to_dir_(path="dataset_pred", verbose=True)


print("dataset =", dataset)
print(dataset[0].get_scalar("feature_1"))