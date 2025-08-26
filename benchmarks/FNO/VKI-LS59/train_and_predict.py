from plaid.containers.dataset import Dataset
from plaid.problem_definition import ProblemDefinition
import numpy as np

import os, shutil
import time

import torch
from physicsnemo.models.fno.fno import FNO

start = time.time()


plaid_location = # path to update
prepared_data_dir = # path to update
predicted_data_dir= # path to update



pb_defpath=os.path.join(plaid_location, "problem_definition")

dataset = Dataset()
dataset._load_from_dir_(os.path.join(prepared_data_dir, "dataset"), verbose=True, processes_number=4)

problem = ProblemDefinition()
problem._load_from_dir_(pb_defpath)

ids_train = problem.get_split('train')
ids_test  = problem.get_split('test')


n_train = len(ids_train)
n_test  = len(ids_test)


in_scalars_names = ['angle_in', 'mach_out']
out_fields_names = ['mach', 'nut']
out_scalars_names = ['Q', 'power', 'Pr', 'Tr', 'eth_is', 'angle_out']


size1 = 301
size2 = 121


# TRAIN

inputs = np.empty((n_train, len(in_scalars_names)+1, size1, size2))
for i, id_sample in enumerate(ids_train):
    for in_chan in range(len(in_scalars_names)+1):
        inputs[i, in_chan, :, :] = dataset[id_sample].get_field("Signed_Distance", base_name="Base_2_2").reshape((size1, size2))
        for k, sn in enumerate(in_scalars_names):
            inputs[i, k+1, :, :] = dataset[id_sample].scalars.get(sn)

outputs = np.empty((n_train, len(out_scalars_names)+len(out_fields_names), size1, size2))
for i, id_sample in enumerate(ids_train):
    for k, fn in enumerate(out_fields_names):
        outputs[i, k, :, :] = dataset[id_sample].get_field(fn, base_name="Base_2_2").reshape((size1, size2))
    for k, sn in enumerate(out_scalars_names):
        outputs[i, k+len(out_fields_names), :, :] = dataset[id_sample].scalars.get(sn)


min_in = inputs.min(axis=(0, 2, 3), keepdims=True)
max_in = inputs.max(axis=(0, 2, 3), keepdims=True)
inputs = (inputs - min_in) / (max_in - min_in)


min_out = outputs.min(axis=(0, 2, 3), keepdims=True)
max_out = outputs.max(axis=(0, 2, 3), keepdims=True)
outputs = (outputs - min_out) / (max_out - min_out)


import torch
from torch.utils.data import Dataset

class GridDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

from torch.utils.data import DataLoader
dataset__ = GridDataset(inputs, outputs)
loader = DataLoader(dataset__, batch_size=64, shuffle=True)



model = FNO(
in_channels=inputs.shape[1],
out_channels=outputs.shape[1],
decoder_layers=4,
decoder_layer_size=64,
dimension=2,
latent_channels=64,
num_fno_layers=4,
padding=0,
).cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

n_epoch = 2000
for epoch in range(n_epoch):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")



# TEST

inputs = np.empty((n_test, len(in_scalars_names)+1, size1, size2))

for i, id_sample in enumerate(ids_test):
    for in_chan in range(len(in_scalars_names)+1):
        inputs[i, in_chan, :, :] = dataset[id_sample].get_field("Signed_Distance", base_name="Base_2_2").reshape((size1, size2))
        for k, sn in enumerate(in_scalars_names):
            inputs[i, k+1, :, :] = dataset[id_sample].scalars.get(sn)

inputs = (inputs - min_in) / (max_in - min_in)


model.eval()
with torch.no_grad():
    x_test = torch.tensor(inputs, dtype=torch.float32).cuda()
    y_pred = model(x_test).cpu().numpy()

outputs_pred = y_pred * (max_out - min_out) + min_out


for i, id_sample in enumerate(ids_test):
    for k, fn in enumerate(out_fields_names):
        dataset[id_sample].add_field(fn, outputs_pred[i, k, :, :].flatten())
    for k, sn in enumerate(out_scalars_names):
        dataset[id_sample].scalars.add(sn, np.mean(outputs_pred[i, k+len(out_fields_names), :, :].flatten()))


if os.path.exists(predicted_data_dir) and os.path.isdir(predicted_data_dir):
    shutil.rmtree(predicted_data_dir)
dataset[ids_test]._save_to_dir_(predicted_data_dir)


print("duration train =", time.time()-start)
# GPUA30, 9344 seconds