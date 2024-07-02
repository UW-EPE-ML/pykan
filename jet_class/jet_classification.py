from kan import KAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import numpy as np
import os
from tqdm import tqdm

from util.dataloader import read_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data and concatenate the features
x_pars, x_jets, ys = [], [], []

particle_features = [
    'part_px', 'part_py', 'part_pz', 'part_energy', 'part_deta', 'part_dphi', 'part_d0val',
    'part_d0err', 'part_dzval', 'part_dzerr', 'part_charge'
]
max_num_particles = 10
num_features_particles = len(particle_features)
n_features = max_num_particles * num_features_particles
output_label = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
    'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl'
]

for jet_data in tqdm(os.listdir('data')):
    if jet_data.endswith('.root') and '120' in jet_data:
        x_particles, x_jet, y = read_file(
            'data/' + jet_data,
            max_num_particles=10,
            particle_features=particle_features,
            labels=output_label,
        )
        x_pars.append(x_particles.reshape(-1, n_features))
        x_jets.append(x_jet)

        # convert one-hot label back
        ys.append(np.argmax(y, axis=1))

x_pars = np.concatenate(x_pars)
x_jets = np.concatenate(x_jets)
ys = np.concatenate(ys)

# shuffle the data
idx = np.arange(len(x_jets))
np.random.shuffle(idx)
x_pars = x_pars[idx]
x_jets = x_jets[idx]
ys = ys[idx]  # .astype(float)

N = len(x_pars)
dataset = {}
dataset['train_input'] = torch.tensor(x_pars[:int(N / 2)]).to(device)
dataset['train_label'] = torch.tensor(ys[:int(N / 2)]).to(device)
dataset['test_input'] = torch.tensor(x_pars[int(N / 2) + 1:]).to(device)
dataset['test_label'] = torch.tensor(ys[int(N / 2) + 1:]).to(device)

# create a KAN: 4-D inputs, 10-D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[n_features, len(output_label)], grid=3, k=3, device=device)


# model.update_grid_from_samples(dataset['train_input'].to(device))

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())


def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())


results = model.train(
    dataset,
    opt="LBFGS",
    steps=50,
    metrics=(train_acc, test_acc),
    loss_fn=torch.nn.CrossEntropyLoss().to(device),
    lamb_coef=1e-3,
    batch=256,
    device=device,
)
print(results['train_acc'][-1], results['test_acc'][-1])
torch.save(results, 'results.pt')

model.save_ckpt('model.ckpt')
