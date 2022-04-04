import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import torch
from torch_geometric.data import DataLoader
from rdkit import Chem
from sklearn.model_selection import train_test_split
import copy
import warnings
warnings.filterwarnings(action='ignore')

from smiles_to_graph import mol2vec
from smiles_to_graph import make_regre_mol
from smiles_to_graph import make_regre_vec
from regre_exp import experiment
import regre_model

parser = argparse.ArgumentParser(
    description="Graph Convolutional Network for logS Regression", 
    epilog="python regression.py -D './myDataset.xlsx' -X1 'Solute SMILES' -X2 'Solvent SMILES' -Y 'LogS' -O './results/myResult.json' -M './results/myModel.pt'")
parser.add_argument('--seed', '-s', type=int, default=123, help='seed')
parser.add_argument('--input_path', '-D', type=str, required=True, help="dataset path and name ('./dataset.xlsx')")
parser.add_argument('--solute_smiles', '-X1', type=str, required=True, help="column name of solute smiles ('Solute SMILES')")
parser.add_argument('--solvent_smiles', '-X2', type=str, required=True, help="column name of solvent smiles ('Solvent SMILES')")
parser.add_argument('--logS', '-Y', type=str, required=True, help="column name of logS ('LogS')")
parser.add_argument('--output_path', '-O', type=str, default='./results/result.json', 
                    help="output path and name (defualt='./results/result.json')")
parser.add_argument('--model_path', '-M', type=str, default='./results/model.pt', help="model path and name ('./results/model.pt')")
parser.add_argument('--conv', '-c', type=str, default='GCNConv', choices=['GCNConv', 'ARMAConv', 'SAGEConv'], 
                    help='GCNConv/ARMAConv/SAGEConv (defualt=GCNConv)')
parser.add_argument('--test_size', '-z', type=float, default=0.2, help='test size (defualt=0.2)')
parser.add_argument('--random_state', '-r', type=int, default=123, help='random state')
parser.add_argument('--batch_size', '-b', type=int, default=256, help='batch size (defualt=256)')
parser.add_argument('--epoch', '-e', type=int, default=200, help='epoch (defualt=200)')
parser.add_argument('--lr', '-l', type=float, default=0.005, help='learning rate (defualt=0.005)')
parser.add_argument('--step_size', '-t', type=int, default=5, help='step_size of lr_scheduler (defualt=5)')
parser.add_argument('--gamma', '-g', type=float, default=0.9, help='gamma of lr_scheduler (defualt=0.9)')
parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout (defualt=0.1)')
parser.add_argument('--exp_name', '-n', type=str, default='myExp', help='experiment name')
args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print()
print('Graph Convolutional Network for logS Regression')
print('Soongsil University, Seoul, South Korea')
print('Computational Science and Artificial Intelligence Lab')
print()
print('[Preparing Data]')
print('- Device :', device)
print()


df = pd.read_csv(args.input_path)
df = pd.concat([df[args.solute_smiles], df[args.solvent_smiles], df[args.logS]], axis=1)
df.columns = ['Solute SMILES', 'Solvent SMILES', 'logS']
df = df.dropna(axis=0).reset_index(drop=True)

X_train, X_test = train_test_split(df, test_size=args.test_size, random_state=args.random_state)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

print('[Converting to Graph]')
train_mols1_key, train_mols2_key, train_mols_value = make_regre_mol(X_train)
test_mols1_key, test_mols2_key, test_mols_value = make_regre_mol(X_test)

train_X1, train_X2 = make_regre_vec(train_mols1_key, train_mols2_key, train_mols_value)
test_X1, test_X2 = make_regre_vec(test_mols1_key, test_mols2_key, test_mols_value)

train_X = []
for i in range(len(train_X1)):
    train_X.append([train_X1[i], train_X2[i]])
test_X = []
for i in range(len(test_X1)):
    test_X.append([test_X1[i], test_X2[i]])

print('- Train Data :', len(train_X))
print('- Test Data :', len(test_X))


train_loader = DataLoader(train_X, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_X, batch_size=len(test_X), shuffle=True, drop_last=True)

model = regre_model.Net(args)
model = model.to(device)

print()
dict_result = dict()
result = vars(experiment(model, train_loader, test_loader, device, args))
dict_result[args.exp_name] = copy.deepcopy(result)
result_df = pd.DataFrame(dict_result).transpose()
result_df.to_json(args.output_path, orient='table')


train_loss = result_df['list_train_loss'].iloc[0]
logS_total = result_df['logS_total'].iloc[0]
pred_logS_total = result_df['pred_logS_total'].iloc[0]

plt.rcParams["figure.figsize"] = (10, 6)
plt.suptitle(args.exp_name, fontsize=16)

plt.subplot(1, 2, 1)
plt.ylim([0, 10])
plt.plot([e for e in range(len(train_loss))], [float(t) for t in train_loss], label="train_loss", c='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
mae_test = 'MAE : ' + str(round(float(result_df['mae'].iloc[0]), 2))
mse_test = 'MSE : ' + str(round(float(result_df['mse'].iloc[0]), 2))
r_test = 'R2 : ' + str(round(float(result_df['r_square'].iloc[0]), 2))
plt.text(0, -1.5, mae_test, fontsize=12)
plt.text(0, -2, mse_test, fontsize=12)
plt.text(0, -2.5, r_test, fontsize=12)

plt.subplot(1, 2, 2)
plt.scatter(logS_total, pred_logS_total, alpha=0.4)
plt.plot(logS_total, logS_total, alpha=0.4, color='black')
plt.xlabel("logS_total")
plt.ylabel("pred_logS_total")

plt.tight_layout()
plt.savefig(str(args.output_path[:-5])+'.png')
print()


