import pandas as pd
import argparse
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim

from smiles_to_graph import mol2vec
from smiles_to_graph import make_regre_mol
from smiles_to_graph import make_regre_vec
import regre_model


parser = argparse.ArgumentParser(
    description="Graph Convolutional Network for logS Regression", 
    epilog="python regre_test.py -M './results/model.pt' -I './smiles.txt' -O './results/pred_results.txt'")
parser.add_argument('--model_path', '-M', type=str, default='./results/model.pt', help="model path and name ('./results/model.pt')")
parser.add_argument('--input_path', '-I', type=str, default='./smiles.txt', help="input path and name ('./smiles.txt')")
parser.add_argument('--output_path', '-O', type=str, default='./results/pred_results.txt', help="output path and name ('./results/pred_results.txt')")
parser.add_argument('--conv', '-c', type=str, default='GCNConv', choices=['GCNConv', 'ARMAConv', 'SAGEConv'], 
                    help='GCNConv/ARMAConv/SAGEConv (defualt=GCNConv)')
parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout (defualt=0.1)')
args = parser.parse_args()              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print()
print('Graph Nonvolutional Network for Regression logS')
print('Soongsil University, Seoul, South Korea')
print('Computational Science and Artificial Intelligence Lab')
print()

def test_only(model, device, data_test, args):
    model.eval()
    with torch.no_grad():
        for i, [solute, solvent] in enumerate(test_loader):
            solute = solute.to(device)
            solvent = solvent.to(device)
            output = model(solute, solvent, device)
    return output

test_df = pd.read_csv(args.input_path)
test_df['logS'] = 0

test_mols1_key, test_mols2_key, test_mols_value = make_regre_mol(test_df)

test_X1, test_X2 = make_regre_vec(test_mols1_key, test_mols2_key, test_mols_value)

test_X = []
for i in range(len(test_X1)):
    test_X.append([test_X1[i], test_X2[i]])

test_loader = DataLoader(test_X, batch_size=len(test_X))

model = regre_model.Net(args)
model = model.to(device)

optimizer = optim.Adam(model.parameters())

checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['State_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

test_result = test_only(model, device, test_loader, args)
test_df['logS'] = test_result.cpu().numpy()
test_df.to_csv(args.output_path, sep='\t', index=False)

print('Done!')
print()


