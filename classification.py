import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import argparse
import torch
from torch_geometric.data import DataLoader
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc
import copy
import pickle
import warnings
warnings.filterwarnings(action='ignore')

from smiles_to_graph import mol2vec
from smiles_to_graph import make_class_mol
from smiles_to_graph import make_class_vec
from class_exp import experiment
import class_model

parser = argparse.ArgumentParser(
    description="Graph Convolutional Network for logS Classification", 
    epilog="python classification.py -D './myDataset.xlsx' -X1 'Solute SMILES' -X2 'Solvent SMILES' -Y 'LogS' -O './results/myResult.json' -M './results/myModel.pt'")
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
parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch (defualt=100)')
parser.add_argument('--lr', '-l', type=float, default=0.0005, help='learning rate (defualt=0.005)')
parser.add_argument('--step_size', '-t', type=int, default=5, help='step_size of lr_scheduler (defualt=5)')
parser.add_argument('--gamma', '-g', type=float, default=0.9, help='gamma of lr_scheduler (defualt=0.9)')
parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout (defualt=0.1)')
parser.add_argument('--exp_name', '-n', type=str, default='myExp', help='experiment name')
parser.add_argument('--ROC', '-ROC', action='store_true', help='save ROC curve result (defualt=False)')
args = parser.parse_args()              


np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print()
print('Graph Convolutional Network for logS Classification')
print('Soongsil University, Seoul, South Korea')
print('Computational Science and Artificial Intelligence Lab')
print()
print('[Preparing Data]')
print('- Device :', device)
print()
      

def df_check(df):
    df['Class'] = np.nan
    for i in range(df.shape[0]):
        logs = df['logS'].iloc[i]
        if logs <= -3:
            df['Class'].iloc[i] = 0
        elif logs > -3:
            if logs <= -1:
                df['Class'].iloc[i] = 1
            if logs > -1:
                df['Class'].iloc[i] = 2
    return df


df = pd.read_csv(args.input_path)
df = pd.concat([df[args.solute_smiles], df[args.solvent_smiles], df[args.logS]], axis=1)
df.columns = ['Solute SMILES', 'Solvent SMILES', 'logS']
df = df.dropna(axis=0).reset_index(drop=True)
df = df_check(df)

X_train, X_test = train_test_split(df, test_size=args.test_size, random_state=args.random_state)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

print('[Converting to Graph]')
train_mols1_key, train_mols2_key, train_mols_value = make_class_mol(X_train)
test_mols1_key, test_mols2_key, test_mols_value = make_class_mol(X_test)

train_X1, train_X2 = make_class_vec(train_mols1_key, train_mols2_key, train_mols_value)
test_X1, test_X2 = make_class_vec(test_mols1_key, test_mols2_key, test_mols_value)

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

model = class_model.Net(args)
model = model.to(device)

print()
dict_result = dict()
result = vars(experiment(model, train_loader, test_loader, device, args))
dict_result[args.exp_name] = copy.deepcopy(result)
result_df = pd.DataFrame(dict_result).transpose()
result_df.to_json(args.output_path, orient='table')


classes = ('Low', 'Medium', 'High')
df = pd.read_json(args.output_path, orient='table')
plt.rcParams["figure.figsize"] = (12, 12)
plt.suptitle(args.exp_name, fontsize=16)

train_loss = df['list_train_loss'].iloc[0]
train_acc = df['list_train_acc'].iloc[0]
c = df['conf'].iloc[0]
confusion_matrix = np.array([c[0], c[1], c[2]], dtype=float)
df_cm = pd.DataFrame(confusion_matrix, range(len(classes)), range(len(classes)))
accuracy = df['total_acc'].iloc[0]
accuracy_1 = df['low'].iloc[0]
accuracy_2 = df['medium'].iloc[0]
accuracy_3 = df['high'].iloc[0]
acc_list = [accuracy, accuracy_1, accuracy_2, accuracy_3]
y_true = df['logS_total'].iloc[0]
y_pred = df['pred_logS_total'].iloc[0]
miscore = df['miscore'].iloc[0]
mascore = df['mascore'].iloc[0]
plt.subplot(2, 2, 1)
plt.ylim([0, 1])
plt.plot([e for e in range(len(train_loss))], [float(t) for t in train_loss], label="train_loss", c='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(2, 2, 2)
plt.ylim([0, 1])
plt.plot([e for e in range(len(train_acc))], [float(t)*0.01 for t in train_acc], label="train_acc", c='blue')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(2, 2, 3)
sn.heatmap(df_cm.astype('int'), annot=True, cmap='Blues', fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.subplot(2, 2, 4)
plt.xlabel('Accuracy (%)')
plt.ylabel("Class")
barlist = plt.barh(['Total', 'Low', 'Medium', 'High'], acc_list, height=0.4, color='cornflowerblue')
barlist[0].set_color('mediumblue')
for i, v in enumerate(acc_list):
    plt.text(v-12, i-0.04, str(round(v,2)), color='white', fontweight='bold')
acc1_test = 'Accuracy : ' + str(round(float(accuracy), 2))
acc2_test = 'Low : ' + str(round(float(accuracy_1), 2))
acc3_test = 'Medium : ' + str(round(float(accuracy_2), 2))
acc4_test = 'High : ' + str(round(float(accuracy_3), 2))
mi_test = 'F-1 Micro : ' + str(round(float(miscore), 2))
ma_test = 'F-1 Macro : ' + str(round(float(mascore), 2))
plt.text(0, -0.85, acc1_test, fontsize=12)
plt.text(0, -1.00, acc2_test, fontsize=12)
plt.text(0, -1.15, acc3_test, fontsize=12)
plt.text(0, -1.3, acc4_test, fontsize=12)
plt.text(0, -1.45, mi_test, fontsize=12)
plt.text(0, -1.6, ma_test, fontsize=12)
plt.tight_layout()
plt.savefig(str(args.output_path[:-5])+'.png')
plt.clf()

if args.ROC:
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.clf()
    with open(str(args.output_path[:-5])+'.pickle', 'rb') as fr:
        result_loaded = pickle.load(fr)
        fig, axes = plt.subplots(nrows=1, ncols=2)
    for i in result_loaded:
        y_test = np.array(result_loaded[i]['y_test'])
        y_score = np.array(result_loaded[i]['y_score'])
        lw = 2
        n_classes = len(classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.suptitle(args.exp_name, fontsize=16)
        axes[0].plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        axes[0].plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        colors = (['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            axes[0].plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        axes[0].plot([0, 1], [0, 1], c='black')
        axes[0].axis(xmin=0, xmax=1, ymin=0, ymax=1.05)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc="lower right")
        axes[1].plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        axes[1].plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        axes[1].plot([0, 1], [0, 1], c='black')
        axes[1].axis(xmin=0, xmax=1, ymin=0, ymax=1.05)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        fig.tight_layout()
        plt.savefig(str(args.output_path[:-5])+'_ROC.png')

print()

