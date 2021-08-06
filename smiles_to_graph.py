import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
 
def get_intervals(l):
    """For list of lists, gets the cumulative products of the lengths"""
    intervals = len(l) * [0]
    # Initalize with 1
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]
    return intervals
 

def safe_index(l, e):
    """Gets the index of e in l, providing an index of len(l) if not found"""
    try:
        return l.index(e)
    except:
        return len(l)
 
 
possible_atom_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 
                      'Fe', 'Ca', 'Cu', 'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn']
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                               Chem.rdchem.HybridizationType.SP3D2]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']

reference_lists = [possible_atom_list, possible_numH_list, possible_valence_list,
                   possible_formal_charge_list, possible_number_radical_e_list,
                   possible_hybridization_list, possible_chirality_list]
 
intervals = get_intervals(reference_lists)
 
def get_feature_list(atom):
    features = 6 * [0]
    features[0] = safe_index(possible_atom_list, atom.GetSymbol())
    features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
    features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
    features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
    features[4] = safe_index(possible_number_radical_e_list,atom.GetNumRadicalElectrons())
    features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
    return features
 
def features_to_id(features, intervals):
    """Convert list of features into index using spacings provided in intervals"""
    id = 0
    for k in range(len(intervals)):
        id += features[k] * intervals[k]
    # Allow 0 index to correspond to null molecule 1
    id = id + 1
    return id

def id_to_features(id, intervals):
    features = 6 * [0]
 
    # Correct for null
    id -= 1
 
    for k in range(0, 6 - 1):
        # print(6-k-1, id)
        features[6 - k - 1] = id // intervals[6 - k - 1]
        id -= features[6 - k - 1] * intervals[6 - k - 1]
    # Correct for last one
    features[0] = id
    return features
 
def atom_to_id(atom):
    """Return a unique id corresponding to the atom type"""
    features = get_feature_list(atom)
    return features_to_id(features, intervals)
 
def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        from rdkit import Chem
        results = np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C','N','O','S','F','Si','P','Cl','Br','Mg',
                                                                   'Na','Ca','Fe','As','Al','I','B','V','K','Tl',
                                                                   'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H',  # H?
                                                                   'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr',
                                                                   'Pt','Hg','Pb','Unknown']) + 
                           one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 
                           one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + 
                           [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + 
                           one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, 
                                                                           Chem.rdchem.HybridizationType.SP2, 
                                                                           Chem.rdchem.HybridizationType.SP3, 
                                                                           Chem.rdchem.HybridizationType.SP3D, 
                                                                           Chem.rdchem.HybridizationType.SP3D2]) + 
                           [atom.GetIsAromatic()])
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = np.array(results.tolist() + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4]))
    if use_chirality:
        try:
            results = np.array(results.tolist() + 
                               one_of_k_encoding_unk(atom.GetProp('_CIPCode'),['R', 'S']) + 
                               [atom.HasProp('_ChiralityPossible')])
        except:
            results = np.array(results.tolist() + 
                               [False, False] + 
                               [atom.HasProp('_ChiralityPossible')])
 
    return np.array(results)
 
def bond_features(bond, use_chirality=False):
    from rdkit import Chem
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                  bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                  bond.GetIsConjugated(),
                  bond.IsInRing()]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(str(bond.GetStereo()),
                                                        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)
 
def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def mol2vec(mol):
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features(atom) for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]
    for bond in bonds:
        edge_attr.append(bond_features(bond))
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    return data

def make_regre_mol(df):
    mols1_key = []
    mols2_key = []
    mols_value = []
    for i in range(df.shape[0]):
        mols1_key.append(Chem.MolFromSmiles(df['Solute SMILES'].iloc[i]))
        mols2_key.append(Chem.MolFromSmiles(df['Solvent SMILES'].iloc[i]))
        mols_value.append(df['logS'].iloc[i])
    return mols1_key, mols2_key, mols_value

def make_class_mol(df):
    mols1_key = []
    mols2_key = []
    mols_value = []
    for i in range(df.shape[0]):
        mols1_key.append(Chem.MolFromSmiles(df['Solute SMILES'].iloc[i]))
        mols2_key.append(Chem.MolFromSmiles(df['Solvent SMILES'].iloc[i]))
        mols_value.append(df['Class'].iloc[i])
    return mols1_key, mols2_key, mols_value

def make_regre_vec(mols1, mols2, value):
    X1 = []
    X2 = []
    Y = []
    for i in range(len(mols1)):
        m1 = mols1[i]
        m2 = mols2[i]
        y = value[i]
        try:
            X1.append(mol2vec(m1))
            X2.append(mol2vec(m2))
            Y.append(y)
        except:
            continue
    for i, data in enumerate(X1):
        y = Y[i]
        data.y = torch.tensor([y], dtype=torch.float)
    for i, data in enumerate(X2):
        y = Y[i]
        data.y = torch.tensor([y], dtype=torch.float)
    return X1, X2

def make_class_vec(mols1, mols2, value):
    X1 = []
    X2 = []
    Y = []
    for i in range(len(mols1)):
        m1 = mols1[i]
        m2 = mols2[i]
        y = value[i]
        try:
            X1.append(mol2vec(m1))
            X2.append(mol2vec(m2))
            Y.append(y)
        except:
            continue
    for i, data in enumerate(X1):
        y = Y[i]
        data.y = torch.tensor([y], dtype=torch.long)
    for i, data in enumerate(X2):
        y = Y[i]
        data.y = torch.tensor([y], dtype=torch.long)
    return X1, X2
