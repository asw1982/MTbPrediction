# -*- coding: utf-8 -*-
# import packages
# general tools
import pandas as pd
import numpy as np
# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# Pytorch and Pytorch Geometric
import torch

import torch.nn as nn
from torch.nn import Linear
import torch.optim as optim
import torch.nn.functional as F # activation function
from torch.utils.data import Dataset, DataLoader # dataset management
import torchvision.datasets as datasets #bank of dataset
import torchvision.transforms as transforms #can create pipeline to preprocess da

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as G_Loader 
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_curve, auc 
from sklearn.metrics import precision_recall_curve

import statistics
#from prettytable import PrettyTable

# performances visualization 
#import matplotlib.pyplot as plt

# PREPARE THE DATASET 
# DATASETS ARE SEPARATED INTO 3 DATA : DATA_TRAIN, DATA_VALIDATION, DATA_TEST (INDEPENDENT DATA_SET)

# THESE ARE MODULES USED TO GENERATE GRAPH STRUCTURED DATASET 
#==============================================================================================================================
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    list_smiles_error= []
    num_error = 0
    for (smiles, y_val) in zip(x_smiles, y):
        
        try :
        # convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
            n_nodes = mol.GetNumAtoms()
            n_edges = 2*mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
            X = np.zeros((n_nodes, n_node_features))

            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = get_atom_features(atom)
            
            X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
            (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
            EF = np.zeros((n_edges, n_edge_features))
        
            for (k, (i,j)) in enumerate(zip(rows, cols)):
            
                EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
            EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
            y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
            data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
        except: 
            num_error = num_error+1
            print (num_error)
            list_smiles_error.append(smiles)
    return data_list


# CREATE THE GRAPH NEURAL NETWORK MODEL 
# THE PARAMETER WHICH ARE USED IN THIS MODEL => HIDDEN CHANNEL , NUM_LAYER , DROP_OUT PERCENTAGE 
#==========================================================================
class modelA1(torch.nn.Module):
    def __init__(self, hidden_channels1,hidden_channels2, num_node_features,heads1,heads2,dropout_rateA,dropout_rateB,dropout_rateC,dense_layer1):
        super(modelA1, self).__init__()
        
        torch.manual_seed(12345)
        self.conv1 = GATConv(num_node_features, hidden_channels1,heads1)
        self.conv2 = GATConv(hidden_channels1*heads1,hidden_channels2, heads2)
        
        self.bn1 = BatchNorm (hidden_channels1*heads1)
        self.bn2 = BatchNorm (hidden_channels2*heads2)
        
        self.dropoutA = dropout_rateA
        self.dropoutB = dropout_rateB
        self.dropoutC = dropout_rateC
        
        self.lin1 = Linear(hidden_channels2*heads2,dense_layer1)
        self.lin2=  Linear(dense_layer1,1)
        
    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        
        x = F.dropout(x, p=self.dropoutA , training=self.training)
        x = x.relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        
        x = F.dropout(x, p=self.dropoutB , training=self.training)
        x = x.relu()
        x = self.bn2(x)  
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropoutC , training=self.training)
        x = x.relu()
        x = self.lin2(x)
        return torch.sigmoid(x)


# Not count the performance metrics only the outcome of prediction
def test_1(loader, gnn_model):
    gnn_model.eval()
    list_pred =[]
    list_targets =[]
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
            out = gnn_model(data.x, data.edge_index, data.batch)  
            out_1 = out[:,0]
            
            list_pred.append(out_1.item())
            list_targets.append(data.y.item())
    return list_pred, list_targets 


# create empty model with the hyperparameter 
nCV = 10
my_hyper={'hidden_channels1': 112, 
 'hidden_channels2': 112,
 'heads1': 10, 
 'heads2': 10,
 'optimizer_type': 2, 
 'dropout_rateA': 0.1875877723990742,
 'dropout_rateB': 0.16097942424559475, 
 'dropout_rateC': 0.13247000729523636, 
 'learning_rate': 0.0001262534912805999, 
 'dense_layer1': 54,
 'decay': 0.0001409381594676602}

hidden_channels1= my_hyper['hidden_channels1']
hidden_channels2= my_hyper['hidden_channels2']
num_node_features =79
heads1=my_hyper['heads1']
heads2=my_hyper['heads2']
dropout_rateA=my_hyper['dropout_rateA']
dropout_rateB=my_hyper['dropout_rateB']
dropout_rateC=my_hyper['dropout_rateC']
dense_layer1=my_hyper['dense_layer1']

list_trained_model =[]
for i in range(10):
    loaded_model = modelA1(hidden_channels1,hidden_channels2, num_node_features,heads1,heads2,dropout_rateA,dropout_rateB,dropout_rateC,dense_layer1) 
    loaded_model.load_state_dict(torch.load("0.78915model_GNN"+ str(i)+ ".pth"))
    list_trained_model.append(loaded_model)
    
# test the model
#=======================================================================

def smiles_to_tuberc(smiles_string):
    
    y = [3] # random value 
    x_smiles = [smiles_string]
    data_list_test = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y)
    test_loader = G_Loader(dataset = data_list_test, batch_size = 1)
    nCV= 10 # ten crossfold validation 
    list_fold_pred =[]
    list_fold_targets =[]
    
    for i, gnn_model in enumerate(list_trained_model):
        #test_acc = test(test_loader,gnn_model)
        list_pred,_ = test_1(test_loader,gnn_model)
        list_fold_pred.append(list_pred[0])
    mean_pred = statistics.mean(list_fold_pred)
    if mean_pred > 0.5 :
        return 'penetrating compound for M.Tuberculosis (' + str(mean_pred) +')'
    else :
        return 'non-penetrating compound for M.Tuberculosis (' + str(mean_pred) +')' 