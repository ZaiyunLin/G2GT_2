# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import os.path as osp
import shutil
from pympler import asizeof
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch_geometric.data import Dataset,InMemoryDataset
from torch_geometric.data import Data
from multiprocessing import Pool 
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
import joblib
import math
from scipy.spatial.distance import cdist
from scipy import sparse as sp
import scipy
import networkx as nx
import torch.nn.functional as F
import random
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

import torch
import numpy as np
import torch_geometric.datasets
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
import pickle
import copy 
import copy
from scipy import sparse as sp
import scipy
import networkx as nx

# ===================== NODE START =====================
atomic_num_list = list(range(119))
chiral_tag_list = list(range(4))
degree_list = list(range(11))
possible_formal_charge_list = list(range(16))
possible_numH_list = list(range(9))
possible_number_radical_e_list = list(range(5))
possible_hybridization_list = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'S']
possible_is_aromatic_list = [False, True]
possible_is_in_ring_list = [False, True]
explicit_valence_list = list(range(13))
implicit_valence_list = list(range(13))
total_valence_list = list(range(26))
total_degree_list = list(range(32))

def simple_atom_feature(atom):
    atomic_num = atom.GetAtomicNum()
    assert atomic_num in atomic_num_list

    chiral_tag = int(atom.GetChiralTag())
    assert chiral_tag in chiral_tag_list

    degree = atom.GetTotalDegree()
    assert degree in degree_list

    possible_formal_charge = atom.GetFormalCharge()
    possible_formal_charge_transformed = possible_formal_charge + 5
    assert possible_formal_charge_transformed in possible_formal_charge_list
    
    possible_numH = atom.GetTotalNumHs()
    assert possible_numH in possible_numH_list
    # 5
    possible_number_radical_e = atom.GetNumRadicalElectrons()
    assert possible_number_radical_e in possible_number_radical_e_list

    possible_hybridization = str(atom.GetHybridization())
    assert possible_hybridization in possible_hybridization_list
    possible_hybridization = possible_hybridization_list.index(possible_hybridization)

    possible_is_aromatic = atom.GetIsAromatic()
    assert possible_is_aromatic in possible_is_aromatic_list
    possible_is_aromatic = possible_is_aromatic_list.index(possible_is_aromatic)

    possible_is_in_ring = atom.IsInRing()
    assert possible_is_in_ring in possible_is_in_ring_list
    possible_is_in_ring = possible_is_in_ring_list.index(possible_is_in_ring)

    explicit_valence = atom.GetExplicitValence()
    assert explicit_valence in explicit_valence_list
    # 10
    implicit_valence = atom.GetImplicitValence()
    assert implicit_valence in implicit_valence_list

    total_valence = atom.GetTotalValence()
    assert total_valence in total_valence_list
    
    total_degree = atom.GetTotalDegree()
    assert total_degree in total_degree_list

    sparse_features = [
        atomic_num, chiral_tag, degree, possible_formal_charge_transformed, possible_numH,
        possible_number_radical_e, possible_hybridization, possible_is_aromatic, possible_is_in_ring, explicit_valence,
        implicit_valence, total_valence, total_degree,
    ]
    return sparse_features

def easy_bin(x, bin):
    x = float(x)
    cnt = 0
    if math.isinf(x):
        return 120
    if math.isnan(x):
        return 121

    while True:
        if cnt == len(bin):
            return cnt
        if x > bin[cnt]:
            cnt += 1
        else:
            return cnt


def peri_features(atom, peri):
    rvdw = peri.GetRvdw(atom.GetAtomicNum())
    default_valence = peri.GetDefaultValence(atom.GetAtomicNum())
    n_outer_elecs = peri.GetNOuterElecs(atom.GetAtomicNum())
    rb0 = peri.GetRb0(atom.GetAtomicNum())
    sparse_features = [
        default_valence,
        n_outer_elecs,
        easy_bin(rvdw, [1.2 , 1.5 , 1.55, 1.6 , 1.7 , 1.8 , 2.4]),
        easy_bin(rb0, [0.33 , 0.611, 0.66 , 0.7  , 0.77 , 0.997, 1.04 , 1.54])
    ]
    return sparse_features

def envatom_feature(mol, radius, atom_idx):  
    env= Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx, useHs=True)
    submol=Chem.PathToSubmol(mol, env, atomMap={})
    return submol.GetNumAtoms()

def envatom_features(mol, atom):
    return [
        envatom_feature(mol, r, atom.GetIdx()) for r in range(2, 9)
    ]

def atom_to_feature_vector(atom, peri, mol):
    sparse_features = []
    sparse_features.extend(simple_atom_feature(atom))
    sparse_features.extend(peri_features(atom, peri))
    sparse_features.extend(envatom_features(mol, atom))
    sparse_features.append(easy_bin(atom.GetProp('_GasteigerCharge'),
      [-0.87431233, -0.47758285, -0.38806704, -0.32606976, -0.28913129,
       -0.25853269, -0.24494531, -0.20136365, -0.12197541, -0.08234462,
       -0.06248558, -0.06079668, -0.05704827, -0.05296379, -0.04884997,
       -0.04390136, -0.03881107, -0.03328515, -0.02582824, -0.01916618,
       -0.01005982,  0.0013529 ,  0.01490858,  0.0276433 ,  0.04070013,
        0.05610381,  0.07337645,  0.08998278,  0.11564625,  0.14390777,
        0.18754518,  0.27317209,  1.        ]))
    return sparse_features


import os.path as osp
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def donor_acceptor_feature(x_num, mol):
    chem_feature_factory_feats = chem_feature_factory.GetFeaturesForMol(mol)
    features = np.zeros([x_num, 2], dtype = np.int64)
    for i in range(len(chem_feature_factory_feats)):
        if chem_feature_factory_feats[i].GetFamily() == 'Donor':
            node_list = chem_feature_factory_feats[i].GetAtomIds()
            for j in node_list:
                features[j, 0] = 1
        elif chem_feature_factory_feats[i].GetFamily() == 'Acceptor':
            node_list = chem_feature_factory_feats[i].GetAtomIds()
            for j in node_list:
                features[j, 1] = 1
    return features

chiral_centers_list = ['R', 'S']
def chiral_centers_feature(x_num, mol):
    features = np.zeros([x_num, 1], dtype = np.int64)
    t = Chem.FindMolChiralCenters(mol)
    for i in t:
        idx, type = i
        features[idx] = chiral_centers_list.index(type) + 1 # 0 for not center
    return features
# ===================== NODE END =====================

# ===================== BOND START =====================
possible_bond_type_list = list(range(32))
possible_bond_stereo_list = list(range(16))
possible_is_conjugated_list = [False, True]
possible_is_in_ring_list = [False, True]
possible_bond_dir_list = list(range(16))

def bond_to_feature_vector(bond):
    # 0
    bond_type = int(bond.GetBondType())
    assert bond_type in possible_bond_type_list

    bond_stereo = int(bond.GetStereo())
    assert bond_stereo in possible_bond_stereo_list

    is_conjugated = bond.GetIsConjugated()
    assert is_conjugated in possible_is_conjugated_list
    is_conjugated = possible_is_conjugated_list.index(is_conjugated)

    is_in_ring = bond.IsInRing()
    assert is_in_ring in possible_is_in_ring_list
    is_in_ring = possible_is_in_ring_list.index(is_in_ring)

    bond_dir = int(bond.GetBondDir())
    assert bond_dir in possible_bond_dir_list

    bond_feature = [
        bond_type,
        bond_stereo,
        is_conjugated,
        is_in_ring,
        bond_dir,
    ]
    return bond_feature
# ===================== BOND END =====================

# ===================== ATTN START =====================
def get_rel_pos(mol):
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        energy = res[index][1]
        conf = new_mol.GetConformer(id=int(index))
    except:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        energy = 0
        conf = new_mol.GetConformer()

    atom_poses = []
    for i, atom in enumerate(new_mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(new_mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    atom_poses = np.array(atom_poses, dtype=float)
    rel_pos_3d = cdist(atom_poses, atom_poses)
    return rel_pos_3d
def shortest_path(mol):
# GetDistanceMatrix returns the molecules 2D (topological) distance matrix:
    dm = Chem.GetDistanceMatrix(mol)
    return dm

''' ===================== Graph 2 Seq ==================== '''

#Indexing
def item2indx(node=None,edge=None):
    
    if not node and not edge:
        raise ValueError('nothing provided')
        
    if node :
        if node[2] ==1 and node[0]==7 :
            indx = 119 + node[1]*119
            return indx
        indx = node[0]+node[1]*119
        # dic[indx] = node
        return indx
            
    if edge:
        indx = edge + 119*4
        # dic[indx] = edge
        return  indx
        


def bond2indx(bond):
    bond_type = int(bond.GetBondType())
    indx = item2indx(edge=bond_type)
    return indx

def atom2indx(atom):
    atomic_num = atom.GetAtomicNum()
    explicit_Hs = atom.GetNumExplicitHs()
    assert atomic_num in atomic_num_list
    chiral_tag = int(atom.GetChiralTag())
    assert chiral_tag in chiral_tag_list
    indx = item2indx(node=[atomic_num,chiral_tag,explicit_Hs])
    return indx

#==========Graph sequence===========
def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output




# ===================== ATTN START =====================

curmax = 0
ditched = 0
def smiles2graph_wrapper(smiles_list):
    
    src = smiles_list[0]
    tgt = smiles_list[1]
    try:
        src_graph = smiles2graph(src,bfs=True)
        tgt_graph = smiles2graph(tgt,bfs=True,neutral_atoms=False)
    except:
        print(src,tgt,"smiles2graph_wrapper error")
        return "error"
    
    return src_graph,tgt_graph
def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol
def smiles2graph(smiles_string, bfs=True,neutral_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    stuff = smiles_string.split(",")
    reverse = -1
    if len(stuff)>1:
        smiles_string = stuff[1]
        reverse = int(stuff[0])
    else:
        smiles_string = stuff[0]
    
    #build networkx graph for bfs
    if bfs:
        G = nx.DiGraph()
    if neutral_atoms:
        mol = neutralize_atoms(Chem.MolFromSmiles(smiles_string))
    else:
        mol = Chem.MolFromSmiles(smiles_string)
    AllChem.ComputeGasteigerCharges(mol)
    peri=Chem.rdchem.GetPeriodicTable()
    g_node_attr = []
    g_edge_attr = []
    # atoms
    i = 0
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom, peri, mol))
        if bfs:
            G.add_nodes_from([(i,{"node":atom2indx(atom)})])
            g_node_attr.append(atom2indx(atom))
            i+=1
    
    x = np.array(atom_features_list, dtype = np.int64)
    x = np.concatenate([x, donor_acceptor_feature(x.shape[0], mol)], axis=1)
    x = np.concatenate([x, chiral_centers_feature(x.shape[0], mol)], axis=1)

    #adj_mat_indx && adj_mat
    atom_num = len(x)
    adj_mat = adj = torch.zeros(atom_num,atom_num)

    
    
    #node_edge_seq
    node_seq = np.zeros(atom_num, dtype = np.int64)
    
    # bonds
    num_bond_features = 5
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            # node_seq[i] = atom2indx(dic, begin_atom)
            # node_seq[j] = atom2indx(dic, end_atom)

            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            
            '''new'''
            #adjacency matrix
            adj_mat[i][j] = adj_mat[j][i] = 1
            adj[j][i] = adj[i][j] = bond2indx(bond) 
            
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            
            edge_features_list.append(edge_feature)
            if bfs:
                G.add_edges_from([(i,j,{"edge": adj[j][i]})])
                G.add_edges_from([(j,i,{"edge": adj[j][i]})])

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    '''New'''
    #graph sequence
    node_edge_seq = []
    frags = Chem.GetMolFrags(mol)
    frags_len = []
    for i in range(len(frags)):
        frags_len.append(len(frags[i]))
        
   
    # attn

    rel_pos_3d = shortest_path(mol)
    graph = dict()

    #graph['node_edge_seq'] = np.array(node_edge_seq,dtype = np.int64)
    graph['node_edge_seq'] = nx.to_numpy_array(G,weight="edge").tolist()
    graph['g_node_attr'] = np.array(g_node_attr)



    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    graph['rel_pos_3d'] = rel_pos_3d
    graph['frags'] = np.array(frags_len,dtype=np.int64)
    graph['reverse'] = reverse
    
    # if bfs:
        # lpe_input,central_input = make_laplacian_cent_attn_bias(40, graph['node_edge_seq'])  
        # graph['central_input'] = central_input
        # graph['lpe_input'] = lpe_input

    return graph 


class UsptoDataset(InMemoryDataset):
    def __init__(self, root = 'data/typed_uspto50k_split2', smiles2graph = smiles2graph, transform=None, pre_transform = None, smiles2graph_wrapper = smiles2graph_wrapper):
        '''
            Pytorch Geometric PCQM4M dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''
        
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.smiles2graph_wrapper = smiles2graph_wrapper
        self.folder = osp.join(root)
        self.trainindx = 0
        self.testindx = 0
        self.valindx = 0
        self.dic = {}
        super(UsptoDataset, self).__init__(self.folder, transform, pre_transform)

        # self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ['test-src.txt','test-tgt.txt','train-src.txt','train-tgt.txt','valid-src.txt','valid-tgt.txt']

    @property
    def processed_file_names(self):
        return ['data_0.pt']



    def process(self):
        data_dic = {'test':{'src':[],'tgt':[]},'train':{'src':[],'tgt':[]},'valid':{'src':[],'tgt':[]}}
        data_df = pd.DataFrame(data=data_dic)
        
        for raw in self.raw_file_names:
            try:
                raw_path = self.root+"/raw/"+raw         
                f = open(raw_path,'r')
                raw= raw.replace(".txt","")
                a,b = raw.split('-')
                data_df[a][b] = f.readlines()
            except:
                continue
            #print(data_df)
        self.testindx = len(data_dic['test']['src'])
        self.trainindx = len(data_dic['train']['src']) +  self.testindx
        self.valindx = len(data_dic['valid']['src']) + self.trainindx
        
       # homolumogap_list = data_df['homolumogap']
            
        print('Converting SMILES strings into graphs...')
        maxl = 0
        count = 0
        np_data_list = []
        y_graph_list = []
        # lpe_inputs = []
        # central_inputs =[]
        src_f = open('src.txt', mode='w',encoding='utf-8')
        tgt_f = open('tgt.txt', mode='w',encoding='utf-8')
        i=0
        rng = np.random.default_rng()
        for j in data_df:
            src_list = data_df[j]['src']
            tgt_list = data_df[j]['tgt']
            smiles_list = list(zip(src_list,tgt_list))
            with Pool(processes = 80) as pool:
                iter = pool.imap(self.smiles2graph_wrapper, smiles_list)
                for idx, graph in tqdm(enumerate(iter),total = len(smiles_list)):
                    if graph=="error":
                        print(i+1)
                        continue
                    # if i==9000:
                    #     break
                    src_graph = graph[0]
                    tgt_graph = graph[1]

                    data = Data()
                    assert(len(src_graph['edge_feat']) == src_graph['edge_index'].shape[1])
                    assert(len(src_graph['node_feat']) == src_graph['num_nodes'])
                    data.__num_nodes__ = int(src_graph['num_nodes'])
                    data.edge_index = src_graph['edge_index']
                    data.edge_attr = src_graph['edge_feat']
                    data.x = src_graph['node_feat']   
                    data.all_rel_pos_3d =src_graph['rel_pos_3d']
                    '''=================new=================='''

                    data.y_edge_index = tgt_graph['edge_index']
                    data.y_num_nodes = int(tgt_graph['num_nodes'])
                    data.y_frags = tgt_graph['frags']
                    data.y_node_attr = tgt_graph['g_node_attr']
                    
                    data.y = tgt_graph['node_edge_seq']
                    #data.reverse = 0
                    data.reverse = src_graph['reverse']
                    # print(data.reverse)
    

                    data.edge_index = torch.from_numpy(data.edge_index).to(torch.int64)
                    data.edge_attr = torch.from_numpy(data.edge_attr).to(torch.int64)
                    data.x = torch.from_numpy(data.x).to(torch.int64)
                    data.reverse = torch.tensor([data.reverse],dtype = torch.int64)
                    data.y_edge_index = torch.from_numpy(data.y_edge_index).to(torch.int64)                    
                    torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                    i += 1


    def len(self):
        #return 1182
        return 50000
        #return 5005+450550+4990 # 50k split2 distilled
        #return 873211+5001 #50k+distilled
        #return 125139
        #return 96025+1681771+96000 #reaxys-uspto
        #return 45294
        #return 96026+768807+852000
        #return 96026+1219662+94967 #full distill
        #return 96026+1422976+96067
        #return 852000
        #return 960916 #960923

        #return 960900
        #return 400000+5000
        #return 45000
        #return 45000*2+5000
    def get(self, idx):
        try:
            data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        except:
            print("\n",idx,"failed load\n")
            raise Exception()
        data = preprocess_item(data, noise=True)
        data.idx = idx
        
        return  data
        #return data
        
    def get_idx_split(self):
        #return 1182,0,0
        #return 5005,5005+450550,5005+450550+4990 # 50k split2 distilled
        #return 1,873211,873211+5001 #50k+distilled
        #return 125139,125139,125139
        #return 96025,96025+1681771+96000,96025+1681771+96000
        #return 45294,45294,45294
        #return 96026,96026+768807+852000,96026+768807+852000
        #return 96026, 96026+1422976,96026+1422976+96067
        #return 50000,767109,852000
        # return 85234,767109,852000
        #return 96026,96026+1219662,96026+1219662+94967 #uspto-full-and-distill
        return 5000,45000,50000
        #return 111098,111098,111098
        #return 100000,500299,500299
        #return 0,40000*5,40000*5+5000

    def get_dic(self):
        return self.dic








def lpe_pos_enc(g, pos_enc_dim,in_d):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.float()
    N = sp.diags(np.array(in_d).clip(1) ** -0.5, dtype=float)
    L = sp.eye(len(in_d)) - N * A * N
    
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L)

    EigVec = torch.from_numpy(EigVec).float()
    EigVec = F.normalize(EigVec, p=2, dim=1, eps=1e-12, out=None)
    
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = np.real(EigVal[idx]), EigVec[:,idx]
    
    pos_enc = EigVec[:,0:pos_enc_dim]

    xlen1,xlen2 = pos_enc.shape    
    length = len(in_d)
    new_pos_enc =torch.zeros([length, pos_enc_dim], dtype=torch.float)
    
    if xlen1<pos_enc_dim and xlen2<pos_enc_dim:
        new_pos_enc[:xlen1,:xlen2] = pos_enc
    else:
        new_pos_enc[:length,:pos_enc_dim] = pos_enc[:length,:pos_enc_dim]
    
    return new_pos_enc

def laplace_decomp(g, max_freqs, in_d):

    n = len(in_d)
    A = g.float()
    N = sp.diags(np.array(in_d).clip(1) ** -0.5, dtype=float)
    L = sp.eye(len(in_d)) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L)
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        EigVecs = F.pad(EigVecs, (0, max_freqs-n), value=0)
    else:
        EigVecs= EigVecs
        
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=0).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    #Save EigVals node features
    Eigvals = EigVals.repeat(n,1)
    
    return EigVecs, Eigvals

'''Centrality and Laplacian'''
def make_laplacian_cent_attn_bias(pos_enc_dim, y):
    M = len(y)
    lpe_bias = torch.zeros([M,M,pos_enc_dim],dtype=torch.float)
    lpe_eigenval = torch.zeros([M,M,pos_enc_dim],dtype=torch.float)
    #lpe_bias = [[]for i in range(M)]for i in range(M)
    cent_bias = torch.zeros([M,M],dtype=torch.long)
    offset = 4
    for i in range(M):
        # if last added is node then update lpe and centrality
        if y[i]!=0 and y[i]> 472+4 and y[i]<=(504+4):
            sub_NE = copy.deepcopy(y[0:i+1])
            G = nx.Graph()
            node_indx = -1
            check_prev = 0
            for j in range(i+1):       
                NodeEdge = sub_NE[j].item()
                #if padding
                if NodeEdge ==0:
                    continue
                # if node
                
                if NodeEdge <=472+offset:
                    check_prev = 0
                    node_indx +=1
                    G.add_nodes_from([(node_indx,{"node":NodeEdge})])   
                #if edge not node       
                elif NodeEdge <=(504+offset):
                    check_prev +=1
                    G.add_edges_from([(node_indx,node_indx-check_prev,{"edge":NodeEdge})])
                elif NodeEdge >506+offset and NodeEdge<531:
                    check_prev = check_prev+ (NodeEdge-(506+offset))
            sub_adj =torch.tensor(nx.adjacency_matrix(G).todense())
            sub_ind = []
            for node in G.nodes:
                sub_ind.append(G.degree[node])
            sub_ind = torch.tensor(sub_ind)
            sub_lpe,sub_eigenval = laplace_decomp(sub_adj,pos_enc_dim,sub_ind)
            node_indx = 0
            for k in range(i+1):
                NodeEdge = sub_NE[k]
                # if node
                if NodeEdge != 0 and NodeEdge <=472+offset:
                    lpe_bias[i][k] = sub_lpe[node_indx]
                    
                    lpe_eigenval[i][k] = sub_eigenval[node_indx]
                    cent_bias[i][k] = sub_ind[node_indx]
                    node_indx+=1

        else:
            lpe_bias[i] = lpe_bias[i-1]
            lpe_eigenval[i] = lpe_eigenval[i-1]
            cent_bias[i] = cent_bias[i-1]

    return lpe_bias,lpe_eigenval,cent_bias

def convert_to_single_emb(x, offset=128):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def g2graph_seq(G,frags,random=True):
    offset = 4

    bfsseq = []
    full_frag = []
    node_edge_seq = []
    start_indx = 0
    for i in range(len(frags)):
        if i!=0:
            start_indx = frags[i-1]

        end_indx = frags[i] + start_indx
        frags[i] = end_indx
        temp_frags = list(range(start_indx,end_indx))

        full_frag.append(temp_frags)
    
    
    rng = np.random.default_rng()

    random = False
    random_type = 0
    if random:
        random_type = rng.choice([0,1,0])
    for i in range(len(full_frag)):
        random_begin_indx = 0
        if random_type ==1:
            random_begin_indx = rng.integers(0,len(full_frag[i]))
        elif random_type == 2:
            random_begin_indx = len(full_frag[i])//2
        elif random_type == 3:
            random_begin_indx = -1
        
        #print(random_begin_indx)
        #random_begin_indx=0
        try:
            bfsseq += bfs_seq(G,full_frag[i][random_begin_indx]) 
        except:
            bfsseq += bfs_seq(G,full_frag[i][-1]) 
    max_prev = 20
    count = 0
    
    for nodeindx in range(len(bfsseq)):
        node_edge_seq.append(G.nodes[bfsseq[nodeindx]]['node'])
        gap =0
        for j in range(0,nodeindx):
            #check i-1, i-2...i-n node
            check_node_idx = nodeindx-(j+1)
            try:
                hasedge = int(G[bfsseq[check_node_idx]][bfsseq[nodeindx]]['weight'])
            except:
                hasedge = None
                gap +=1
            if j<max_prev and hasedge:
                if gap>0:
                    node_edge_seq.append((506+offset)+gap)
                node_edge_seq.append(hasedge)  
                gap = 0 
    return torch.tensor(node_edge_seq,dtype=torch.long),random_type



def preprocess_item(item, noise=False,random=True):

    offset = 4
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    '''new-======================='''
    y_num_nodes, y_edge_index, y,y_frags,y_node_attr=item.y_num_nodes, item.edge_index, item.y, item.y_frags,item.y_node_attr

    '''matrix to nx graph'''
    
    y = nx.from_numpy_array(np.array(y),create_using=nx.DiGraph)
    for i in range(y_num_nodes):
        y.nodes[i]['node'] = y_node_attr[i].item()

    
    '''nx graph to graph sequence'''
    y,begin_indx = g2graph_seq(y,y_frags,random=random)
    N = x.size(0)
    x = convert_to_single_emb(x)
    
    #mask
    M = y.size(0)
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    K = y_num_nodes
    lpe_input,lpe_eigenval, central_input = make_laplacian_cent_attn_bias(30,y)  
    y_attn_bias = torch.zeros([M+1 , M+1], dtype=torch.float)

    '''=================='''   
        
    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]

    all_rel_pos_3d_with_noise = torch.from_numpy(algos.bin_rel_pos_3d_1(item.all_rel_pos_3d, noise=noise)).long()

    rel_pos_3d_attr = all_rel_pos_3d_with_noise[edge_index[0, :], edge_index[1, :]]

    edge_attr = torch.cat([edge_attr, rel_pos_3d_attr[:, None]], dim=-1)
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float) # with graph token
    


    #split 
    start = torch.tensor([532],dtype=torch.long)
    end = torch.tensor([506+offset],dtype=torch.long)
    y = torch.cat((start+item.reverse,y))
    # print(y)

    y = torch.cat((y,end))
        


    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.all_rel_pos_3d_1 = torch.from_numpy(item.all_rel_pos_3d).float()
    
    # new 
    item.central_input = central_input
    item.lpe_input = lpe_input
    item.lpe_eigenval = lpe_eigenval
    item.y_attn_bias = y_attn_bias
    item.y = y

    return item


