# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data import get_dataset
from lr import PolynomialDecayLR
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl
from utils.flag import flag, flag_bounded
import numpy as np
import torch.nn.functional as F
import sys
from pprint import pprint
from torch.autograd import Variable
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(precision=10)
from scipy.spatial.distance import cdist
from scipy import sparse as sp
import scipy
import networkx as nx
np.random.seed(0)
import random
import time
import pickle

def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

class RBFLayer(nn.Module):
    def __init__(self, K=64, cutoff=10, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff

        centers = torch.tensor(softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)), dtype=dtype)
        self.centers = nn.Parameter(F.softplus(centers))

        widths = torch.tensor([softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K, dtype=dtype)
        self.widths = nn.Parameter(F.softplus(widths))
    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        return torch.where(x < 1, 1-6*x5+15*x4-10*x3, torch.zeros_like(x))
    def forward(self, D):
        D = D.unsqueeze(-1)
        return self.cutoff_fn(D) * torch.exp(-self.widths*torch.pow((torch.exp(-D) - self.centers), 2))

class GraphFormer(pl.LightningModule):
    def __init__(
            self,
            n_layers,
            head_size,
            hidden_dim,
            dropout_rate,
            intput_dropout_rate,
            weight_decay,
            ffn_dim,
            dataset_name,
            warmup_updates,
            tot_updates,
            peak_lr,
            end_lr,
            edge_type,
            multi_hop_max_dist,
            attention_dropout_rate,
            flag=False,
            flag_m=3,
            flag_step_size=1e-3,
            flag_mag=1e-3,
            inference_path = None,
            weak_ensemble = 0,
            
        ):
        super().__init__()
        self.save_hyperparameters()
        self.inference_path = inference_path
        self.weak_ensemble = weak_ensemble
        self.head_size = head_size

        offset = 4
        self.atom_encoder = nn.Embedding(128 * 37 + 1, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(128 * 6 + 1, head_size, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(128 * head_size * head_size,1)
        self.rel_pos_encoder = nn.Embedding(512, head_size, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        self.input_dropout2d = nn.Dropout2d(0.05)

        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]

        
        '''NEW=============================================='''
        self.gelu = nn.GELU()
        self.atom_edge_encoder = nn.Embedding(118*4+32+(2)+1+20+offset +50+1, hidden_dim, padding_idx=0) # atom*chiral +bond+start+end+0+gapnumber+[nh] offset +50+1split
        self.centrality_encoder = nn.Embedding(50, head_size, padding_idx=0)
        self.lpe_linear = nn.Linear(2,head_size)
        self.lpe_linear3 = nn.Linear(30,head_size)
        self.position = PositionalEncoding(hidden_dim,0)
        
        self.outp_logits = nn.Linear(hidden_dim, 118*4+32+(2)+1+20+offset)
        
        ### ===== DECODER =====. need hidden_dim2, ffn_dim2


        decoders=[DecoderLayer(hidden_dim,ffn_dim, dropout_rate, attention_dropout_rate, head_size) for _ in range(n_layers)]
        self.decoderLayers = nn.ModuleList(decoders)
        self.layers = nn.ModuleList(encoders)
        

        self.graph_token = nn.Embedding(50, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, head_size)
        
        #todo fix eval
        self.evaluator = get_dataset(dataset_name)['evaluator']
        self.metric = get_dataset(dataset_name)['metric']
        self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name
        
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag
 
        K = 256
        cutoff = 10
        self.rbf = RBFLayer(K, cutoff)
        self.rel_pos_3d_proj = nn.Linear(K, head_size)

        unfreeze = True
        
        self.atom_encoder.weight.requires_grad = unfreeze
        self.edge_encoder.weight.requires_grad = unfreeze
        self.rel_pos_encoder.weight.requires_grad = unfreeze
        self.in_degree_encoder.weight.requires_grad = unfreeze
        self.out_degree_encoder.weight.requires_grad = unfreeze

        self.atom_edge_encoder.weight.requires_grad = unfreeze
        self.centrality_encoder.weight.requires_grad = unfreeze
        self.lpe_linear.weight.requires_grad = unfreeze
        self.lpe_linear3.weight.requires_grad = unfreeze
        
        self.graph_token.weight.requires_grad = unfreeze
        self.graph_token_virtual_distance.weight.requires_grad = unfreeze



    def translate_encoder(self,batched_data, beam=1, perturb=None, y=None, valid = True):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
        all_rel_pos_3d_1 = batched_data.all_rel_pos_3d_1
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        #unsqueeze for multihead
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) # [n_graph, n_head, n_node+1, n_node+1]

        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        rbf_result = self.rel_pos_3d_proj(self.rbf(all_rel_pos_3d_1)).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rbf_result

        # reset rel pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.head_size, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            rel_pos_ = rel_pos.clone()
            rel_pos_[rel_pos_ == 0] = 1 # set pad to 1
            rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_) # set 1 to 1, x > 1 to x - 1
            if self.multi_hop_max_dist > 0:
                rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            edge_input = self.edge_encoder(edge_input).mean(-2) #[n_graph, n_node, n_node, max_dist, n_head]
            max_dist = edge_input.size(-2)
            try:
                edge_input_flat = edge_input.permute(3,0,1,2,4).reshape(max_dist, -1, self.head_size)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(-1, self.head_size, self.head_size)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.head_size).permute(1,2,3,0,4)
                edge_input = (edge_input.sum(-2) / (rel_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            except:
                print("Warning!!!!!!!!!!!!!!!!")
                edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1) # reset
        
        # print("x",x.shape)
        node_feature = self.atom_encoder(x).mean(dim=-2)           # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token(batched_data.reverse) #[n_graph, n_hidden]
        graph_token_feature = graph_token_feature.unsqueeze(1) # [n_graph, n_node, n_hidden]



        if beam>1:
            node_feature = node_feature.repeat(beam,1,1)
        # print("graph_token_feature, node_feature",graph_token_feature.shape, node_feature.shape)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        enc_out = graph_node_feature
        
        
            
        for enc_layer in self.layers:
            enc_out = enc_layer(enc_out, graph_attn_bias,valid=valid)
        # if beam>1:
        #     enc_out = enc_out.repeat(beam,1,1)


        return enc_out
    def translate_decoder(self, batched_data, enc_out,perturb=None, y=None, valid = True):
        if y is None:
            y = batched_data.y
        batch_size=y.size(0)
        y_attn_bias = batched_data.y_attn_bias
        
        
        y_graph_attn_bias = y_attn_bias.clone()

        #unsqueeze for multihead
        y_graph_attn_bias = y_graph_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) # [n_graph, n_head, n_node+1, n_node+1]

        if batched_data.subsequent_mask is not None:
            tgt_subsq_mask = batched_data.subsequent_mask
            tgt_subsq_mask = tgt_subsq_mask.unsqueeze(0)
            tgt_subsq_mask = tgt_subsq_mask.unsqueeze(0).repeat(batch_size, self.head_size, 1, 1)
        else:
            tgt_subsq_mask = None


        '''=========Centrality section========'''
        central_input = batched_data.central_input
        
        central_input = self.centrality_encoder(central_input).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        y_graph_attn_bias[:,:,1:,1:] = y_graph_attn_bias[:,:,1:,1:] + central_input #[n_graph, n_head, n_node, n_node]


        '''=========LPE section========'''
     
        lpe_input = batched_data.lpe_input
        lpe_eigenval = batched_data.lpe_eigenval
        lpe_input = torch.cat((lpe_input.unsqueeze(-1),lpe_eigenval.unsqueeze(-1)),dim=-1)
        
        lpe_input = self.lpe_linear(lpe_input)
        lpe_input = torch.nansum(lpe_input,-2,keepdim = False)
     
        #lpe_input = lpe_input.unsqueeze(1).repeat(1,self.head_size,1,1)
        lpe_input = lpe_input.permute(0,3,1,2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        y_graph_attn_bias[:,:,1:,1:]= y_graph_attn_bias[:,:,1:,1:] + lpe_input
        


        y_graph_attn_bias = y_graph_attn_bias + y_attn_bias.unsqueeze(1)
        # print(y)
        NE_feature = self.atom_edge_encoder(y)
        output = self.position(NE_feature)

        for dec_layer in self.decoderLayers:
            output = dec_layer(output,enc_out,enc_out,tgt_subsq_mask,y_graph_attn_bias,valid=valid,check=False)
                

        
        output = self.outp_logits(output)

       
        return output

    def forward(self, batched_data, perturb=None, y=None, valid = False):

        
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
        all_rel_pos_3d_1 = batched_data.all_rel_pos_3d_1

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        #unsqueeze for multihead
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) # [n_graph, n_head, n_node+1, n_node+1]
        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        rbf_result = self.rel_pos_3d_proj(self.rbf(all_rel_pos_3d_1)).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rbf_result

        # reset rel pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.head_size, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            rel_pos_ = rel_pos.clone()
            rel_pos_[rel_pos_ == 0] = 1 # set pad to 1
            rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_) # set 1 to 1, x > 1 to x - 1
            if self.multi_hop_max_dist > 0:
                rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            edge_input = self.edge_encoder(edge_input).mean(-2) #[n_graph, n_node, n_node, max_dist, n_head]
            max_dist = edge_input.size(-2)
            try:
                edge_input_flat = edge_input.permute(3,0,1,2,4).reshape(max_dist, -1, self.head_size)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(-1, self.head_size, self.head_size)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.head_size).permute(1,2,3,0,4)
                edge_input = (edge_input.sum(-2) / (rel_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            except:
                print("Warning!!!!!!!!!!!!!!!!")
                edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1) # reset
        
        # node feauture + graph token
        node_feature = self.atom_encoder(x).mean(dim=-2)           # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token(batched_data.reverse)

        graph_token_feature = graph_token_feature.unsqueeze(1)
        # node_feature += graph_token_feature
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        ##encoder part
        
        # transfomrer encoder
        # key value
        '''=================================new============================='''
        #tgt data
        if y is None:
            y = batched_data.y
            
        batch_size=y.size(0)
        y_attn_bias = batched_data.y_attn_bias
        y_graph_attn_bias = y_attn_bias.clone()

        #unsqueeze for multihead
        y_graph_attn_bias = y_graph_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) # [n_graph, n_head, n_node+1, n_node+1]

        if batched_data.subsequent_mask is not None:
            tgt_subsq_mask = batched_data.subsequent_mask
            tgt_subsq_mask = tgt_subsq_mask.unsqueeze(0)
            tgt_subsq_mask = tgt_subsq_mask.unsqueeze(0).repeat(batch_size, self.head_size, 1, 1)
        else:
            tgt_subsq_mask = None

        '''=========Centrality section========'''
        central_input = batched_data.central_input
        central_input = self.centrality_encoder(central_input).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        y_graph_attn_bias[:,:,1:,1:] = y_graph_attn_bias[:,:,1:,1:] + central_input #[n_graph, n_head, n_node, n_node]

        '''=========LPE section========'''
        #print(lpe_input.size(), end="lpepre mean\n")
        lpe_input = batched_data.lpe_input
        lpe_eigenval = batched_data.lpe_eigenval
        lpe_input = torch.cat((lpe_input.unsqueeze(-1),lpe_eigenval.unsqueeze(-1)),dim=-1)
        lpe_input = self.lpe_linear(lpe_input) #2->head size
        #lpe_input = lpe_encoder(lpe_input)
        lpe_input = self.gelu(lpe_input)
        lpe_input = torch.nansum(lpe_input,-2,keepdim = False)


        
        

        lpe_input = lpe_input.permute(0,3,1,2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        y_graph_attn_bias[:,:,1:,1:]= y_graph_attn_bias[:,:,1:,1:] + lpe_input
        y_graph_attn_bias = y_graph_attn_bias + y_attn_bias.unsqueeze(1)

        # model(input,prev_output)
        

        enc_out = graph_node_feature    
        

        for enc_layer in self.layers:
            enc_out = enc_layer(enc_out, graph_attn_bias,valid=valid)
        

        enc_out = self.input_dropout(enc_out)
        enc_out[:,1:]= self.input_dropout2d(enc_out[:,1:])
        

        '''=====================Decoder===================='''
        NE_feature = self.atom_edge_encoder(y)
        output = self.position(NE_feature)  

        for dec_layer in self.decoderLayers:
            output = dec_layer(output,enc_out,enc_out,tgt_subsq_mask,y_graph_attn_bias,valid=valid,check=False)

      
        output = self.outp_logits(output)

        return output
    
    
    
    def laplace_decomp(self,g, max_freqs, in_d):

        n = len(in_d)
        A = g.float()
        N = sp.diags(np.array(in_d).clip(1) ** -0.5, dtype=float)
        L = sp.eye(len(in_d)) - N * A * N

        # Eigenvectors with numpy
        EigVals, EigVecs = np.linalg.eigh(L)
        EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

        # Normalize and pad EigenVectors
        EigVecs = torch.from_numpy(EigVecs).float().to(self.device)
        EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
        
        if n<max_freqs:
            EigVecs = F.pad(EigVecs, (0, max_freqs-n), value=0)
        else:
            EigVecs= EigVecs
            
        #Save eigenvales and pad
        EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))).float().to(self.device) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
        
        if n<max_freqs:
            EigVals = F.pad(EigVals, (0, max_freqs-n), value=0).unsqueeze(0)
        else:
            EigVals=EigVals.unsqueeze(0)
            
        #Save EigVals node features
        Eigvals = EigVals.repeat(n,1)
        #print("in model",EigVecs)
        return EigVecs, Eigvals

    def make_laplacian_cent_attn_bias(self,pos_enc_dim, y):
        offset = 4
        y = y.cpu().detach().numpy()
        M = len(y)
        lpe_bias = torch.zeros((M,pos_enc_dim),dtype=torch.float,device= self.device)
        lpe_eigenval = torch.zeros((M,pos_enc_dim),dtype=torch.float,device= self.device)
        #lpe_bias = [[]for i in range(M)]for i in range(M)
        cent_bias = torch.zeros((M),dtype=torch.long,device= self.device)
       

            # if last added is node then update lpe and centrality
            
        sub_NE = y
        G = nx.Graph()
        node_indx = -1
        check_prev = 0
        g_start=time.time()
        node_pos = []
        for j in range(M):       
            NodeEdge = sub_NE[j]
            #if padding
            if NodeEdge ==0:
                continue
            # if node
            if NodeEdge <=472+offset:
                check_prev = 0
                node_indx +=1
                G.add_nodes_from([(node_indx,{"node":NodeEdge})])   
                node_pos.append(j)
            #if edge not node       
            elif NodeEdge <=504+offset:
                check_prev +=1
                G.add_edges_from([(node_indx,node_indx-check_prev,{"edge":NodeEdge})])
            elif NodeEdge > 506+offset and NodeEdge < 532 :         # split = 20
                check_prev =check_prev+ (NodeEdge-(506+offset))
        sub_adj =torch.tensor(nx.adjacency_matrix(G).todense())
        sub_ind = []
        for node in G.nodes:
            sub_ind.append(G.degree[node])
        g_end=time.time()
        g_time = g_end-g_start
        
        sub_lpe,sub_eigenval = self.laplace_decomp(sub_adj,pos_enc_dim,sub_ind)
        sub_ind = torch.tensor(sub_ind,dtype=torch.long,device=self.device)
        sub_ind[sub_ind>30] = 30
        lpe_start =time.time()

        # print("sub_lpe",sub_lpe)
        node_indx = 0
        for k in range(M):
            NodeEdge = sub_NE[k]
            # if node
            if NodeEdge != 0 and NodeEdge <=472+offset:
                lpe_bias[k] = sub_lpe[node_indx]
                lpe_eigenval[k] = sub_eigenval[node_indx]
                cent_bias[k] = min(sub_ind[node_indx],30)
                node_indx+=1

        lpe_end = time.time()
        lpe_time = lpe_end-lpe_start

        return lpe_bias,lpe_eigenval,cent_bias



    def training_step(self, batched_data, batch_idx):
    
        
        if not self.flag:
            batch_pos_enc = batched_data.lpe_input
            sign_flip = torch.rand(batch_pos_enc.size(-1),device=self.device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            batched_data.lpe_input = batch_pos_enc

            y_hat = self(batched_data)
            y_hat = y_hat.view(-1,y_hat.size(-1))
            y_gt = batched_data.y_gt.view(-1)
            
            mask = ~torch.isnan(y_gt)


            loss = self.loss_fn(y_hat[mask],y_gt[mask],ignore_index=0) 
    

        else:
            y_gt = batched_data.y.view(-1).float()
            forward = lambda perturb: self(batched_data, perturb)
            model_forward = (self, forward)
            n_graph, n_node = batched_data.x.size()[:2]
            perturb_shape = (n_graph, n_node, self.hidden_dim)

            optimizer = self.optimizers()
            optimizer.zero_grad()
            loss, _ = flag_bounded(model_forward, perturb_shape, y_gt, optimizer, batched_data.x.device, self.loss_fn,
                            m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag)
            self.lr_schedulers().step()
        self.log('train_loss', loss.detach(),sync_dist=True)
        return loss

    def validation_step(self, batched_data, batch_idx):

        
        
        y_pred = self(batched_data,valid = True)
        y_pred = y_pred.view(-1,y_pred.size(-1))
        y_true = batched_data.y_gt
        y_true = y_true.view(-1)

        return {
            'y_pred': y_pred,
            'y_true': y_true,
        }

    def validation_epoch_end(self, outputs):

        correct = 0
        l = 0
        print("validation_epoch_end")
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        for i in outputs:

            if torch.equal(torch.argmax(i['y_pred'],dim=-1),i['y_true']):
                correct+=1
                out = torch.argmax(i['y_pred'],dim=-1)

        mask = ~torch.isnan(y_true)
        loss = self.loss_fn(y_pred[mask], y_true[mask],ignore_index=0) 
        g_acc = correct/len(outputs)
        print(f"graph acc: {g_acc}")
        acc = (torch.argmax(y_pred,dim=-1)==y_true).float().mean()


        self.log('graph accuracy', g_acc, sync_dist=True)
        self.log('validation accuracy', acc, sync_dist=True)
        self.log('validation loss', loss, sync_dist=True)
        print(f"valid accuracy: {acc}")
   
    def get_square_subsequent_mask(self,seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len,device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def test_step_sample(self, batched_data, batch_idx):

        def choose_topk(scores,new_scores):
            new_scores = torch.log(new_scores)+scores

            topk_scores,topk_indx = torch.topk(new_scores.view(-1),beam)
            scores = topk_scores.reshape(beam,1)
            topk_r,topk_c = topk_indx//beam,topk_indx%beam
            topk_next_words = all_next_word[topk_r,topk_c]
            #print("topk_next_words",topk_next_words)
            return topk_next_words.view(-1),topk_r,scores
        def sampling(scores,logits,indices):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(gen.seed())
            next_tokens = torch.multinomial(logits, 1, generator=gen)
            next_token, ind = torch.min(next_tokens, -1, keepdim=False, out=None)

            # map logits index to actual onehot index
            next_words = torch.zeros((topk),dtype=torch.long,device=self.device)
            
            for i in range(indices.size(0)):
                next_words[i] = indices[i][next_token[i]]
                scores[i] =  torch.log(logits[i][next_token[i]]) + scores[i]
            return next_words,scores

        def update_y(next_words,ys,prediction,scores,batched_data):
            break_flag = True
            #create new input array
            new_ys = torch.zeros((topk,ys.size(-1)+1),dtype=torch.long,device=self.device)
            # if words is start token, repeat topk times
            if ys.size(0)==1:
                ys=ys.repeat(topk,1)
             #if length of prediction bucket >= topk stop prediction
            if len(prediction)>=topk:
                break_flag = False
            
            #loop through k options, topk actually means prick k not topk 
            for i in range(topk):
        
                #if stop words
                if next_words[i] == 506+offset:
                    #print("stoppppusingthis")
                    #raise Exception("stop")
                    prediction.append((ys[i].detach().clone().cpu().numpy(),scores[i].item()))
                    scores[i]=-9999
                    prediction.sort(key = lambda x: x[1], reverse=True)

                new_ys[i] = torch.cat((ys[i],next_words[i].reshape(1)))

            ys = new_ys.detach().clone()
            #print(ys)
            return ys,prediction,scores,break_flag,batched_data
        def top_k_top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_logits= F.softmax(sorted_logits,dim=-1)
                cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
               
                sorted_indices_to_keep = ~sorted_indices_to_remove
                
                logits = sorted_logits*sorted_indices_to_keep.int().float()
                #print(logits[0])
            return logits,sorted_indices

        offset = 4
        "=======Inference========"
        beam = 5
        # topk actually means pick k not topk 
        topk = 10
        top_p = 0.6
        temperature = 8

        
        with torch.no_grad():
            #initialize first token
            ys = torch.tensor([[505+offset]],dtype=torch.long,device=self.device)
            ys = ys.repeat(topk,1)
            y_true = batched_data.y_gt.view(-1)
            #prepare first item's input data
            batched_data.central_input = torch.zeros((topk,1,1),dtype=torch.long,device = self.device)
            batched_data.lpe_input = torch.zeros((topk,1,1,30),dtype=torch.float,device = self.device)
            batched_data.lpe_eigenval = torch.zeros((topk,1,1,30),dtype=torch.float,device = self.device)
            batched_data.y_attn_bias = torch.zeros((topk,1,1), dtype=torch.float,device = self.device)
            batched_data.subsequent_mask =None

            # temporary vars which hold the attn_bias info of last iteration
            lpe_input = torch.zeros((topk,1,30),dtype=torch.float,device= self.device)
            lpe_eigenval = torch.zeros((topk,1,30),dtype=torch.float,device= self.device)
            central_input = torch.zeros((topk,1),dtype=torch.long,device= self.device)
          
            enc_out = self.translate_encoder(batched_data,beam=topk,valid=False)
            prediction = []



        scores = torch.zeros((topk,1),dtype=torch.float,device = self.device)
        for i in range(len(y_true)+100):
            
            y_pred = self.translate_decoder(batched_data,enc_out=enc_out,y=ys,valid=False)
            
            #keep last prediction
            y_pred = y_pred[:, -1, :]/temperature
            #top p logits
            
            logits,indices = top_k_top_p_filtering(y_pred,top_p = top_p)

            #print('logits',logits)
            #choose only top k, filter others, no need for this step
            next_words,scores = sampling(scores,logits,indices)
            


            if batched_data.lpe_input.size(0) ==1:
                batched_data.lpe_input = batched_data.lpe_input.repeat(topk,1,1,1)
                batched_data.lpe_eigenval = batched_data.lpe_eigenval.repeat(topk,1,1,1)
                batched_data.central_input = batched_data.central_input.repeat(topk,1,1)

            #input nextwords, cor_indx, old_ys
            #concat next words to y, if top_k nextword == stopword, append to prediction
            #if prediction size == beam break, return false
            
            ys,prediction,scores,break_flag,batched_data =update_y(next_words,ys,prediction,scores,batched_data)
            
            if break_flag == False:
                break

                
            # new attn_bias and subsequent mask, lpe, cent padding for each iteration
            #[batch,sequence_length,sequence_length]
            batchsize = ys.size(0)

            batched_data.y_attn_bias =torch.zeros((batchsize,i+2,i+2), dtype=torch.float,device = self.device)
            batched_data.subsequent_mask = self.get_square_subsequent_mask(i+2)

            M = i+1
            lpe_input_pad = torch.zeros((batchsize,M,M,30),dtype=torch.float,device= self.device)
            lpe_eigenval_pad = torch.zeros((batchsize,M,M,30),dtype=torch.float,device= self.device)
            central_input_pad = torch.zeros((batchsize,M,M),dtype=torch.long,device=self .device)

            # copy old data to new padding
            # if old input pad dim ==[1,M,M,30], new should be[beam,M,M,30]
            lpe_input_pad[:,:-1,:-1] = batched_data.lpe_input
            lpe_eigenval_pad[:,:-1,:-1] = batched_data.lpe_eigenval
            central_input_pad[:,:-1,:-1] = batched_data.central_input
        


            for j in range(topk):
                next_word = next_words[j] 
                
                if next_word!=0 and next_word> 472+offset and next_word<(505+offset):
                    #if is edge, calculate new lpe,cent
                    
                    lpe_input,lpe_eigenval,central_input = self.make_laplacian_cent_attn_bias(30,ys[j,1:])

                    
                    #print(lpe_input)
                    # add new lpe, cent to the last row
                    lpe_input_pad[j,-1] = lpe_input
                    lpe_eigenval_pad[j,-1] = lpe_eigenval
                    central_input_pad[j,-1] = central_input

                else:
                    #if not edge, copy last line of batched data to the last line of padding
                    # last line, till -1 token
                    
                    lpe_input_pad[j,-1,:-1] = batched_data.lpe_input[j,-1] # [Batch, row, col, hidden]
                    lpe_eigenval_pad[j,-1,:-1] = batched_data.lpe_eigenval[j,-1]
                    central_input_pad[j,-1,:-1] = batched_data.central_input[j,-1]
        
            # update batch data with newly calculated lpe,cent
            batched_data.lpe_input = lpe_input_pad
            batched_data.lpe_eigenval = lpe_eigenval_pad
            batched_data.central_input = central_input_pad
        prediction.sort(key = lambda x: x[1], reverse=True)

        
        try:
            prediction = prediction[:topk]
        except:
            prediction = prediction
        correct = 0
        
        return {
            'y_pred': prediction,
            'correct': correct,
            'y_true': y_true,
            'idx': batched_data.idx,
        }
 
    def sampling(self,scores,logits,indices,topk):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(gen.seed())
        next_tokens = torch.multinomial(logits, 1, generator=gen)
        next_token, ind = torch.min(next_tokens, -1, keepdim=False, out=None)

        # map logits index to actual onehot index
        next_words = torch.zeros((topk),dtype=torch.long,device=self.device)
        
        for i in range(indices.size(0)):
            next_words[i] = indices[i][next_token[i]]
            scores[i] =  torch.log(logits[i][next_token[i]]) + scores[i]
        return next_words,scores
    def update_y(self,next_words,ys,prediction,scores,batched_data,topk,offset): 
        break_flag = True
        #create new input array
        new_ys = torch.zeros((topk,ys.size(-1)+1),dtype=torch.long,device=self.device)
        # if words is start token, repeat topk times
        if ys.size(0)==1:
            ys=ys.repeat(topk,1)
            #if length of prediction bucket >= topk stop prediction
        if len(prediction)>=(topk):
            break_flag = False
        
        #loop through k options, topk actually means pick k not topk 
        for i in range(topk):
    
            #if stop words
            if next_words[i] == 506+offset and scores[i]>-9999:
                prediction.append((ys[i].detach().clone().cpu().numpy(),scores[i].item()))
                scores[i] = -9999
                prediction.sort(key = lambda x: x[1], reverse=True)

            new_ys[i] = torch.cat((ys[i],next_words[i].reshape(1)))

        ys = new_ys.detach().clone()
        return ys,prediction,scores,break_flag,batched_data
    def top_k_top_p_filtering(self,logits, top_p=0.0, filter_value=-float('Inf')):
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_logits= F.softmax(sorted_logits,dim=-1)
            cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            sorted_indices_to_keep = ~sorted_indices_to_remove
            
            logits = sorted_logits*sorted_indices_to_keep.int().float()
            #print(logits[0])
        return logits,sorted_indices

    def test_step(self, batched_data, batch_idx): #split
        start =time.time()
        # if batched_data.idx in self.keydic:
        #     return None
        def choose_topk(scores,new_scores):

            new_scores = torch.log(new_scores)+scores

            topk_scores,topk_indx = torch.topk(new_scores.view(-1),beam)
            scores = topk_scores.reshape(beam,1)
            topk_r,topk_c = topk_indx//beam,topk_indx%beam
            topk_next_words = all_next_word[topk_r,topk_c]

            return topk_next_words.view(-1),topk_r,scores

        def sampling(scores,logits,indices):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(gen.seed())
            next_tokens = torch.multinomial(logits, 1, generator=gen)
            next_token, ind = torch.min(next_tokens, -1, keepdim=False, out=None)
            # next_token = next_tokens[:,0]
            #next_token = next_token.view(-1)
            # map logits index to actual onehot index
            next_words = torch.zeros((topk),dtype=torch.long,device=self.device)
            
            for i in range(indices.size(0)):
                next_words[i] = indices[i][next_token[i]]
                scores[i] =  torch.log(logits[i][next_token[i]]) + scores[i]
            return next_words,scores
        def update_y(next_words,ys,prediction,scores,batched_data):
            break_flag = True
            #create new input array
            new_ys = torch.zeros((topk,ys.size(-1)+1),dtype=torch.long,device=self.device)
            # if words is start token, repeat topk times
            if ys.size(0)==1:
                ys=ys.repeat(topk,1)
             #if length of prediction bucket >= topk stop prediction
            if len(prediction)>=(topk):
                break_flag = False
            
            #loop through k options, topk actually means pick k not topk 
            for i in range(topk):
        
                #if stop words
                if next_words[i] == 506+offset and scores[i]>-9999:
                    #print("stoppppusingthis")
                    #raise Exception("stop")
                    prediction.append((ys[i].detach().clone().cpu().numpy(),scores[i].item()))
                    scores[i] = -9999
                    prediction.sort(key = lambda x: x[1], reverse=True)

                new_ys[i] = torch.cat((ys[i],next_words[i].reshape(1)))

            ys = new_ys.detach().clone()
            #print(ys)
            return ys,prediction,scores,break_flag,batched_data
        def top_k_top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
            top_k_filter = 4
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_logits= F.softmax(sorted_logits,dim=-1)
                cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
               
                sorted_indices_to_keep = ~sorted_indices_to_remove
                
                sorted_indices_to_keep[...,top_k_filter:] = 0

                logits = sorted_logits*sorted_indices_to_keep.int().float()
                #print(logits[0])
            # if top_k_filter >0:

            return logits,sorted_indices

        offset = 4
        "=======Inference========"
       
        # topk actually means pick k not topk 
        topk = 50
        y_true = batched_data.y_gt.view(-1)
        if len(y_true)>160 and len(y_true)<=200:
            print(batch_idx,len(y_true),"case1")
            
        elif len(y_true)>200 and len(y_true)<350:
            topk = 30
            print(batch_idx,len(y_true),"case2")
        elif len(y_true)>350 :
            print(batch_idx,len(y_true),"case3")
            topk = 10

        temperature = 4.5
        prediction = []
        
        repeat_start =time.time()
        # top_p = 0.5 + jklm*0.02
        top_p = 0.75
        with torch.no_grad():
            encoder_start = time.time()
            if self.weak_ensemble == 1:
                ys = torch.randint(low=532,high=582,size=(topk,1),dtype=torch.long,device=self.device)
            else:
                ys = torch.tensor([[532]],dtype=torch.long,device=self.device) + batched_data.reverse
                ys = ys.repeat(topk,1)

            y_true = batched_data.y_gt.view(-1)
            if self.weak_ensemble == 1:
                batched_data.reverse = torch.randint(low=0,high=50,size=(topk,),dtype=torch.long,device=self.device)
            else:
                batched_data.reverse = batched_data.reverse.repeat(topk,1)

            batched_data.central_input = torch.zeros((topk,1,1),dtype=torch.long,device = self.device)
            batched_data.lpe_input = torch.zeros((topk,1,1,30),dtype=torch.float,device = self.device)
            batched_data.lpe_eigenval = torch.zeros((topk,1,1,30),dtype=torch.float,device = self.device)
            batched_data.y_attn_bias = torch.zeros((topk,1,1), dtype=torch.float,device = self.device)
            batched_data.subsequent_mask =None
            enc_out = self.translate_encoder(batched_data,beam=topk,valid=True)
            
            # temporary vars which hold the attn_bias info of last iteration
            lpe_input = torch.zeros((topk,1,30),dtype=torch.float,device= self.device)
            lpe_eigenval = torch.zeros((topk,1,30),dtype=torch.float,device= self.device)
            central_input = torch.zeros((topk,1),dtype=torch.long,device= self.device)

            scores = torch.zeros((topk,1),dtype=torch.float,device = self.device)
            encoder_end = time.time()
            decoder_start = time.time()
            for i in range(len(y_true)+70):  
                y_pred = self.translate_decoder(batched_data,enc_out=enc_out,y=ys,valid=True)
                #keep last prediction
                y_pred = y_pred[:, -1, :]/temperature
                #top p logits
                logits,indices = top_k_top_p_filtering(y_pred,top_p = top_p)

                #choose only top k, filter others, no need for this step
                next_words,scores = sampling(scores,logits,indices)
                if batched_data.lpe_input.size(0) ==1:
                    batched_data.lpe_input = batched_data.lpe_input.repeat(topk,1,1,1)
                    batched_data.lpe_eigenval = batched_data.lpe_eigenval.repeat(topk,1,1,1)
                    batched_data.central_input = batched_data.central_input.repeat(topk,1,1)


                ys,prediction,scores,break_flag,batched_data =update_y(next_words,ys,prediction,scores,batched_data)
                
                if break_flag == False:
                    break

                    
                # new attn_bias and subsequent mask, lpe, cent padding for each iteration
                #[batch,sequence_length,sequence_length]
                batchsize = ys.size(0)

                batched_data.y_attn_bias =torch.zeros((batchsize,i+2,i+2), dtype=torch.float,device = self.device)
                batched_data.subsequent_mask = self.get_square_subsequent_mask(i+2)

                M = i+1
                lpe_input_pad = torch.zeros((batchsize,M,M,30),dtype=torch.float,device= self.device)
                lpe_eigenval_pad = torch.zeros((batchsize,M,M,30),dtype=torch.float,device= self.device)
                central_input_pad = torch.zeros((batchsize,M,M),dtype=torch.long,device=self .device)

                # copy old data to new padding
                # if old input pad dim ==[1,M,M,30], new should be[beam,M,M,30]
                lpe_input_pad[:,:-1,:-1] = batched_data.lpe_input
                lpe_eigenval_pad[:,:-1,:-1] = batched_data.lpe_eigenval
                central_input_pad[:,:-1,:-1] = batched_data.central_input
            

                start_lpe = time.time()
                for j in range(topk):
                    next_word = next_words[j] 
                    
                    if next_word!=0 and next_word> 472+offset and next_word<(505+offset):
                        #if is edge, calculate new lpe,cent
                        
                        lpe_input,lpe_eigenval,central_input = self.make_laplacian_cent_attn_bias(30,ys[j,1:])
                        #print(lpe_input)
                        # add new lpe, cent to the last row
                        lpe_input_pad[j,-1] = lpe_input
                        lpe_eigenval_pad[j,-1] = lpe_eigenval
                        central_input_pad[j,-1] = central_input

                    else:
                        #if not edge, copy last line of batched data to the last line of padding
                        # last line, till -1 token         
                        lpe_input_pad[j,-1,:-1] = batched_data.lpe_input[j,-1] # [Batch, row, col, hidden]
                        lpe_eigenval_pad[j,-1,:-1] = batched_data.lpe_eigenval[j,-1]
                        central_input_pad[j,-1,:-1] = batched_data.central_input[j,-1]
            
                # update batch data with newly calculated lpe,cent
                batched_data.lpe_input = lpe_input_pad
                batched_data.lpe_eigenval = lpe_eigenval_pad
                batched_data.central_input = central_input_pad
                end_lpe = time.time()
            decoder_end = time.time()
        repeat_end = time.time()

        prediction.sort(key = lambda x: x[1], reverse=True)


        correct = 0

        end = time.time()

        return {
            'y_pred': prediction,
            'correct': correct,
            'y_true': y_true,
            'idx': batched_data.idx,
        }


    def test_step_beam(self, batched_data, batch_idx):#beam
        beam = 5
        topk = 10
        def choose_topk(scores,new_scores):
            new_scores = torch.log(new_scores)+scores

            topk_scores,topk_indx = torch.topk(new_scores.view(-1),beam)
            scores = topk_scores.reshape(beam,1)
            topk_r,topk_c = topk_indx//beam,topk_indx%beam
            topk_next_words = all_next_word[topk_r,topk_c]
            return topk_next_words.view(-1),topk_r,scores
        def choose_topk_sampling(scores,new_scores):
            new_scores = torch.log(new_scores)+scores

            temperature=1
            prob=F.softmax(new_scores.view(-1)/temperature,dim=-1)
            topk_indx = torch.multinomial(prob,beam)
            topk_scores = torch.zeros((len(topk_indx),1),dtype=torch.float,device=self.device)
            new_scores = new_scores.view(-1)
            for i in range(len(topk_indx)):
                topk_scores[i] = new_scores[topk_indx[i]]

            scores = topk_scores.reshape(beam,1)
            topk_r,topk_c = topk_indx//beam,topk_indx%beam
            topk_next_words = all_next_word[topk_r,topk_c]
            #print("topk_next_words",topk_next_words)
            return topk_next_words.view(-1),topk_r,scores   
        def update_y(next_words,old_ys_indxes,ys,prediction,scores,batched_data):
            break_flag = True
            new_ys = torch.zeros((beam,ys.size(-1)+1),dtype=torch.long,device=self.device)
            if ys.size(0)==1:
                ys=ys.repeat(beam,1)
             #check current lowest probability and highest in the results
            if len(prediction)>=topk:
                
                if scores[0].item()<prediction[-1][1]:
                    break_flag = False
                if scores[0]<=-99999:
                    break_flag = False
            
            for i in range(beam):
                old_ys_indx = old_ys_indxes[i]
                #if stop words
                if next_words[i] == 506+offset:
                    #raise Exception("stop")
                    prediction.append((ys[old_ys_indx].detach().clone().cpu().numpy(),scores[i].item()))
                    prediction.sort(key = lambda x: x[1], reverse=True)
                    scores[i] = -99999

                batched_data.lpe_input[i] = batched_data.lpe_input[old_ys_indx]
                batched_data.lpe_eigenval[i] = batched_data.lpe_eigenval[old_ys_indx]
                batched_data.central_input[i] = batched_data.central_input[old_ys_indx]

                new_ys[i] = torch.cat((ys[old_ys_indx],next_words[i].reshape(1)))

            ys = new_ys.detach().clone()
            return ys,prediction,scores,break_flag,batched_data


        offset = 4
        "=======Inference========"
        
        with torch.no_grad():

            ys = torch.tensor([[532]],dtype=torch.long,device=self.device)
            
            y_true = batched_data.y_gt.view(-1)
            #prepare first item's input data
            batched_data.central_input = torch.zeros((1,1,1),dtype=torch.long,device = self.device)
            batched_data.lpe_input = torch.zeros((1,1,1,30),dtype=torch.float,device = self.device)
            batched_data.lpe_eigenval = torch.zeros((1,1,1,30),dtype=torch.float,device = self.device)
            batched_data.y_attn_bias = torch.zeros((1,1,1), dtype=torch.float,device = self.device)
            batched_data.subsequent_mask =None

            # temporary vars which hold the attn_bias info of last iteration
            lpe_input = torch.zeros((1,1,30),dtype=torch.float,device= self.device)
            lpe_eigenval = torch.zeros((1,1,30),dtype=torch.float,device= self.device)
            central_input = torch.zeros((1,1),dtype=torch.long,device= self.device)
          
            enc_out = self.translate_encoder(batched_data,beam=beam,valid=True)
            prediction = []



        # try:
        scores = torch.zeros((1,1),dtype=torch.float,device = self.device)
        for i in range(len(y_true)+100):
            
            try:
                y_pred = self.translate_decoder(batched_data,enc_out=enc_out,y=ys,valid=True)
            except:
                print("sadsdasd")
                print(i,len(prediction),ys)
                raise Exception()
            #top k
            temperature = 1
            new_scores, all_next_word = torch.topk(torch.nn.functional.softmax(y_pred/temperature,dim=-1),beam,dim=-1)
        
            new_scores = new_scores[:,-1,:beam]
            all_next_word = all_next_word[:,-1,:beam]
            next_words,old_ys_indxes,scores = choose_topk(scores,new_scores)

            #input nextwords, cor_indx, old_ys
            #concat next words to y, if top_k nextword == stopword, append to prediction
            #if prediction size == beam break, return false
            if batched_data.lpe_input.size(0) ==1:
                batched_data.lpe_input = batched_data.lpe_input.repeat(beam,1,1,1)
                batched_data.lpe_eigenval = batched_data.lpe_eigenval.repeat(beam,1,1,1)
                batched_data.central_input = batched_data.central_input.repeat(beam,1,1)
            
            ys,prediction,scores,break_flag,batched_data =update_y(next_words,old_ys_indxes,ys,prediction,scores,batched_data)
                
            if break_flag == False:
                break

                
            # new attn_bias and subsequent mask, lpe, cent padding for each iteration
            #[batch,sequence_length,sequence_length]
            batchsize = ys.size(0)

            batched_data.y_attn_bias =torch.zeros((batchsize,i+2,i+2), dtype=torch.float,device = self.device)
            batched_data.subsequent_mask = self.get_square_subsequent_mask(i+2)

            M = i+1
            lpe_input_pad = torch.zeros((batchsize,M,M,30),dtype=torch.float,device= self.device)
            lpe_eigenval_pad = torch.zeros((batchsize,M,M,30),dtype=torch.float,device= self.device)
            central_input_pad = torch.zeros((batchsize,M,M),dtype=torch.long,device=self .device)

            # copy old data to new padding
            # if old input pad dim ==[1,M,M,30], new should be[beam,M,M,30]
            lpe_input_pad[:,:-1,:-1] = batched_data.lpe_input
            lpe_eigenval_pad[:,:-1,:-1] = batched_data.lpe_eigenval
            central_input_pad[:,:-1,:-1] = batched_data.central_input

            for j in range(beam):
                next_word = next_words[j] 
                
                if next_word!=0 and next_word> 472+offset and next_word<(505+offset):
                    #if is edge, calculate new lpe,cent
                    
                    lpe_input,lpe_eigenval,central_input = self.make_laplacian_cent_attn_bias(30,ys[j,1:])
                    # add new lpe, cent to the last row
                    lpe_input_pad[j,-1] = lpe_input
                    lpe_eigenval_pad[j,-1] = lpe_eigenval
                    central_input_pad[j,-1] = central_input

                else:
                    #if not edge, copy last line of batched data to the last line of padding
                    # last line, till -1 token
                    
                    lpe_input_pad[j,-1,:-1] = batched_data.lpe_input[j,-1] # [Batch, row, col, hidden]
                    lpe_eigenval_pad[j,-1,:-1] = batched_data.lpe_eigenval[j,-1]
                    central_input_pad[j,-1,:-1] = batched_data.central_input[j,-1]
        
            # update batch data with newly calculated lpe,cent
            batched_data.lpe_input = lpe_input_pad
            batched_data.lpe_eigenval = lpe_eigenval_pad
            batched_data.central_input = central_input_pad
        prediction.sort(key = lambda x: x[1], reverse=True)

        try:
            prediction = prediction[:50]
        except:
            prediction = prediction
        correct = 0
        
        return {
            'y_pred': prediction,
            'correct': correct,
            'y_true': y_true,
            'idx': batched_data.idx,
        }

    def test_epoch_end(self, outputs):
        """
        outputs: list of individual outputs of each validation step.
        """
        import pickle
        import os
        total = 0
        correct = 0
        out_dic = {}
        i = 0
        path = self.inference_path
        
        # path ="g2gt_github/src/results/typed_uspto50k_split2"
        if not os.path.exists(path):
            os.mkdir(path)

        while os.path.exists(path+"out%s" % i):
            i += 1
        filename = path+"out"

        outfile = open(f"{filename}{i}","wb")
        for i in outputs:
            if i!=None:
                total +=1
                correct += i["correct"]
                idx = i['idx'][0].item()
                gt = i["y_true"].detach().cpu().numpy()
                results = i['y_pred']
                out_dic[idx] = [results,gt]
                #print(type(idx),idx,type(results))
        pickle.dump(out_dic,outfile)
        outfile.close()
        
        acc = correct/total
        self.log('inference accuracy', acc, sync_dist=True)

    def test_epoch_end_(self, outputs):
        import pickle
        import os
        total = 0
        correct = 0
        out_dic = {}
        
        i = 0
        path ="/da1/G2GT/src/results/uspto-full-aug/"
        if not os.path.exists(path):
            os.mkdir(path)

        while os.path.exists(path+"out%s" % i):
            i += 1
        filename = path+"out"
        outfile = open(f"{filename}{i}","wb")
        for i in outputs:
            total +=1
            correct += i["correct"]
            idx = i['idx'][0].item()
            gt = i["y_true"].detach().cpu().numpy()
            results = i['enc_out'].detach().cpu().numpy()
            out_dic[idx] = [results,gt]
            #print(type(idx),idx,type(results))
        pickle.dump(out_dic,outfile)
        outfile.close()
        
        

        acc = correct/total
        self.log('inference accuracy', acc, sync_dist=True)



    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=2.0,
            ),
            'name': 'learning_rate',
            'interval':'step',
            'frequency': 1,
        }
        
        return [optimizer],[lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GraphFormer")
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--head_size', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--ffn_dim', type=int, default=512)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--attention_dropout_rate', type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-5)
        parser.add_argument('--edge_type', type=str, default='multi_hop')
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--flag', action='store_true')
        parser.add_argument('--flag_m', type=int, default=3)
        parser.add_argument('--flag_step_size', type=float, default=1e-3)
        parser.add_argument('--flag_mag', type=float, default=1e-3)
        parser.add_argument('--beam', type=int, default=1)
        parser.add_argument('--inference_path', type=str, default="g2gt_github/src/results/typed_uspto50k_split2")
        return parent_parser

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x, valid=False):
        temp = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        temp = self.dropout(temp)
        x = x + temp
        # if valid:
        #     return x
        return x





class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, subsq_mask = None, valid=False,check=False):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)


        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        q = q * self.scale

        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        if subsq_mask is not None:
            x = x + subsq_mask

        x = torch.softmax(x, dim=-1)

        if not valid:
            x = self.att_dropout(x)

        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None,valid=False):
        y = self.self_attention_norm(x)
        
        y = self.self_attention(y, y, y, attn_bias,valid=True)
        if not valid:
            y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        if not valid:
            y = self.ffn_dropout(y)
        x = x + y
        return x
'''Important'''    
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(DecoderLayer, self).__init__()
        
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        '''Masked Attention'''
        self.self_mask_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        '''Mem Attention'''
        self.mem_att_sublayer_norm = torch.nn.LayerNorm(hidden_size)
        self.self_mem_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_mem_attention_dropout = nn.Dropout(dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    '''todo self_mem_attention, and masked self_attention '''
    def forward(self, x, memk,memv, tgt_padding_mask,attn_bias=None,valid=False,check=False):
        # if check:
        #     print("X0",x[0][20])

        y = self.self_attention_norm(x)

        y = self.self_mask_attention(y,y,y,attn_bias,subsq_mask=tgt_padding_mask,valid=valid,check=check)
        # if check:
        #     print("X2",y[0][20])
        if not valid:
            y = self.self_attention_dropout(y)
        x = x + y

        y = self.mem_att_sublayer_norm(x)
        y = self.self_mem_attention(y,memk,memv,valid=valid)
        if not valid:
            y = self.self_mem_attention_dropout(y)
        x = x+y
        # if check:
        #     print("X2",x[0][20])
        y = self.ffn_norm(x)
        y = self.ffn(y)
        if not valid:
            y = self.ffn_dropout(y)
        x = x + y
        # if check:
        #     print("X3",x[0][20])
        return x
