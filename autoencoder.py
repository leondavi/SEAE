
import torch

import torch.nn as nn
import networkx as nx
from support import *

"""
S - Similarity/Adjacency torch Matrix 
R - DNN layers number (each layer of size n
"""
class AutoEncoderClustering():
    def __init__(self,S,R):
        self.D = diag_mat(S).float()
        self.S = S.float()
        self.R = R
        self.Dinv_mm_S = torch.mm(self.D.inverse(), self.S)

    def run(self):

        Xj = self.Dinv_mm_S[:, :]
        for i in range(0,self.R):
            Hsize = (Xj.shape[0],int(Xj.shape[1]/2))
            sae = SpectralAutoEncoder(Xj,Hsize)

            pred = sae.forward(Xj)
            sae.train(Xj)






"""
An auto encoder based on the algorithm from paper: 
Learning Deep Representations for Graph Clustering
by Fei Tian et al. 

Input: 
input layer to the autoencoder Xi
size of hidden layer as a tuple (n,n)

"""
class SpectralAutoEncoder(nn.Module):
    def __init__(self,X,Hsize,learning_rate = 1e-4):
        super().__init__()

        self.encoder_grid =  nn.Linear(X.shape[0]*X.shape[1],Hsize[0]*Hsize[1])
        self.encoder_act = nn.Sigmoid()
        self.decoder_grid = nn.Linear(Hsize[0] * Hsize[1], X.shape[0] * X.shape[1])
        self.decoder_act = nn.Sigmoid()

        self.learning_rate = learning_rate


    def forward(self,X):
        enc_grid_out = self.encoder_grid(X.flatten())
        enc_act_out = self.encoder_act(enc_grid_out)
        h = enc_act_out
        dec_grid_out = self.decoder_grid(enc_act_out)
        y_pred = self.decoder_act(dec_grid_out)
        return h,y_pred.view(X.shape)

    def train(self,X):
        h,y_pred = self.forward(X)
        loss = self.loss(X,y_pred,h)

    def loss(self,y_pred,y,h):
        return torch.mean((y_pred-y)**2)+self.bkl(h) #TODO add blk

    def bkl(self,h,ro=0.01): #TODO write blk after getting activation values
        ro_avg = torch.mean(h)
        return ro*torch.log(ro_avg/ro)+(1-ro)*torch.log((1-ro)/(1-ro_avg))

    #def train(self):

