
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
            pred





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

        self.encoder = nn.Sequential(
                        nn.Linear(X.shape[0]*X.shape[1],Hsize[0]*Hsize[1]),
                        nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
                        nn.Linear(Hsize[0] * Hsize[1], X.shape[0] * X.shape[1]),
                        nn.Sigmoid()
        )
        self.learning_rate = learning_rate



    def forward(self,X):
        h = self.encoder(X)
        y = self.decoder(h)
        return y

    #def train(self):