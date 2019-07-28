
import torch

import torch.nn as nn
import networkx as nx
from support import *
from spectral_clusttering import KMeans
import numpy as np
import keras
import time

from joblib import Parallel, delayed
import multiprocessing
"""
S - Similarity/Adjacency torch Matrix 
R - DNN layers number (each layer of size n
"""

def par_func(h,col_idx,Xj):
    Hsize = (Xj.shape[0],1) #int(Xj.shape[1]/2

    sae = SpectralAutoEncoder_keras(Xj[:,col_idx].reshape(Xj.shape[0],1),Hsize)

    sae.fit()
    return sae.predict().flatten()


class AutoEncoderClustering():
    def __init__(self,S):
        self.D = diag_mat(S)
        self.S = S
        self.Dinv_mm_S = np.matmul(np.linalg.inv(self.D), self.S)

    def run(self,K=10,dims=2):

        Xj = self.Dinv_mm_S[:, :]
        for i in range(0,3):
            start_t = time.time()
            h = np.zeros(shape=(Xj.shape[0],int(Xj.shape[1]/2)))

            results = Parallel(n_jobs=6)(delayed(par_func)(h,col_idx,Xj) for col_idx in range(0,h.shape[1]))

            # for col_idx in range(0,h.shape[1]):
            #     Hsize = (Xj.shape[0],1) #int(Xj.shape[1]/2
            #
            #     sae = SpectralAutoEncoder_keras(Xj[:,col_idx].reshape(Xj.shape[0],1),Hsize)
            #
            #     sae.fit()
            #     h[:,col_idx] = sae.predict().flatten()

            for col in range(0,h.shape[1]):
                h[:, col] = results[col]

            Xj = h
            end_t = time.time()
            print("[AES] Iteration " + str(i) + " took " + str(end_t - start_t) + "sec")


        #     loss = 1000000
        #     while loss > 0.1:
        #         lr = 0.01
        #         loss = sae.train(Xj,lr)
        #        # print("loss: "+str(loss) + " learning rate: "+str(lr))
        #
        #     h,pred_y = sae.forward(Xj)
        #     end_t = time.time()
        #     print("[AES] Iteration "+str(i)+" took "+str(end_t-start_t)+"sec")
        #     print("[AES] loss: " + str(loss) + " learning rate: " + str(lr))
        #     Xj = h
        #
        x = Xj[:, 1:(dims + 1)]
        classes, centroids = KMeans(x, K)
        return classes





class SpectralAutoEncoder_keras():
    def __init__(self,X,Hsize,learning_rate = 1e-4):
        self.encoding_dim =  Hsize[0] * Hsize[1]
        self.decoding_dim = X.shape[0] * X.shape[1]
        self.input_shape = (X.shape[0] * X.shape[1],)

        self.autoencoder = keras.models.Sequential([
            keras.layers.Dense(self.encoding_dim,input_shape=self.input_shape),#encoded
            keras.layers.Activation('sigmoid'),#encoded_act
            keras.layers.Dense(self.decoding_dim),#decoded_layer
            keras.layers.Activation('sigmoid') #decoded act
        ])

        self.Xflatten_size = X.shape[0]*X.shape[1]
        self.X = X

    def fit(self,epoch = 2, batch_size=200):
        self.autoencoder.compile(optimizer='adam',loss='mse')
        self.autoencoder.fit(self.X.flatten().reshape(1,self.Xflatten_size),self.X.flatten().reshape(1,self.Xflatten_size),verbose=0,epochs=epoch,batch_size=batch_size)

    def predict(self):
        h_model = keras.models.Sequential([
            keras.layers.Dense(self.encoding_dim, input_shape=self.input_shape,weights=self.autoencoder.layers[0].get_weights()),  # encoded
            keras.layers.Activation('sigmoid') ])  # encoded_act

        H = h_model.predict(self.X.flatten().reshape(1,self.Xflatten_size))

        return H


    def loss(self, y_pred, y, h):
        return torch.mean((y_pred - y) ** 2) + self.bkl(h)  # TODO add blk

    def bkl(self, h, ro=0.01):  # TODO write blk after getting activation values
        ro_avg = torch.mean(h)
        return ro * torch.log(ro_avg / ro) + (1 - ro) * torch.log((1 - ro) / (1 - ro_avg))


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

        self.cuda = False# torch.cuda.is_available()

        if self.cuda:
            self.encoder_grid = nn.Linear(X.shape[0] * X.shape[1], Hsize[0] * Hsize[1]).cuda()
            self.encoder_act = nn.Sigmoid().cuda()
            self.decoder_grid = nn.Linear(Hsize[0] * Hsize[1], X.shape[0] * X.shape[1]).cuda()
            self.decoder_act = nn.Sigmoid().cuda()
        else:
            self.encoder_grid =  nn.Linear(X.shape[0]*X.shape[1],Hsize[0]*Hsize[1])
            self.encoder_act = nn.Sigmoid()
            self.decoder_grid = nn.Linear(Hsize[0] * Hsize[1], X.shape[0] * X.shape[1])
            self.decoder_act = nn.Sigmoid()

        self.h_shape = Hsize

        self.learning_rate = learning_rate


    def forward(self,X):

        if self.cuda:
            print("[AES] using cuda")
            enc_grid_out = self.encoder_grid(X.flatten()).cuda()
            enc_act_out = self.encoder_act(enc_grid_out).cuda()
            h = enc_act_out
            dec_grid_out = self.decoder_grid(enc_act_out).cuda()
            y_pred = self.decoder_act(dec_grid_out).cuda()
        else:
            enc_grid_out = self.encoder_grid(X.flatten())
            enc_act_out = self.encoder_act(enc_grid_out)
            h = enc_act_out
            dec_grid_out = self.decoder_grid(enc_act_out)
            y_pred = self.decoder_act(dec_grid_out)

        return h.view(self.h_shape),y_pred.view(X.shape)

    def train(self,X,lr):
        h,y_pred = self.forward(X)
        if self.cuda:
            loss = self.loss(X, y_pred, h).cuda()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            loss.backward(retain_graph=True)
            optimizer.step()
        else:
            loss = self.loss(X, y_pred, h).cuda()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            loss.backward(retain_graph=True)
            optimizer.step()

        return loss


    def loss(self,y_pred,y,h):
        return torch.mean((y_pred-y)**2)+self.bkl(h) #TODO add blk

    def bkl(self,h,ro=0.01): #TODO write blk after getting activation values
        ro_avg = torch.mean(h)
        return ro*torch.log(ro_avg/ro)+(1-ro)*torch.log((1-ro)/(1-ro_avg))

    #def train(self):

