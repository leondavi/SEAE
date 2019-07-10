import torch
from matplotlib import pyplot as plt
import time

def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = torch.Tensor( x[:,None,:] )  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = torch.Tensor( c[None,:,:] ).float()  # (1, Nclusters, D)
        D_ij = ((x_i - c_j)**2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl  = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl.float()

    end = time.time()

    if verbose:
        print("K-means input with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                Niter, end - start, Niter, (end-start) / Niter))

    return cl, c



"""
S - Input Similarity Matrix - sparse pytorch matrix

Return - Matrix of eigen vectors 
"""
def spectral_embedding(S,dims=2):
    D = torch.diag(S)
    L = D-S
    (eigenvalues,eigenvectors) = torch.symeig(L.float(),eigenvectors=True)

    return eigenvectors

def spectral_clusttering(S,K=10,dims=2):


    eigenvectors = spectral_embedding(S,dims)
    dims = min(eigenvectors.shape[-1]-1,dims) # 0 doesn't count
    x = eigenvectors[:,1:(dims+1)]
    classes, centroids = KMeans(x,K) #taking eigenvectors (without 0 column)
    if dims == 2:
        plt.figure()
        plt.title("2D projection")
        plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c=classes.cpu(), cmap="tab10")
        plt.scatter(centroids[:, 0].cpu(), centroids[:, 1].cpu(), c='black', s=50, alpha=.8)
        plt.show()

    pass