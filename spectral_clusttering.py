import torch
from matplotlib import pyplot as plt
import time
import networkx as nx
from support import *
from itertools import combinations

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

    # if verbose:
    #     print("K-means input with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
    #     print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
    #             Niter, end - start, Niter, (end-start) / Niter))

    return cl, c



"""
S - Input Similarity Matrix - sparse pytorch matrix

Return - Matrix of eigen vectors 
"""
def spectral_embedding(S,dims=2):
    D = diag_mat(S)
    #D = torch.diag(S)
    L = D-S
    (eigenvalues,eigenvectors) = torch.symeig(L.float(),eigenvectors=True)

    return eigenvectors

def spectral_clusttering(S,K=10,dims=2):
    start_t = time.time()
    eigenvectors = spectral_embedding(S,dims)
    dims = min(eigenvectors.shape[-1]-1,dims) # 0 doesn't count
    x = eigenvectors[:,1:(dims+1)]
    classes, centroids = KMeans(x,K) #taking eigenvectors (without 0 column)
    # if dims == 2:
    #     plt.figure()
    #     plt.title("2D projection")
    #     plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c=classes.cpu(), cmap="tab10")
    #     plt.scatter(centroids[:, 0].cpu(), centroids[:, 1].cpu(), c='black', s=50, alpha=.8)
    #     plt.show()
    end_t = time.time()
    print("[SVD] " + str(end_t - start_t) + "sec")
    return classes #returns vector of nodes, each index is node's class number

def generate_k_clusttered_graph(classes,graph_name=None):
    G = nx.Graph()
    num_of_nodes = classes.shape[0]
    G.add_nodes_from(list(range(num_of_nodes)))
    unique_classes = torch.unique(classes)
    gateways_nodes = np.ndarray(shape=(0,2))
    for c in unique_classes:
        nodes_of_current_class = (classes == c).nonzero()
        subG = nx.subgraph(G,nodes_of_current_class.numpy().flatten())
        percentage = 0.1
        newG = nx.fast_gnp_random_graph(nx.number_of_nodes(subG),percentage)
        while (not nx.is_connected(newG)):
            percentage += 0.05
            newG = nx.fast_gnp_random_graph(nx.number_of_nodes(subG),percentage)
            G.add_edges_from(list(newG.edges))
        list_of_edges_to_add = []
        for edge in newG.edges:
            list_of_edges_to_add.append((list(subG.nodes)[edge[0]], list(subG.nodes)[edge[1]]))
        G.add_edges_from(list_of_edges_to_add)

        #gateways_nodes = np.vstack([gateways_nodes, np.random.choice(nodes_of_current_class.numpy().flatten(), 2)])
        gateways_nodes = np.vstack([gateways_nodes, choice_unique(nodes_of_current_class.numpy().flatten(), 2)])

    #TODO connect clusters

    edges_to_add = list(combinations(gateways_nodes.T[0], 2))

    for edge in edges_to_add:
        G.add_edge(edge[0],edge[1])
    return G

def generate_graph_plot(G,GraphName):
    plt.figure()
    if GraphName == None:
        plt.title("clusterred graph")
    else:
        plt.title(GraphName)
    nx.draw(G)




