
import networkx as nx

import numpy as np
import torch
import random

#input ndarray
def choice_unique(in_ndarray,in_elements):
    in_ndarray_copy = in_ndarray[:]
    indexes = range(0,len(in_ndarray)-1)
    res = []
    for elem in range(0,in_elements):
        idx_to_remove = np.random.choice(indexes)
        new_elem = in_ndarray_copy[idx_to_remove]
        in_ndarray_copy = np.delete(in_ndarray_copy,idx_to_remove)
        res.append(new_elem)

    return res


def print_graph_data(G,Graph_Name = "No Name"):
    num_of_nodes = str(len(nx.nodes(G)))
    num_of_edges = str(len(nx.edges(G)))

    print("Graph -"+Graph_Name)
    print("Number of nodes: "+num_of_nodes)
    print("Number of edges: "+num_of_edges)
    if nx.is_connected(G):
        print("The Graph is connected")
    else:
        print("The Graph isn't connected")


def diag_mat(AdjMat):
    return torch.sum(AdjMat,dim=0).diag()