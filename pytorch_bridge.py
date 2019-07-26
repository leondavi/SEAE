import torch
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx


class Pytorch_Bridge:
    def __init__(self):
        pass

    """
    csr matrix to pytorch sparse
    Taken from 
    https://gist.github.com/aesuli/319d71707a5ee96086aa2439b87d4e38 
    """
    @staticmethod
    def csr_matrix_to_pytorch_sparse(CsrMat):
        Acoo = CsrMat.tocoo()
        return torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                              torch.LongTensor(Acoo.data.astype(np.int32)))
    @staticmethod
    def nx_to_torch_adjecency(G):
        return torch.from_numpy(np.array(nx.adjacency_matrix(G).toarray()))

