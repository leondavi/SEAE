import networkx as nx
import torch
from pytorch_bridge import *
from spectral_clusttering import *
import numpy as np

cuda_available = torch.cuda.is_available()

print("Is cuda available: "+str(cuda_available))


G = nx.random_regular_graph(6,100)

A = np.array(nx.adjacency_matrix(G).toarray())

At = torch.from_numpy(A)

NodesClasses = spectral_clusttering(At,5)
generate_k_clusttered_graph(NodesClasses)