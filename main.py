import torch
from pytorch_bridge import *
from spectral_clusttering import *
import numpy as np
from support import *
from autoencoder import *

cuda_available = torch.cuda.is_available()

print("Is cuda available: "+str(cuda_available))


G = nx.random_regular_graph(6,100)
#G = nx.complete_graph(50)
#print_graph_data(G,"Random Regular 6")
#G = nx.random_lobster(100,0.5,0.5)

A = np.array(nx.adjacency_matrix(G).toarray())

At = torch.from_numpy(A)



#
# print_graph_data(G_spectral_clusttered,"Spectral Clusttered Graph")

NodesClasses = spectral_clusttering(At,3)
G_spectral_clusttered = generate_k_clusttered_graph(NodesClasses,"Spectral Clusttered Graph")
print_graph_data(G_spectral_clusttered,"Spectral Clusttered Graph")

aec_ = AutoEncoderClustering(At)
aes_classes = aec_.run(3)
G_aes_clusttered = generate_k_clusttered_graph(aes_classes,"AES Graph")
print_graph_data(G_aes_clusttered,"AES Graph")

plt.show()