import torch
from pytorch_bridge import *
from spectral_clusttering import *
import numpy as np
from support import *
from autoencoder import *
from traces_extraction import *
from experiment import Simulation

# Functions

def generate_graph_by_activity_pairs(nodes_pairs):
    G = nx.Graph()
    for pair in nodes_pairs:
        G.add_edge(int(pair[0]),int(pair[1]))
    return G

def generate_spectral_clusttering_graph(Smat,no_of_clusters):
    NodesClasses = spectral_clusttering(Smat,no_of_clusters)
    G_spectral_clusttered = generate_k_clusttered_graph(NodesClasses,"Spectral Clusttered Graph")
    return G_spectral_clusttered

def generate_aes_clusttering_graph(Smat,no_of_clusters):
    aec_ = AutoEncoderClustering(Smat)
    aes_classes = aec_.run(no_of_clusters)
    G_aes_clusttered = generate_k_clusttered_graph(aes_classes)
    return G_aes_clusttered


#Initializations


Experiments = dict()
Experiments["traceA"] = "data/Trace_In_cluster_A_segment_9_{Time,Src,Dst}_.csv"
#Experiments["traceB"] = "data/Trace_In_cluster_B_segment_9_{Time,Src,Dst}_.csv"






cuda_available = torch.cuda.is_available()

print("Is cuda available: "+str(cuda_available))

#Experiment start

CurrentTraceInst = trace("traceA", "data/Trace_In_cluster_A_segment_9_{Time,Src,Dst}_.csv")
CurrentTraceInst.extract_statistics()
Sm_actMat_torch = CurrentTraceInst.get_similarity_matrix_by_activity()

nodes_hash = CurrentTraceInst.get_convert_hash()
trace_table = CurrentTraceInst.get_trace_table()
nodes_pairs = CurrentTraceInst.get_list_of_pairs_nodes()
nodes_activity = CurrentTraceInst.get_nodes_activity()

#generate graph types

Graphs = dict()
Graphs["Random Regular 6"] = nx.random_regular_graph(6,Sm_actMat_torch.shape[1])
Graphs["Activity pairs based"] = generate_graph_by_activity_pairs(nodes_pairs)
for clusters in range (2,6):
    Graphs["Spectral embedding - "+str(clusters)+"clusters"]
    Graphs["AutoEncoder embedding - "+str(clusters)+"clusters"]


G_rand_reg_6 = nx.random_regular_graph(6,Sm_actMat_torch.shape[1])

sim = Simulation(G_rand_reg_6,nodes_activity,nodes_hash,trace_table)
print("Avg # of hops: "+str(sim.run_avg_dist()))
print_graph_data(G_rand_reg_6)




#
# print_graph_data(G_spectral_clusttered,"Spectral Clusttered Graph")




plt.show()

