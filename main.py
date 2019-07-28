import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from spectral_clusttering import *
from autoencoder import *
from traces_extraction import *
from experiment import Simulation
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

info_st_ = "[INFO] "
prog_st_ = "[PROGRESS]"
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


results = pd.DataFrame(columns=["trace_name","graph_name","clusters","simulation_avg_hops","num_of_edges","num_of_nodes","graph_is_connected"])

#Experiment start
print(info_st_+"Experiment starting: \n--------------------")

CurrentTraceInst = trace("traceA", "data/Trace_In_cluster_A_segment_9_{Time,Src,Dst}_.csv")
CurrentTraceInst.extract_statistics()
Sm_actMat_torch = CurrentTraceInst.get_similarity_matrix_by_activity()

nodes_hash = CurrentTraceInst.get_convert_hash()
trace_table = CurrentTraceInst.get_trace_table()
nodes_pairs = CurrentTraceInst.get_list_of_pairs_nodes()
nodes_activity = CurrentTraceInst.get_nodes_activity()
trace_name = CurrentTraceInst.get_trace_name()

#generate graph types
print("\n"+info_st_+"Generates graphs: \n ---------------------")
Graphs = dict()
Graphs["Random Regular 6"] = (nx.random_regular_graph(6,Sm_actMat_torch.shape[1]),0)
print(prog_st_+"0%")
Graphs["Activity pairs based"] = (generate_graph_by_activity_pairs(nodes_pairs),0)
graphs_similarities = []
for clusters in range (2,3):
    Graphs["Spectral embedding - "+str(clusters)+"clusters"] = (generate_spectral_clusttering_graph(Sm_actMat_torch,clusters),clusters)
    Graphs["AutoEncoder embedding - "+str(clusters)+"clusters"] = (generate_aes_clusttering_graph(Sm_actMat_torch,clusters),clusters)
    graphs_similarities.append(("Spectral embedding - "+str(clusters)+"clusters","AutoEncoder embedding - "+str(clusters)+"clusters"))
    print(prog_st_+str(100*((clusters+1)/5.))+"%")

#Graphs simulations and performances checks

print(info_st_+"Performs simulations on graphs and perfomance checks \n --------------------------------------------------------")
print(info_st_+"Results will be saved to file at the end")
for graph_name in Graphs.keys():
    current_G = Graphs[graph_name][0]
    clusters = Graphs[graph_name][1]
    sim = Simulation(current_G, nodes_activity, nodes_hash, trace_table)
    num_of_nodes = str(len(nx.nodes(current_G)))
    num_of_edges = str(len(nx.edges(current_G)))
    graph_is_connected = nx.is_connected(current_G)
    simulation_avg_hops = 0
    if graph_is_connected:
        simulation_avg_hops = sim.run_avg_dist()

    results = results.append(pd.DataFrame([[trace_name,graph_name,clusters,simulation_avg_hops,num_of_edges,num_of_nodes,graph_is_connected]],columns=results.columns))

results.reset_index(inplace=True,drop=True)
results.to_csv("results.csv")

# print("Graph similarities: ")
# for graphs in graphs_similarities:
#     res = nx.graph_s(Graphs[graphs[0]][0],Graphs[graphs[1]][0])
#     print("Similarity res: "+str(res)+" for "+graphs[0]+" and "+graphs[1])




print(results)

