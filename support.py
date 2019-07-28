
import networkx as nx

import numpy as np
from collections import defaultdict
import copy
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
    return np.diag(np.sum(AdjMat,axis=0))#np.diag(np.diag(AdjMat))#torch.sum(AdjMat,dim=0).diag()


def insert_row(idx, df, df_insert):
    return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop = True)

def _is_converge(s1, s2, eps=1e-4):
  for i in s1.keys():
    for j in s1[i].keys():
      if abs(s1[i][j] - s2[i][j]) >= eps:
        return False
  return True

def simrank(G, r=0.9, max_iter=100):
  # init. vars
  sim_old = defaultdict(list)
  sim = defaultdict(list)
  for n in G.nodes():
    sim[n] = defaultdict(int)
    sim[n][n] = 1
    sim_old[n] = defaultdict(int)
    sim_old[n][n] = 0

  # recursively calculate simrank
  for iter_ctr in range(max_iter):
    if _is_converge(sim, sim_old):
      break
    sim_old = copy.deepcopy(sim)
    for u in G.nodes():
      for v in G.nodes():
        if u == v:
          continue
        s_uv = 0.0
        for n_u in G.neighbors(u):
          for n_v in G.neighbors(v):
            s_uv += sim_old[n_u][n_v]
        sim[u][v] = (r * s_uv / (len(G.neighbors(u)) * len(G.neighbors(v))))
  return sim

