import networkx as nx
import random
import torch
from traces_extraction import *

class Simulation():
    def __init__(self,G,nodes_activity,nodes_hash,trace,Iterations = 1000):
        self.G = G
        self.Trace = trace
        self.Iterations = Iterations
        self.NodesHash = nodes_hash
        self.NodesActivity = nodes_activity

    def run_avg_dist(self):
        if (self.Iterations > self.Trace.shape[0]):
            self.Iterations = self.Trace.shape[0]
       # node_by_dist = self.hash_by_distribution(self.NodesHash ,self.NodesActivity)

        sum_of_hopes = 0
        trace_subsampled = self.Trace.sample(n=self.Iterations)
        for index, row in trace_subsampled.iterrows():
            source_add = self.NodesHash[row.values[DEF_COL_SRC]]
            destination_add = self.NodesHash[row.values[DEF_COL_DST]]
            sum_of_hopes += nx.shortest_path_length(self.G , source=source_add, target=destination_add)
        activity_dist_avg_dist = sum_of_hopes / self.Iterations
        return activity_dist_avg_dist

    def hash_by_distribution(self, nodes_hash, nodes_activity_distribution):
        list_nodes_by_dist = []
        total_nodes = len(nodes_activity_distribution)
        for idx in range(0,len(nodes_activity_distribution)):
            list_nodes_by_dist += [idx]*int(nodes_activity_distribution[idx]*total_nodes)

        return list_nodes_by_dist