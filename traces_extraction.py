
import pandas as ps
from enum import Enum
import matplotlib.pyplot as plot
from collections import Counter
import numpy as np
import csv
from threading import Thread
import matplotlib.cm as cm
from scipy.sparse import csr_matrix
import torch
import pytorch_bridge

DEF_COL_TIME = 0
DEF_COL_SRC = 1
DEF_COL_DST = 2

NODES_ACTIVITY_ATTR = "nodes activity"
PAIRS_ACTIVITY_ATTR = "pairs activity"

class trace:

    def __init__(self,experimentName,traceFileName):
        self.trace = ps.read_csv(traceFileName,nrows=10000)#TODO remove nrows=
        self.experimentName = experimentName
        self.expStrBlock = "["+self.experimentName+"] "
        self.convert_hash = dict()

    def extract_statistics(self):
        statistics = dict()
        TimeColumn = self.trace.iloc[:,DEF_COL_TIME]
        SourcesColumn = self.trace.iloc[:,DEF_COL_SRC]
        DestinationsColumn = self.trace.iloc[:,DEF_COL_DST]

        print(self.expStrBlock+"Converting strings to integer representation")
        UniqueAddressesInt,SourcesColumn,DestinationsColumn,UnifiedTxRxNodes,self.convert_hash = self.convert_string_column_to_indexes(SourcesColumn,DestinationsColumn)

        print("# of unique addresses: "+str(len(UniqueAddressesInt)))
        print(self.expStrBlock+"Calculating unique src/dst")

        statistics["# of unique sources"] = len(ps.unique(SourcesColumn))
        statistics["# of unique destinations"] = len(ps.unique(DestinationsColumn))
        statistics["# of unique addresses"] = len(UniqueAddressesInt)

        listOfPairs = self.two_columns_to_list_of_pairs(SourcesColumn,DestinationsColumn)
        self.listOfPairs = listOfPairs
        print("# num of pairs "+str(len(listOfPairs)))
        print(self.expStrBlock+"Calculating unique requests")

        statistics["# of unique requests"] = len(ps.unique(listOfPairs))

        #statistics[NODES_ACTIVITY_ATTR],statistics[PAIRS_ACTIVITY_ATTR] = self.generateHistograms(UnifiedTxRxNodes,listOfPairs)
        #statistics[NODES_ACTIVITY_ATTR] = self.generate_nodes_activity(listOfPairs)

     #   self.nodes_activity_df = ps.DataFrame([statistics[NODES_ACTIVITY_ATTR][0],statistics[NODES_ACTIVITY_ATTR][1]])
     #   self.nodes_activity_df = (-1*(self.nodes_activity_df.T)).sort_values(by=[1])*(-1)

        self.JointlyDistMat_Calc(UniqueAddressesInt,listOfPairs)


        self.statistics = statistics #save the dictionary
        self.SimilarityActivityMat = self.generate_similarity_matrix_by_activity(UniqueAddressesInt,listOfPairs)

        statistics[NODES_ACTIVITY_ATTR] = np.sum(self.SimilarityActivityMat,axis=0)#torch.sum(self.SimilarityActivityMat,dim=0) #sum rows to get activity per node

        print(self.expStrBlock+"Analayze completed")

    def get_convert_hash(self):
        return self.convert_hash

    def get_trace_table(self):
        return self.trace

    def get_trace_data(self):
        return self.statistics

    def get_nodes_activity(self):
        return self.statistics[NODES_ACTIVITY_ATTR]

    def get_list_of_pairs_nodes(self):
        return self.listOfPairs

    def get_trace_name(self):
        return self.experimentName

    def convert_string_column_to_indexes(self,SourcesColumnStr,DestinationsColumnStr):
        UnifiedList = ps.concat([SourcesColumnStr, DestinationsColumnStr])
        print(self.expStrBlock + "Generating array of uniques")
        UniqueAddresses = ps.unique(UnifiedList)
        UniqueAddressesInt = []

        print(self.expStrBlock + "Generating hash table")
        convert_hash = dict()
        for idx,address in enumerate(UniqueAddresses):
            convert_hash[address] = idx
            UniqueAddressesInt.append(idx)

        print(self.expStrBlock + "Generating new columns")
        SourcesIntCol = np.zeros(shape=SourcesColumnStr.shape)
        DestinationsIntCol =  np.zeros(shape=DestinationsColumnStr.shape)

        for idx in range(SourcesColumnStr.shape[0]):
            SourcesIntCol[idx] = convert_hash[SourcesColumnStr[idx]]
            DestinationsIntCol[idx] = convert_hash[DestinationsColumnStr[idx]]


        UnifiedListInt = np.concatenate((SourcesIntCol,DestinationsIntCol))

        return UniqueAddressesInt,SourcesIntCol,DestinationsIntCol,UnifiedListInt,convert_hash

    def generateHistograms(self,UnifiedTxRxNodes,listOfPairs):
        print(self.expStrBlock+"Generating nodes activity histogram")
        nodesActivity = self.generate_activity_histogram(UnifiedTxRxNodes, 'Nodes Activity')
        print(self.expStrBlock+"Generating edges activity histogram")
        pairsActivity = self.generate_activity_histogram(listOfPairs, 'Pairs (Edges) Activity')
        return nodesActivity,pairsActivity


    def JointlyDistMat_Calc(self,UniqueAddresses,listOfPairs):
        print(self.expStrBlock+"Generating Jointly Distribution Matrix")
        row = []
        col = []
        data = []
        #labels, values = zip(*Counter(listOfPairs).items())
        labels, values = self.times_appeared_in_list(listOfPairs)

        for idx,pair in enumerate(labels):
            row.append(pair[0])
            col.append(pair[1])
            data.append(values[idx])

        self.JointlyDistMat = csr_matrix((data,col,row),dtype=int)

    def generate_similarity_matrix_by_activity(self,UniqueAddresses,listOfPairs):
        shape = [len(UniqueAddresses)]*2
        S = np.zeros(shape,dtype=np.float)
        num_of_pairs = len(listOfPairs)
        labels, values = self.times_appeared_in_list(listOfPairs)
        for idx, pair in enumerate(labels):
            S[int(pair[0])][int(pair[1])] += (values[idx]/(2*num_of_pairs))
            S[int(pair[1])][int(pair[0])] += (values[idx]/(2*num_of_pairs))
        return S

    def get_similarity_matrix_by_activity(self):
        return self.SimilarityActivityMat

    def two_columns_to_list_of_pairs(self,ColA,ColB):
        Res = []
        for idx,val in enumerate(ColA):
            Res.append((val,ColB[idx]))
        return Res

    def generate_nodes_activity(self,givenList):
        labels, values = self.times_appeared_in_list(givenList)
        return (labels,values)

    def generate_activity_histogram(self,givenList,PlotName):
        #plot.hist(x=,bins='auto',color='#0504aa',alpha=0.7,rwidth=0.85)
        #labels,values = zip(*Counter(givenList).items())
        labels, values = self.times_appeared_in_list(givenList)
        #indexes = range(0,len(labels))
        #valuesDist = [x/sum(values) for x in values]
        # plot.figure()
        # plot.ylabel('Probability')
        # plot.title(PlotName+" Distribution")
        # plot.bar(indexes, valuesDist, alpha=0.75, color="skyblue")
        # fname = self.experimentName+"_bar_"+PlotName+".png"
        # plot.savefig(fname)
        plot.figure()
        plot.ylabel('Occurances')
        plot.title(PlotName+" Number Of Transmits")
        plot.hist(x=values, bins='auto', alpha=0.75, color="skyblue")
        plot.yscale("log")
        fname = self.experimentName + "_hist_" + PlotName + ".png"
        plot.savefig(fname)
        return (labels,values)

    def times_appeared_in_list(self,givenList):
        counts_dict = dict()
        for elem in givenList:
            if elem in counts_dict:
                counts_dict[elem] += 1
            else:
                counts_dict[elem] = 1
        return list(counts_dict.keys()), list(counts_dict.values())

    def print_to_file(self):
        print(self.expStrBlock+"Saving results to files")
        #plot.figure()
        #plot.imshow(self.JointlyDistMat,interpolation='bilinear')
        #plot.savefig(self.experimentName+"_JointlyDistMatrix_HeatMap.png")
        #np.savetxt(self.experimentName+"_JointlyDistMatrix.csv",self.JointlyDistMat,delimiter=",")
        #saving the statitstics
        with open(self.experimentName+'_statistics.csv', 'w') as f:
            for key in self.statistics.keys():
                f.write("%s,%s\n" % (key, self.statistics[key]))
        self.save_sparseMatcsv(self.experimentName+"_JointlyDistMatrix.csv",self.JointlyDistMat)
        print(self.expStrBlock+"File were saved")

    def save_sparseMatcsv(self,filename,SparseMat):
        # with open(filename, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['x', 'y', 'value'])
        #     MatCoo = SparseMat.tocoo()
        #     for idx,data in enumerate(MatCoo.data):
        #         writer.writerow([MatCoo.row[idx],MatCoo.col[idx],data])
        print(self.expStrBlock+"Saving sparse matrix")
        MatCoo = SparseMat.tocoo()
        ColStack = np.column_stack((MatCoo.row,MatCoo.col,MatCoo.data))
        dframe = ps.DataFrame(ColStack,columns=['row','col','val'],index=range(len(ColStack)))
        dframe.to_csv(filename)
        #np.savetxt(filename,ColStack,delimiter=",")
        print(self.expStrBlock+"Sparse matrix was saved successfuly")
