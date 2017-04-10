import networkx as nx
from refugees import MonthData

fileName = "normalized_refugees_dataset.csv"
month = "January"

def graphA(fileName,month):
    file = open(fileName,"r")
    lines = file.readlines()
    DG = nx.DiGraph()
    for line in lines:
        result = line.strip().split(",")
        if month == result[3]:
           node1 = result[1]
           node2 = result[0]
           weight = result[4]
           DG.add_weighted_edges_from([(node1,node2,int(weight))])
    return DG




DG = graphA(fileName,month)
print(DG.out_degree("Afghanistan",weight='weight'))