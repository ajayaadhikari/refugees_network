import networkx as nx
from math import pow

fileName = "/home/raj/Dropbox/sem2/complexNetworks/data.csv"

def graphA(fileName):
    file = open(fileName,"r")
    lines = file.readlines()
    G=nx.Graph()
    for line in lines[1:]:
        node1,node2,time = line.strip().split(",")
        G.add_edge(int(node1), int(node2))
    return G

graph = graphA(fileName)

def getAllNodes(fileName):
    return graphA(fileName).nodes()


def getStatsA(graph):
    getMean = lambda container: reduce(lambda i,j: i+j, container)/float(len(container))
    getVariance = lambda container,mean: getMean(map(lambda x: pow(float(x)-mean, 2), container))
    degreeNodes = graph.degree(graph.nodes()).values()

    numOfNodes = len(graph.nodes())
    numOfEdges = len(graph.edges())
    graphDensity = 2*numOfEdges/float(numOfNodes*(numOfNodes-1))
    averageDegree = getMean(degreeNodes)
    varianceDegree = getVariance(degreeNodes, averageDegree)


    print("Exercise 1:")
    print("Number of nodes: ", str(numOfNodes))
    print("Number of edges: ", str(numOfEdges))
    print("Link density: ", str(graphDensity))
    print("Average degree: ", str(averageDegree))
    print("Variance degree: ", str(varianceDegree))

    print("Exercise 2:")

    assortativity = nx.degree_assortativity_coefficient(graph)
    print("Exercise 3:")
    print("Assortativity: ", str(assortativity), "Positive value indications that there is a correlation between nodes of similar degree," \
                                                    + " while negative values indicate that there is a correlation between nodes of different degree.")

    clusteringCoefficient = nx.average_clustering(graph)    
    print("Exercise 4:")
    print("Average clustering coefficient: ", str(clusteringCoefficient))

    averageHopCount = nx.average_shortest_path_length(graph)
    diameter = nx.diameter(graph)
    print("Exercise 5:")
    print("Average hop count: ", str(averageHopCount))
    print("Diameter:", str(diameter))

    print("Exercise 6:")

    adjacencySpectrum = sorted(nx.adjacency_spectrum(graph))
    print("Exercise 7:")
    print("Spectral radius (largest eigenvalue of the adjacency matrix):", str(adjacencySpectrum[-1]))

    laplacianSpectrum = sorted(nx.laplacian_spectrum(graph))
    print("Exercise 8:")
    print("Algebraic connectivity (second largest eigenvalue of the laplacian matrix):", str(laplacianSpectrum[-2]))
