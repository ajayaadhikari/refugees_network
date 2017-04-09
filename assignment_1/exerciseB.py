import networkx as nx
from math import pow
from exerciseA import getAllNodes,graphA
fileName = "/home/raj/Dropbox/sem2/complexNetworks/data.csv"

getMean = lambda container: reduce(lambda i,j: i+j, container)/float(len(container))

def getVariance(container):
    mean = getMean(container)
    return getMean(map(lambda x: pow(float(x)-mean, 2), container))

def graphB(fileName):
    file = open(fileName,"r")
    lines = file.readlines()
    graphPerTime = {}
    i = 1
    for line in lines[1:]:
        node1,node2,time = line.strip().split(",")
        if time not in graphPerTime:
            graphPerTime[time]=nx.Graph()
            i += 1
        graphPerTime[time].add_edge(int(node1), int(node2))

    return graphPerTime


def averageInfectionPerTime():
    graphPerTime = graphB(fileName)
    allNodes = getAllNodes(fileName)
    numInfectedNodesPerTime = {}
    timeKeys = sorted(graphPerTime.keys())
    # Initialize the lists per time
    # time -> {number of infected nodes with seed1, seed2, ...}
    for time in timeKeys:
        numInfectedNodesPerTime[time] = []
    #Take each note as seed
    for seed in allNodes:
        # Set the seed as the initial infected node
        infectedNodes = set([seed])
        # Per time step compute the number of infected nodes
        for time in timeKeys:
            currentInfectedNodes = list(infectedNodes)
            graph = graphPerTime[time]
            # Per infected node make its neighbours infected
            for infectedNode in currentInfectedNodes:
                if graph.has_node(infectedNode):
                    infectedNodes = infectedNodes.union(graph.neighbors(infectedNode))
            # Add the number of infected node for the current time step to the list of the seed
            numInfectedNodesPerTime[time].append(len(infectedNodes))
        #print(numInfectedNodesPerTime)
    mean = map(lambda x: getMean(numInfectedNodesPerTime[x]), sorted(numInfectedNodesPerTime.keys()))
    variance = map(lambda x: getVariance(numInfectedNodesPerTime[x], ), sorted(numInfectedNodesPerTime.keys()))
    print(mean)
    print(variance)
    return numInfectedNodesPerTime

def influenceOfNodes():
    # This percentage is used to get the number of timestep per seed when @percentage nodes are infected
    percentage = 0.8
    graphPerTime = graphB(fileName)
    allNodes = getAllNodes(fileName)
    influencePerSeed = {}
    timeKeys = sorted(graphPerTime.keys())

    #Take each note as seed
    for seed in allNodes:
        # Set the seed as the initial infected node
        infectedNodes = set([seed])
        # Per time step compute the number of infected nodes
        for time in timeKeys:
            currentInfectedNodes = list(infectedNodes)
            graph = graphPerTime[time]
            # Per infected node make its neighbours infected
            for infectedNode in currentInfectedNodes:
                if graph.has_node(infectedNode):
                    infectedNodes = infectedNodes.union(graph.neighbors(infectedNode))
            # Add the number of infected node for the current time step to the list of the seed
            if len(infectedNodes)/float(len(allNodes)) >= percentage:
                influencePerSeed[seed] = int(time)
                break
    print(sorted(influencePerSeed.items(), key=lambda x: x[1], reverse=True))
    return influencePerSeed    

graphA = graphA(fileName)

def getRankDegree():
    return map(lambda y: y[0], sorted(graphA.degree(graphA.nodes()).items(), key=lambda x: x[1], reverse=True))

def getRankClustering():
    return map(lambda y: y[0], sorted(nx.clustering(graphA).items(), key=lambda x: x[1], reverse=True))

def getRankInfluence():
    # Computed in exerciseB: influenceOfNodes
    return [182, 154, 94, 49, 38, 34, 241, 236, 232, 181, 179, 177, 176, 175, 174, 173, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 158, 157, 156, 155, 153, 147, 126, 116, 115, 114, 113, 112, 111, 109, 108, 107, 106, 105, 104, 103, 102, 100, 99, 98, 97, 96, 95, 233, 228, 227, 224, 223, 221, 219, 217, 216, 213, 212, 211, 210, 208, 207, 205, 203, 202, 201, 198, 196, 195, 193, 190, 189, 184, 183, 180, 178, 172, 171, 159, 152, 150, 149, 148, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 129, 128, 127, 125, 124, 123, 122, 121, 120, 119, 118, 117, 110, 101, 93, 87, 82, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 48, 47, 46, 45, 44, 42, 41, 40, 39, 37, 36, 35, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 226, 225, 222, 220, 218, 215, 214, 209, 204, 200, 199, 197, 194, 192, 191, 188, 187, 186, 185, 86, 85, 74, 43, 242, 240, 239, 238, 237, 235, 231, 230, 229, 206, 92, 91, 89, 88, 84, 83, 81, 80, 79, 78, 77, 76, 75, 130, 90, 146, 151, 234]

def influenceRelation(secondRank):
    influence = getRankInfluence()
    numOfNodes = len(secondRank)

    result = []
    for i in range(1,11):
        percentage = i * 0.05;
        lastIndex = int(round(percentage*numOfNodes))
        measure = len(set(secondRank[:lastIndex]) & set(influence[:lastIndex]))/float(lastIndex+1)
        result.append(measure)
    return result


print(influenceRelation(getRankDegree()))
print(influenceRelation(getRankClustering()))