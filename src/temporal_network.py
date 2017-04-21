import networkx as nx

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def get_temporal_network(fileName):
    N = {}
    file = open(fileName,"r")
    lines = file.readlines()
    lines.pop(0)
    for year in range(1999, 2018):
        for month in months:
            if year == 2017 and month == "February" :
                break
            DG = nx.DiGraph()

            N[(month,year)] = DG

    for line in lines:
        result = line.strip().split(",")
        node1 = result[1].strip("\"") + " " + result[2].strip("\"") if len(result) == 6 else result[1]
        node2 = result[0]
        weight = result[4]
        N[(result[3],int(result[2]))].add_weighted_edges_from([(node1, node2, int(weight))])
    return N

#N = get_temporal_network("normalized_refugees_dataset.csv")
#print(N[("January", 2000)].in_degree("Greece",weight='weight'))