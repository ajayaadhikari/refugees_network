import networkx as nx
import temporal_network
import matplotlib.pyplot as plt

fileName = "normalized_refugees_dataset.csv"


class PolicyChange:
    def __init__(self):
        self.temporal_network = temporal_network.get_temporal_network(fileName)

        #self.visualize_graph(self.temporal_network[("January",2003)])



    def aggregate(self,month,start,DG):
        m = temporal_network.months[month]
        DG_temp = self.temporal_network[m, start[0]]
        list_of_edges = DG_temp.edges()
        for edge in list_of_edges:
            if DG.has_edge(edge[0], edge[1]):
                DG[edge[0]][edge[1]]['weight'] += DG_temp[edge[0]][edge[1]]['weight']
            else:
                DG.add_edge(edge[0], edge[1], weight = DG_temp[edge[0]][edge[1]]['weight'])
        return DG

        # Input: start, end
        #     Format: start:(year, month), end:(year, month)
        # Output: aggregated graph from $start to $end
    def get_aggregated_graph(self, start, end):

        DG =  self.temporal_network[start[1],start[0]]
        month_number_start = temporal_network.months.index(start[1]);
        month_number_end = temporal_network.months.index(end[1]);

        if start[0] == end[0]:
            for month in range(month_number_start+1, month_number_end+1):
                DG = self.aggregate( month, start, DG)

        else :
            for year in range(start[0],end[0]+1) :
                if year == start[0]:
                    for month in range(month_number_start+1, 12):
                        DG = self.aggregate(month, start, DG)

                elif year == end[0]:
                    for month in range(1, month_number_end+1):
                        DG = self.aggregate(month, start, DG)
                else:
                    for month in range(1,12):
                        DG = self.aggregate(month, start, DG)

        return DG






    # Output: graph
    #   The weights are the percentage of outflow
    def get_distribution_outflow(self, graph):
        distribution_network = nx.DiGraph()
        nodes = graph.nodes()
        for node in nodes:
            neighbors = graph[node]
            # Get the weights of each outgoing edge from $node, format: [(node1, weight), ..]
            distribution = [(n, neighbors[n]["weight"]) for n in neighbors.keys()]
            total = float(sum([x[1] for x in distribution]))
            # Normalize by the total sum
            distribution = map(lambda x: (x[0], x[1]/total), distribution)

            for node2 in distribution:
                distribution_network.add_weighted_edges_from([(node, node2[0], node2[1])])
        return distribution_network


    # Output: temporal graphs
    #   Formate: { ("January", 1999): graph, ...}
    def get_policy_change_grpah(self):
        pass

    @staticmethod
    #   Example: self.visualize_graph(self.temporal_network[("January",2003)])
    def visualize_graph(graph):
        labels = {}
        [labels.update({x: x[:3].decode('utf-8')}) for x in graph.nodes()]
        nx.draw(graph, arrows=True, labels=labels)  # use spring layout
        plt.show()

a = PolicyChange()
