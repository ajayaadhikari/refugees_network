import networkx as nx
import temporal_network
import matplotlib.pyplot as plt

fileName = "normalized_refugees_dataset.csv"


class PolicyChange:
    def __init__(self):
        self.temporal_network = temporal_network.get_temporal_network(fileName)
        a = self.get_distribution_outflow(self.temporal_network[("January",2003)])
        print(a["Afghanistan"])

    # Input: start, end
    #     Format: start:(year, month), end:(year, month)
    # Output: aggregated graph from $start to $end
    def get_aggregated_graph(self, start, end):
        pass

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
#print(a["Afganistan"])
