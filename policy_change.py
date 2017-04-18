import networkx as nx
import temporal_network
import matplotlib.pyplot as plt

fileName = "normalized_refugees_dataset.csv"


class PolicyChange:
    def __init__(self):
        self.temporal_network = temporal_network.get_temporal_network(fileName)

        self.visualize_graph(self.temporal_network[("January",2003)])

    # Input: start, end
    #     Format: start:(year, month), end:(year, month)
    # Output: aggregated graph from $start to $end
    def get_aggregated_graph(self, start, end):
        pass

    # Output: graph
    #   The weights are the percentage of outflow
    def get_distribution_outflow(self, graph):
        pass

    # Output: temporal graphs
    #   Formate: { ("January", 1999): graph, ...}
    def get_policy_change_grpah(self):
        pass

    @staticmethod
    def visualize_graph(graph):
        labels = {}
        [labels.update({x: x[:3].decode('utf-8')}) for x in graph.nodes()]
        nx.draw(graph, arrows=True, labels=labels)  # use spring layout
        plt.show()

a = PolicyChange()
