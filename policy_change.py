import networkx as nx


class PolicyChange:
    def __init__(self):
        pass

    # Input: start, end
    #     Format: start:(year, month), end:(year, month)
    # Output: aggregated graph from $start to $end
    def get_aggregated_graph(self, start, end):
        pass

    # Output: graph
    #   The weights are the percentage of outflow
    def get_distribution_outflow(self):
        pass

    # Output: temporal graphs
    #   Formate: { ("January", 1999): graph, ...}
    def get_policy_change_grpah(self):
        pass

    @staticmethod
    def visualize_graph(graph):
        pass