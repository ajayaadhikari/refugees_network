import networkx as nx
import temporal_network
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path

fileName = "../dataset/normalized_refugees_dataset.csv"


class PolicyChange:
    def __init__(self):
        self.number_of_months = 3
        self.load_policy_change_graphs()

    def load_policy_change_graphs(self):
        graph_path = '../dataset/policy_graphs_months_%s.pkl' % self.number_of_months

        print("Start reading from file and building temporal network")
        self.temporal_network = temporal_network.get_temporal_network(fileName)
        print("\tDone!!")
        if os.path.isfile(graph_path):
            print("Policy graphs already exists with %s months, retrieving now." % self.number_of_months)
            with open(graph_path, 'rb') as input:
                self.policy_graphs = pickle.load(input)
            print("\tDone!!!")
            print("Removing noise")
            self.remove_small_outflow_all()
            print("\tDone!!!")
        else:
            print("Start building temporal policy graphs using %s months to compute the expected distribution" % self.number_of_months)
            self.policy_graphs = self.get_policy_change_graphs(self.number_of_months)
            print("\tDone!!!")
            print("Removing noise")
            self.remove_small_outflow_all()
            print("\tDone!!!")
            print("Saving graphs to file to avoid re-computation next time.")
            self.save_policy_graphs()
            print("\tDone!!!")

    def save_policy_graphs(self):
        pickle.dump(self.policy_graphs, open('../dataset/policy_graphs_months_%s.pkl' % self.number_of_months, 'wb'), pickle.HIGHEST_PROTOCOL)

    def aggregate(self,DG_temp,DG):
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
        month_number_start = temporal_network.months.index(start[1])
        month_number_end = temporal_network.months.index(end[1])

        if start[0] == end[0]:
            for month in range(month_number_start+1, month_number_end+1):
                m = temporal_network.months[month]
                DG_temp = self.temporal_network[m, start[0]]
                DG = self.aggregate( DG_temp, DG)

        else:
            for year in range(start[0],end[0]+1) :
                if year == start[0]:
                    for month in range(month_number_start+1, 12):
                        m = temporal_network.months[month]
                        DG_temp = self.temporal_network[m, year]
                        DG = self.aggregate(DG_temp, DG)

                elif year == end[0]:
                    for month in range(0, month_number_end+1):
                        m = temporal_network.months[month]
                        DG_temp = self.temporal_network[m, year]
                        DG = self.aggregate(DG_temp, DG)
                else:
                    for month in range(0,12):
                        m = temporal_network.months[month]
                        DG_temp = self.temporal_network[m, year]
                        DG = self.aggregate(DG_temp, DG)

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

            # Normalize by the total sum
            total = float(sum([x[1] for x in distribution]))
            if total != 0:
                distribution = map(lambda x: (x[0], x[1]/total), distribution)

            # Add the normalized weighted according the outflow to the new graph
            for node2 in distribution:
                distribution_network.add_weighted_edges_from([(node, node2[0], node2[1])])
        return distribution_network

    # Output: temporal graphs
    #   Format: { ("January", 2000): graph1, (February, 2000): graph2 ...}
    def get_policy_change_graphs(self, number_of_months):
        N = {}
        for year in range(2000, 2018):
            for month in temporal_network.months:
                # divide the number of months with 12(=1year) in order to get
                # how many years back you have to go for the aggregate network
                years_back = number_of_months // 12
                # the modulo indicates the months that you have to go back in time
                months = number_of_months % 12
                # no data before 1999
                if year - years_back <= 1999:
                    years_back = years_back - (1999 - (year - years_back))
                    months = 0

                month_number = temporal_network.months.index(month)
                # if its negative means that we have to go back plus one year,
                # for example imagine we are in year 2001 - January and the number of months is 14
                # in this case we want to go at 1999 - November
                if month_number - months < 0 :
                    years_back = years_back + 1

                month_start = temporal_network.months[month_number - months]
                month_end = temporal_network.months[month_number - 1]
                # no data after february 2017 so break
                if year == 2017 and month == "February":
                    break
                else:
                    expected_graph = self.get_distribution_outflow(self.get_aggregated_graph(
                                                                                            (year-years_back, month_start),
                                                                                            (year-1 if month == "January" else year, month_end)))
                    real_graph = self.get_distribution_outflow(self.temporal_network[month,year])
                    policy_change_graph = self.policy_change_graph(expected_graph,real_graph)
                    N[(month, year)] = policy_change_graph
            print("\tPolicy graphs done for year " + str(year))

        return N


    @staticmethod
    def policy_change_graph(graph1, graph2):
        graph1 = graph1.copy()
        graph2 = graph2.copy()

        # Input: [(node1,node2),...] Output: [(node1, node2, 0), ..]
        add_zero_weights = lambda container: map(lambda x: (x[0], x[1], 0), container)

        # Add zero weighted edges to the graphs such that both graphs contain the same edges
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())
        diff1 = add_zero_weights(list(edges1 - edges2))
        diff2 = add_zero_weights(list(edges2 - edges1))
        graph1.add_weighted_edges_from(diff2)
        graph2.add_weighted_edges_from(diff1)

        policy_change_graph = nx.DiGraph()
        for edge in graph1.edges():
            weight1 = graph1[edge[0]][edge[1]]["weight"]
            weight2 = graph2[edge[0]][edge[1]]["weight"]
            policy_change_graph.add_weighted_edges_from([(edge[0], edge[1], weight2 - weight1)])

        return policy_change_graph

#############################################################################################################
############################## Analyses of policy graphs ####################################################
#############################################################################################################
# Use the $self.number_of_months attribute to compute the expected distribution in all of the following functions,
# unless indicated otherwise
# Use the $self.policy_graphs attribute

    # Output format: {"Afghanistan": 456, ...}
    @staticmethod
    def get_outflow_per_country(original_graph):
        result = {}
        for node in original_graph.nodes():
            neighbors = original_graph[node]
            weights = map(lambda neighbor: neighbors[neighbor]["weight"], neighbors.keys())
            result[node] = sum(weights)
        return result

    @staticmethod
    def remove_small_outflow(policy_graph, original_graph):
        countries = policy_graph.nodes()
        outflow_per_country = PolicyChange.get_outflow_per_country(original_graph)

        for country in countries:
            if policy_graph.out_degree(country) > policy_graph.in_degree(country):
                if country in outflow_per_country:
                    total_outflow = outflow_per_country[country]
                    if total_outflow < 200:
                        policy_graph.remove_node(country)
                        policy_graph.add_node(country)
                else:
                    policy_graph.remove_node(country)
                    policy_graph.add_node(country)

    # Remove the countries with less than total 200 outgoing refugees per policy graph
    def remove_small_outflow_all(self):
        for time_period in self.policy_graphs.keys():
            self.remove_small_outflow(self.policy_graphs[time_period], self.temporal_network[time_period])

    def write_change_per_pair_to_file(self):
        file = open("../output/per_pair_output/policy_change_%s_months.csv" % self.number_of_months, "w")
        file.write("Destination,Origin,Month,Year,Day/month/year,Change\n")
        time_periods = self.policy_graphs.keys()
        for time_period in time_periods:
                for origin in self.policy_graphs[time_period]:
                    for destination in self.policy_graphs[time_period][origin].keys():
                        if origin != destination:
                            month, year = time_period
                            day_month_year = "1/%s/%s" % (temporal_network.months.index(month) + 1, year)
                            file.write("%s,%s,%s,%s,%s,%s\n" % (destination,
                                                                origin,
                                                                month,
                                                                year,
                                                                day_month_year,
                                                                self.policy_graphs[time_period][origin][destination]["weight"]))
        file.close()

    # write to file the average change and the standard deviation per country
    def write_average_change_and_std_per_country_to_file(self):
        file = open("../output/per_country_output/policy_change_%s_months.csv" % self.number_of_months, "w")
        file.write("Country,Month,Year,Average Policy Change, Standard Deviation\n")
        time_periods = self.policy_graphs.keys()
        for time_period in time_periods:
            for country in self.policy_graphs[time_period]:
                weights = []
                weight = 0.0;

                if self.policy_graphs[time_period].in_degree(country) > self.policy_graphs[time_period].out_degree(country) :
                    predecessors = self.policy_graphs[time_period].predecessors(country)
                    for predecessor in predecessors:
                        weight += self.policy_graphs[time_period][predecessor][country]["weight"]
                        weights.append(self.policy_graphs[time_period][predecessor][country]["weight"])

                    average_policy_change = weight / len(predecessors)
                    file.write("%s,%s,%s,%s,%s\n" % (country, time_period[0], time_period[1], average_policy_change, np.std(weights)))
        file.close()



    def write_change_per_pair_to_file_tableau_type(self):
        file = open("../output/per_pair_output/policy_change_%s_months_tableau_type.csv" % self.number_of_months, "w")
        file.write("country1,country2,Destination-Origin,month,year,Day/month/year,Change\n")
        time_periods = self.policy_graphs.keys()
        for time_period in time_periods:
                for origin in self.policy_graphs[time_period]:
                    for destination in self.policy_graphs[time_period][origin].keys():
                        if origin != destination:
                            month, year = time_period
                            destination_origin = "%s-%s" % (destination,origin)
                            day_month_year = "1/%s/%s" % (temporal_network.months.index(month) + 1, year)
                            file.write("%s,%s,%s,%s,%s,%s,%s\n" % (destination,
                                                                   origin,
                                                                   destination_origin,
                                                                   month,
                                                                   year,
                                                                   day_month_year,
                                                                   self.policy_graphs[time_period][origin][destination]["weight"]))
                            file.write("%s,%s,%s,%s,%s,%s,%s\n" % (origin,
                                                                   destination,
                                                                   destination_origin,
                                                                   month,
                                                                   year,
                                                                   day_month_year,
                                                                   self.policy_graphs[time_period][origin][destination]["weight"]))
        file.close()

    @staticmethod
    #   Example: self.visualize_graph(self.temporal_network[("January",2003)])
    def visualize_graph(graph):
        labels = {}
        [labels.update({x: x[:3].decode('utf-8')}) for x in graph.nodes()]
        nx.draw(graph, arrows=True, labels=labels)  # use spring layout
        plt.show()


pg = PolicyChange()
pg.write_average_change_and_std_per_country_to_file()

