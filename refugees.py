import networkx as nx
from math import pow


csv_file_name = "unhcr_popstats_export_asylum_seekers_monthly_all_data.csv"
months_sorted = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
attributes = ['Country / territory of asylum/residence', 'Origin', 'Year', 'Month', 'Value']

class Refugees:
    def __init__(self):
        self.attributes, self.data = self.read_csv_file()
        self.preprocess()

    def preprocess(self):
        self.replace_stars()
        self.normalize()

    # Replace the * in the value attributes to 2
    def replace_stars(self):
        value_index = self.attributes.index("Value")
        for i in range(len(self.data)):
            if self.data[i][value_index].isdigit():
                self.data[i][value_index] = int(self.data[i][value_index])
            else:
                self.data[i][value_index] = 2

    # Return ([attribute_name1, ...], [[value_1,...], [value_2, ...])
    @staticmethod
    def read_csv_file():
        file = open(csv_file_name,"r")
        lines = file.readlines()
        result = map(lambda x: x.strip().split(","), lines)
        result = map(lambda x: [x[0], x[1].strip("\"") + ": " + x[2].strip("\""), x[3], x[4], x[5]] if len(x)==6 else x, result)
        return result[0], result[1:]

    # Return { attribute_value_1: [value1, value2, ...], ...}
    def get_values_per_attribute(self, attribute):
        result = {}
        index_attribute = self.attributes.index(attribute)
        for object in self.data:
            attribute_name = object[index_attribute]
            if attribute_name not in result:
                result[attribute_name] = []
            result[attribute_name].append(object[self.attributes.index("Value")])
        return result

    # Return { attribute_value_1: total_value, ...}
    def get_aggregated_value(self, attribute):
        values = self.get_values_per_attribute(attribute)
        for attribute_value in values.keys():
            values[attribute_value] = sum(values[attribute_value])
        return values

    # Return [("January", value1), ("Februari", value2), ...)]
    def get_aggregated_value_per_month(self):
        aggregated_values = self.get_aggregated_value("Month")
        return [(x, aggregated_values[x]) for x in months_sorted]

    # Return {"January": normalization_value1, ...}
    # This function returns the normalization values to mitigate the seasons bias from the data
    def get_normalization_values(self):
        values_per_month = self.get_aggregated_value_per_month()
        print(values_per_month)
        minimum_value = float(min(values_per_month, key=lambda x: x[1])[1])
        result = {}
        for month,value in values_per_month:
            result[month] = value/minimum_value
        return result

    # This functions mitigates the seasons bias from the data
    def normalize(self):
        normalization_values = self.get_normalization_values()
        value_index = self.attributes.index("Value")
        month_index = self.attributes.index("Month")
        for index in range(len(self.data)):
            object = self.data[index]
            self.data[index][value_index] = int(round(object[value_index]/normalization_values[object[month_index]]))

    def export_csv(self):
        target = open("normalized_refugees_dataset.csv", 'w')
        list_to_csv = lambda container: reduce(lambda x,y: "%s,%s" % (x,y), container)

        target.write(list_to_csv(self.attributes) + "\n")
        for i in range(len(self.data)-1):
            target.write(list_to_csv(self.data[i]) + "\n")
        target.write(list_to_csv(self.data[-1]))
        target.close()


refugees = Refugees()

print(refugees.export_csv())