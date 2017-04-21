import networkx as nx
from math import pow
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


csv_file_name_original = "unhcr_popstats_export_asylum_seekers_monthly_all_data.csv"
csv_file_name_preprocessed = "normalized_refugees_dataset.csv"
read_normalized = True
months_sorted = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
attributes = ['Country / territory of asylum/residence', 'Origin', 'Year', 'Month', 'Value']
years = ((1999, "January"), (2017, "January"))


class MonthData:
    def __init__(self, list_attributes):
        self.host, self.origin, self.year, self.month, self.value = list_attributes
        self.year = int(self.year)

        if self.value.isdigit():
            self.value = int(self.value)
        else:
            self.value = 2

    def __str__(self):
        return "%s,%s,%s,%s,%s" % (self.host, self.origin, self.year, self.month, self.value)


class Refugees:
    def __init__(self):
        if read_normalized:
            self.data = self.read_csv_file(csv_file_name_preprocessed)
        else:
            self.data = self.read_csv_file(csv_file_name_original)
            self.preprocess()

    def preprocess(self):
        self.normalize()

    # Return ([attribute_name1, ...], [[value_1,...], [value_2, ...])
    @staticmethod
    def read_csv_file(file_name):
        file = open(file_name,"r")
        lines = file.readlines()
        result = map(lambda x: x.strip().split(","), lines)
        result = map(lambda x: [x[0], x[1].strip("\"") + ": " + x[2].strip("\""), x[3], x[4], x[5]] if len(x)==6 else x, result)
        data = map(lambda x: MonthData(x), result[1:])
        return data

    # Return { attribute_value_1: [value1, value2, ...], ...}
    def get_values_per_attribute(self, attribute_name, constraint_attribute_names=None, constraint_values=None):
        result = {}
        for object in self.data:
            attribute_value = getattr(object, attribute_name)
            skip = False
            if constraint_attribute_names is not None:
                for i in range(len(constraint_attribute_names)):
                    attr = getattr(object, constraint_attribute_names[i])
                    t = constraint_values[i]
                    y = attr != t
                    if getattr(object, constraint_attribute_names[i]) != constraint_values[i]:
                        skip = True
            if not skip:
                if attribute_value not in result:
                    result[attribute_value] = []
                result[attribute_value].append(object.value)
        return result

    # Return { attribute_value_1: total_value, ...}
    # E.g. refugees.get_aggregated_value("month", constraint_attribute_names=["origin","year"], constraint_values=["Afghanistan", 1999])
    def get_aggregated_value(self, attribute_name, constraint_attribute_names=None, constraint_values=None):
        values = self.get_values_per_attribute(attribute_name, constraint_attribute_names, constraint_values)
        for attribute_value in values.keys():
            values[attribute_value] = sum(values[attribute_value])
        return values


    # Return [("January", value1), ("Februari", value2), ...]
    def get_aggregated_value_per_month(self):
        aggregated_values = self.get_aggregated_value("month")
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
        for index in range(len(self.data)):
            object = self.data[index]
            self.data[index].value = int(round(object.value/normalization_values[object.month]))

    def export_csv(self):
        target = open("normalized_refugees_dataset.csv", 'w')
        list_to_csv = lambda container: reduce(lambda x,y: "%s,%s" % (x,y), container)

        target.write(list_to_csv(attributes) + "\n")
        for i in range(len(self.data)-1):
            target.write(str(self.data[i]) + "\n")
        target.write(str(self.data[-1]))
        target.close()

    def example_curve_fitting(self):
        func = lambda x,a,b,c: a * np.exp(-b * x) + c
        func2 = lambda x,a,b,c: a*np.exp(x,2) + b*x + c

        xdata = np.linspace(0, 4, 50)
        y = func(xdata, 2.5, 1.3, 0.5)
        y_noise = 0.01 * np.random.normal(size=xdata.size)
        ydata = y + y_noise
        plt.plot(xdata, ydata, 'b-', label='data')

        popt, pcov = curve_fit(func, xdata, ydata)
        plt.plot(xdata, func(xdata, *popt), 'r-', label='fit')
        perr = sum(np.sqrt(np.diag(pcov)))
        plt.show()
