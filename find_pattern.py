import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Pattern:
    def __init__(self, data):
        self.data = data

    def filter_data_country_origin(self, country):
        pass


class Data:
    def __init__(self, year, month):
        self.year = int(year)
        self.month = month