import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Pattern:
    def __init__(self, data):
        self.data = data
        self.model = lambda x,a,b,c: a*pow(x,2) + b*x + c


class Data:
    def __init__(self, year, month):
        self.year = int(year)
        self.month = month
 