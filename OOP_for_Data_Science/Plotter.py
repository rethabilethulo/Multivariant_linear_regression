import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    
    def __init__(self, y_test, y_predictor):
        self.y_test = y_test
        self.y_predictor = y_predictor

    def test_calculations(self):
        return self.y_test - self.y_predictor

    def plot(self):
        plt.hist(self.test_calculations())
        plt.title("Model Prediction Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()
