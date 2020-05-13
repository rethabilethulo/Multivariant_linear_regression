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


##Histogram plotter
class HistogramPlotter(Plotter):

    def __init__(self, y_test, y_predictor):
        super().__init__(y, y_predictor)



#Scatterplot plotter
class ScatterPlotter(Plotter):
    
    def __init__(self, y_test, y_predictor):
        super().__init__(y_test, y_predictor)
     
def plot(self):
        df = pd.DataFrame({"y_test":self.y_test, "y_prediction":self.y_prediction})
        plot.scatter(data = df, x = "y_test", y = "y_predictions", color = "indigo")
        plt.title("Model Predictions vs Actual Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Prediction")
        plt.show()