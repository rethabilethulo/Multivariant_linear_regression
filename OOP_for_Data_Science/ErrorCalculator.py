import pandas as import pd
import numpy as np
import matplotlib.pyplot as pyplot
import statsmodels.api as sm 
from statsmodels.tools.eval_measures import mse
from statsmodels.tools.eval_measures import rmse



class ErrorCalculator:

    def __init__(self, y, y_predictor):

# target is y and prediction of target is y_prediction
        self.y          =   np.array(y)       
        self.y_predictor     =   np.array(y_predictor)  

    # check that len of y_prediction is equall to len of y

    def dimension(self):

        if len(self.y.shape) == len(self.y_predictor.shape):
            return True

        else:
            return False



    def get_residuals(self):

        residuals = self.y - self.y_predictor
        return residuals

    def get_standardised_residuals(self):

        standardised_residuals = self.get_residuals() / (self.get_residuals().std())
        return standardised_residuals

    def get_mse(self):

        mse = np.square(np.subtract(self.y , self.predictor)).mean()
        return mse

    def get_rmse(self):

        rmse = np.sqrt(((self.y_prediction - self.y)**2).mean())
        return rmse

    def error_summary(self):

        print("error_summary: ")

        std_resids = self.get_standardised_residuals()

        print(f"The Mean standard residuals = {std_resids.mean()}")
        print(f"The minimum standard residuals = {min(std_resids)}")
        print(f"The maximum standard residuals = {max(std_resids)}")
        print(f"The Mean Square Error = {self.get_mse()}")
        print(f"The Root Square Mean Error = {self.get_rmse()}")