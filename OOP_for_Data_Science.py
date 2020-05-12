import pandas as import pd
import numpy as np
import matplotlib.pyplot as pyplot
import statsmodels.api as sm 


class ErrorCalculator:

    def __init__(self, y, y_predictor):

        self.y          =   np.array(y)       # target
        self.y_predictor     =   np.array(y_predictor)  # prediction of target

    # check that len of y_pred is equall to len of y

    def dimension(self):

        if len(self.y.shape) == len(self.y_predictor.shape):
            return True

        else:
            raise ValueError(f'shape of y: {self.y} != shape of y_predictor: {self.y_predictor}')



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

        rmse = np.sqrt(((self.y_pred - self.y)**2).mean())
        return rmse

    def error_summary(self):

        print('error_summary: ')

        std_resid = self.get_standardised_residuals()

        print(f'average standard residuals: {std_resid.mean()}')
        print(f'minimum standard residuals: {min(std_resid)}')
        print(f'maximum standard residuals: {max(std_resid)}')
        print(f'MSE: {self.get_mse()}')
        print(f'RSME: {self.get_rmse()}')