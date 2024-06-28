import os
import sys
import pandas as pd
import numpy as np


class ft_linear_regression():
    """
    Class for linear regression
    """
    def __init__(self, learning_rate: float = 0.001, number_of_iterations: int = 1000):
        self.theta0 = 0
        self.theta1 = 0
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.mileages_mean = None
        self.mileages_std = None
        self.prices_mean = None
        self.prices_std = None

    def calculate_normalization_parameters(self, mileages: np.array, prices: np.array) -> None:
        """
        Calculate the mean and standard deviation of the data
        """
        self.mileages_mean = np.mean(mileages)
        self.mileages_std = np.std(mileages)
        self.prices_mean = np.mean(prices)
        self.prices_std = np.std(prices)

    def normalize(self, values: np.array, mean: float, std: float) -> np.array:
        """
        Normalize the data
        """
        return (values - mean) / std

    def denormalize(self, values: np.array, mean: float, std: float) -> np.array:
        """
        Denormalize the data
        """
        return values * std + mean
    
    def predict(self, mileages: np.array) -> np.array:
        """
        Predict the value of Y
        """
        normalized_predictions = self.theta0 + self.theta1 * mileages
        return self.denormalize(normalized_predictions, self.prices_mean, self.prices_std)
    
    def fit(self, mileages, prices) -> object:
        """
        Fit the model: gradient descent algorithm to find the best theta0 and theta1 values
        """
        m = len(prices)
        for _ in range(self.number_of_iterations):
            predictions = self.predict(mileages)
            tmp_theta0 = self.learning_rate * 1/m * np.sum(predictions - prices)
            tmp_theta1 = self.learning_rate * 1/m * np.sum((predictions - prices) * mileages)
            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1
        self.save()
        return self

    def save(self) ->  None:
        """
        Save the theta values to a file
        """
        df = pd.DataFrame({"theta0": [self.theta0], "theta1": [self.theta1]})
        df.to_csv("theta.csv", index=False)

def main():
    df = pd.read_csv("data.csv")
    mileages = df['km'].values
    prices = df['price'].values

    model = ft_linear_regression()

    # Calculate normalization parameters
    model.calculate_normalization_parameters(mileages, prices)

    # Normalize the data
    mileages_normalized = model.normalize(mileages, model.mileages_mean, model.mileages_std)
    prices_normalized = model.normalize(prices, model.prices_mean, model.prices_std)

    # Fit the model with normalized data
    model.fit(mileages_normalized, prices_normalized)

    # Example usage of predict with denormalization
    predictions = model.predict(mileages_normalized)
    print(predictions)

if __name__ == "__main__":
    main()