import os
import sys
import pandas as pd
import numpy as np
import argparse
from tabulate import tabulate
from utils import ANSIColor

def argument_parser() -> argparse.Namespace:
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser(description="Train the linear regression model")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.01, help="Learning rate for the model")
    parser.add_argument("-i", "--number_of_iterations", type=int, default=1000, help="Number of iterations for the model")
    parser.add_argument("--info", action="store_true", help="Print information about the dataset and exit")
    return parser.parse_args()

class ft_linear_regression():
    """
    Class for linear regression: fit the model and save the theta values to a theta csv file
    """
    def __init__(self, learning_rate: float = 0.01, number_of_iterations: int = 1000):
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
        Calculate the mean and standard deviation of the data for normalization
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
        Predict the value of Y for a given X
        """
        return self.theta0 + self.theta1 * mileages
    
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
        return self

    def save(self) ->  None:
        """
        Save the theta values to a file
        """
        df = pd.DataFrame({"theta0": [self.theta0], "theta1": [self.theta1]})
        df.to_csv("theta.csv", index=False)

def check_dataset(mileages: np.array, prices: np.array, args: argparse.Namespace) -> None:
    """
    Check if the dataset is valid
    """
    if len(mileages) == 0 or len(prices) == 0:
        print(ANSIColor.text("No data found", "red", None), file=sys.stderr)
        sys.exit(1)

    if args.learning_rate <= 0:
        print(ANSIColor.text("Learning rate must be positive", "red", None), file=sys.stderr)
        sys.exit(1)
    elif args.number_of_iterations <= 0:
        print(ANSIColor.text("Number of iterations must be positive", "red", None), file=sys.stderr)
        sys.exit(1)

def apply_args(args: argparse.Namespace, df) -> None:
    """
    Apply the command line arguments
    """
    if args.info:
        print(ANSIColor.text("Dataset information : ", "yellow", None))
        print(ANSIColor.text(f"\t - Shape : ", "green", None) + f"{df.shape}")
        print(ANSIColor.text(f"\t - Description : ", "green", None))
        description = df.describe()
        table_str = tabulate(description, headers='keys', tablefmt='psql')
        tabulated_str = '\n'.join('\t' + line for line in table_str.split('\n'))

        print(tabulated_str)

        print(ANSIColor.text(f"Learning rate : ", "yellow", None) + ANSIColor.text(f"{args.learning_rate}", "green", None))
        print(ANSIColor.text(f"Number of iterations : ", "yellow", None) + ANSIColor.text(f"{args.number_of_iterations}", "green", None))
        sys.exit(0)

def main() -> None:
    """
    Main function
    """
    args = argument_parser()
    if not os.path.exists("data.csv"):
        print(ANSIColor.text("Data file not found", "red", None), file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv("data.csv")
    mileages = df['km'].values
    prices = df['price'].values
    
    check_dataset(mileages, prices, args)
    apply_args(args, df)
    
    model = ft_linear_regression(learning_rate=args.learning_rate, number_of_iterations=args.number_of_iterations)

    model.calculate_normalization_parameters(mileages, prices)

    mileages_normalized = model.normalize(mileages, model.mileages_mean, model.mileages_std)
    prices_normalized = model.normalize(prices, model.prices_mean, model.prices_std)

    model.fit(mileages_normalized, prices_normalized)

    model.theta0 = model.denormalize(model.theta0, model.prices_mean, model.prices_std)
    model.theta1  = model.theta1 * (model.prices_std / model.mileages_std)

    model.save()

if __name__ == "__main__":
    main()