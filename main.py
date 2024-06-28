import os
import sys
import pandas as pd

def estimate_price(mileage, theta0, theta1) -> float:
    """
    Estimate the price of a car given its mileage
    """
    return (theta0 + theta1 * mileage)

def main():
    """
    Main function
    """
    mileage = input("Enter the mileage, km: ")
    try:
        mileage = float(mileage)
        if mileage < 0:
            raise ValueError("Mileage cannot be negative")
    except ValueError as e:
        print(f"Invalid input : {e}", sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv("theta.csv")
    estimated_p = estimate_price(mileage, df["theta0"][0], df["theta1"][0])
    print(f"Estimated price for that mileage is {estimated_p}")

if __name__ == '__main__':
    """
    This block is executed only if the script is run directly
    """
    main()