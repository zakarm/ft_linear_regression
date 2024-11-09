import os
import sys
import pandas as pd
from utils import ANSIColor


def read_theta() -> tuple:
    """
    Read the theta values from a file
    """
    if not os.path.exists("theta.csv") or os.path.getsize("theta.csv") == 0:
        return (0, 0)
    
    df = pd.read_csv("theta.csv")
    if df.empty or "theta0" not in df.columns or "theta1" not in df.columns or len(df) == 0:
        return (0, 0)
    
    return (df["theta0"][0], df["theta1"][0])

def estimate_price(mileage: float) -> float:
    """
    Estimate the price of a car given its mileage
    """
    theta0, theta1 = read_theta()
    return (theta0 + theta1 * mileage)

def main() -> None:
    """
    Main function
    """
    mileage = input(ANSIColor.text("Enter the mileage, km: ", "yellow", None))
    try:
        mileage = float(mileage)
        if mileage < 0:
            raise ValueError("Mileage cannot be negative")
    except ValueError as e:
        print(ANSIColor.text(f"Invalid input : {e}", "red", None), file = sys.stderr)
        sys.exit(1)
    estimated_p = estimate_price(mileage)
    print(ANSIColor.text(f"Estimated price for that mileage is {estimated_p}", "green", None))

if __name__ == '__main__':
    """
    This block is executed only if the script is run directly
    """
    main()