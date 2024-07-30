# library imports
import numpy as np
from scipy.stats import pearsonr

# project imports
from algo import generate_synthetic_dataset_ml


def loss(df):
    """
    Compute the loss based on the average Pearson correlation being as close to 0.5 as possible.

    Parameters:
    df (pd.DataFrame): DataFrame with 'y' column added.

    Returns:
    float: Loss value.
    """
    correlations = []
    for col in df.columns:
        correlation, _ = pearsonr(df[col], df['y'])
        correlations.append(correlation)
    avg_correlation = np.mean(correlations)
    return abs(avg_correlation - 0.5)  # Minimize the absolute difference from 0.5


def y_function(df):
    """
    Add a column 'y' to the DataFrame, which is the sum of the other columns with added normal noise.

    Parameters:
    df (pd.DataFrame): Input DataFrame with X columns.

    Returns:
    pd.DataFrame: Modified DataFrame with an added 'y' column.
    """
    # Sum the other columns
    y = df.sum(axis=1)

    # Add normal noise with mean 0 and std 1
    noise = np.random.normal(loc=0, scale=1, size=df.shape[0])
    df['y'] = y + noise

    return df


def run():
    print(generate_synthetic_dataset_ml(num_samples=100,
                                        distributions=[{"name": "uniform", "params": {"mean": 0, "std": i}} for i in range(5)],
                                        cols_to_add=2,
                                        loss_function=loss,
                                        y_function=y_function))


if __name__ == '__main__':
    run()
