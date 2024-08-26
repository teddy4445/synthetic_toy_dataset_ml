# library imports
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

# project imports
from synthetic_data_generation import generate_synthetic_dataset_ml


def loss(df):
    """
    Compute the loss based on the average Pearson correlation being as close to 'target_corr' as possible.

    Parameters:
    df (pd.DataFrame): DataFrame with 'y' column added.

    Returns:
    float: Loss value.
    """
    target_corr = 0.9
    correlations = []
    for col in df.columns:
        if col == 'y':
            continue
        correlation, _ = pearsonr(df[col], df['y'])
        correlations.append(correlation)
    avg_correlation = np.mean(correlations)
    return abs(avg_correlation - target_corr)  # Minimize the absolute difference from 0.5


def y_function(df, threshold=None):
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

    # Determine the threshold if not provided
    if threshold is None:
        threshold = df['y'].median()

    # Convert to binary class labels based on the threshold
    df['y'] = (df['y'] > threshold).astype(int)

    return df

def pearson_correlation_matrix(df):
    """
    Compute the Pearson correlation coefficient matrix for all columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with numerical columns.
    
    Returns:
    pd.DataFrame: A DataFrame containing the Pearson correlation coefficient matrix.
    """
    # Ensure that we only include numerical columns
    numerical_df = df.select_dtypes(include=['number'])
    
    # Compute the Pearson correlation coefficient matrix
    correlation_matrix = numerical_df.corr(method='pearson')
    
    return correlation_matrix

def create_label(df, alpha=0.1):
    """
        Creates binary label column 'y' based on K-means clustering
        with K=2 and swaps some labels with a given probability alpha.

        Parameters:
        - df (pd.DataFrame): Dataframe containing the n-dimensional Pearson vectors
        - alpha (float): Probability of swapping the cluster label to create noise.
                         Default is 0.1

        Returns:
        - pd.DataFrame: Dataframe with an added 'y' column containing the labels
    """
    # Check if alpha is a probability
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    # Standard Scaling, otherwise clustering is biased
    scaler = StandardScaler().set_output(transform="pandas")
    df_scaled = scaler.fit_transform(df)

    # Apply K-means clustering with K=2
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_scaled['y'] = kmeans.fit_predict(df_scaled)

    # Introduce noise by swapping labels with probability alpha
    swap_mask = np.random.rand(len(df_scaled)) < alpha
    df_scaled.loc[swap_mask, 'y'] = 1 - df_scaled.loc[swap_mask, 'y']

    return df_scaled


def run(num_samples):
    csv_path=f"path_to_where_your_csv_file_should_be_saved_to"
    distributions = [
        {"name": "normal", "params": {"mean": 0, "std": 1}},
        {"name": "normal", "params": {"mean": 0, "std": 2}},
        #{"name": "normal", "params": {"mean": 0, "std": 3}},
        #{"name": "normal", "params": {"mean": 0, "std": 4}},
        {"name": "uniform", "params": {"low": 0, "high": 1}},
        {"name": "uniform", "params": {"low": 0.5, "high": 1.5}}#,
        #{"name": "uniform", "params": {"low": 1, "high": 2}},
        #{"name": "uniform", "params": {"low": 1.5, "high": 2.5}}
    ]
    data = generate_synthetic_dataset_ml(num_samples=num_samples,
                                        distributions=distributions,
                                        cols_to_add=3,
                                        loss_function=loss,
                                        y_function=create_label)
    print(f'Pearson correlation matrix: {pearson_correlation_matrix(data)}')

    # Use the create label function to give the data a more difficult label
    data.to_csv(csv_path,index=False)


if __name__ == '__main__':
    num_samples = 100
    start_time = time.time()
    run(num_samples)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to simulate {num_samples} samples of Pearson vectors: {elapsed_time / 3600} hours.")
