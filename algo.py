import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms


def generate_dataframe(num_samples, distributions):
    """
    Generate a DataFrame with specified distributions for each column.

    Parameters:
    num_samples (int): Number of samples to generate.
    distributions (list of dict): List of distributions for each column.
                                  Each distribution is a dict with keys 'name' and 'params'.

    Returns:
    pd.DataFrame: Generated DataFrame.
    """
    data = {}
    for i, dist in enumerate(distributions):
        if dist['name'] == 'normal':
            data[f'X{i + 1}'] = np.random.normal(loc=dist['params']['mean'],
                                                 scale=dist['params']['std'],
                                                 size=num_samples)
        elif dist['name'] == 'uniform':
            data[f'X{i + 1}'] = np.random.uniform(low=dist['params']['low'],
                                                  high=dist['params']['high'],
                                                  size=num_samples)
        # Add more distributions as needed
        # elif dist['name'] == 'another_distribution':
        #     data[f'X{i+1}'] = ...

    return pd.DataFrame(data)


def optimize_columns_using_genetic_algorithm(df, cols_to_add, loss_function, num_generations=50, population_size=100):
    """
    Use a genetic algorithm to optimize and add new columns to minimize the loss function.

    Parameters:
    df (pd.DataFrame): Input DataFrame with existing columns.
    cols_to_add (int): Number of new columns to add.
    loss_function (function): A function that takes a DataFrame and returns a positive number.
    num_generations (int): Number of generations for the genetic algorithm.
    population_size (int): Size of the population for the genetic algorithm.

    Returns:
    pd.DataFrame: DataFrame with optimized new columns.
    """
    # Define the individual and fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    num_samples = df.shape[0]

    # Attribute generator for each new column
    def generate_column():
        return np.random.random(num_samples)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, generate_column, n=cols_to_add)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        df_copy = df.copy()
        for i, col in enumerate(individual):
            df_copy[f'new_col_{i + 1}'] = col
        return loss_function(df_copy),

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Generate initial population
    population = toolbox.population(n=population_size)

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations,
                        stats=None, halloffame=None, verbose=False)

    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]

    # Add the optimized new columns to the DataFrame
    for i, col in enumerate(best_individual):
        df[f'new_col_{i + 1}'] = col
    return df


def generate_synthetic_dataset_ml(num_samples, distributions, cols_to_add, loss_function, y_function):
    """
    Generate a synthetic dataset, compute a positive number using a loss function,
    and add a 'y' column using a y_function.

    Parameters:
    num_samples (int): Number of samples to generate.
    distributions (list of dict): List of distributions for each column.
                                  Each distribution is a dict with keys 'name' and 'params'.
    cols_to_add (int): Number of new columns to add.
    loss_function (function): A function that takes a DataFrame and returns a positive number.
    y_function (function): A function that takes a DataFrame and adds a 'y' column to it, returning the modified DataFrame.

    Returns:
    pd.DataFrame: Modified DataFrame with added 'y' column.
    float: A positive number computed using the loss function.
    """
    # Generate the initial DataFrame
    df = generate_dataframe(num_samples, distributions)

    # Optimize and add new columns using the genetic algorithm
    df = optimize_columns_using_genetic_algorithm(df, cols_to_add, loss_function)

    # Add the 'y' column using the y_function
    df_with_y = y_function(df)

    return df_with_y

