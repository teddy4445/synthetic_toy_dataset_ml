# Synthetic Dataset Generator with Genetic Algorithm
This project generates synthetic datasets using specified distributions for features and optimizes additional columns using a genetic algorithm based on a given loss function. Additionally, a target variable y is added to the dataset using a specified function. The goal of the project is to create a versatile and customizable tool for generating and optimizing synthetic data for machine learning experiments.

## Features
Generate synthetic datasets with customizable distributions for each feature.
Optimize additional columns in the dataset using a genetic algorithm.
Define a loss function to guide the optimization process.
Add a target variable y using a specified function.

## Requirements
Python 3.7+
pandas
numpy
DEAP
scipy
Installation

To install the required dependencies, run:
`pip install pandas numpy deap scipy`

## Usage
Import the "algo.py" file into your project and call the function `generate_synthetic_dataset_ml`

## Example
To see how to use the algorithm, run `python example.py`