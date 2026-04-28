"""
Module for visualizing data distributions in machine learning pipelines.

This module provides functions to automatically plot distributions of numeric and categorical features,
and a custom transformer for integrating visualizations into scikit-learn pipelines.
"""

import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.base import BaseEstimator, TransformerMixin  # Import base classes for custom transformers


def plot_all_distributions(df, n_cols=3):
    """
    Universal visualizer: Automatically detects types and 
    plots histograms for numbers and countplots for categories.
    """
    # 1. Automatically separate columns by type
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select columns with numeric data types
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns  # Select columns with categorical data types

    # --- Plot Numerical Distributions ---
    if len(numeric_cols) > 0:  # Check if there are numeric columns to plot
        _grid_plot(df, numeric_cols, n_cols, "Numerical", is_numeric=True)  # Call helper to plot numerical distributions

    # --- Plot Categorical Distributions ---
    if len(categorical_cols) > 0:  # Check if there are categorical columns to plot
        _grid_plot(df, categorical_cols, n_cols, "Categorical", is_numeric=False)  # Call helper to plot categorical distributions


def _grid_plot(df, columns, n_cols, title_prefix, is_numeric=True):
    """Helper function to handle the grid logic (Private)"""
    n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed for the subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))  # Create a figure with subplots
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]  # Flatten the axes array for easy iteration

    for i, col in enumerate(columns):  # Iterate over each column to plot
        if is_numeric:  # If the column is numeric
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='royalblue')  # Plot histogram with kernel density estimate
        else:  # If the column is categorical
            # Countplot for categories
            sns.countplot(data=df, x=col, ax=axes[i], palette='viridis')  # Plot count plot for categorical data
            axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
            
        axes[i].set_title(f'{title_prefix}: {col}')  # Set the title for each subplot

    # Hide extra boxes
    for ax in axes[len(columns):]:  # Iterate over any unused axes
        ax.set_visible(False)  # Hide the unused subplots
    
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()  # Display the plot


class PipelineVisualizer(BaseEstimator, TransformerMixin):
    """A custom pipeline step that draws plots and then passes data forward."""
    def __init__(self, title="Pipeline Step", feature_names=None):
        self.title = title  # Set the title for the visualization step
        self.feature_names = feature_names  # Store the feature names

    def fit(self, X, y=None):
        return self  # Fit method does nothing, just returns self

    def transform(self, X):
        # Convert to DataFrame if it's a numpy array from a previous step
        # 1. Use feature_names if provided, otherwise stick to default 0, 1, 2...
        if self.feature_names is not None:  # Check if feature names are available
            df_to_plot = pd.DataFrame(X, columns=self.feature_names)  # Create DataFrame with specified column names
        else:  # If no feature names
            df_to_plot = pd.DataFrame(X)  # Create DataFrame with default integer column names
        print(f"--- Visualizing: {self.title} ---")  # Print a message indicating the visualization step
        plot_all_distributions(df_to_plot)  # Call the function to plot all distributions
        return X  # Return the input data unchanged