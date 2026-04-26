"""Plotting utilities for Titanic dataset visualizations.

This module contains helper functions for visualizing numerical data
from pandas DataFrames, such as histograms with kernel density estimate
(KDE) overlays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_numerical_distributions(df, n_cols=3):
    """Plot histograms with KDE curves for all numerical dataframe columns.

    This function creates a grid of subplots, each containing a histogram with
    an overlaid kernel density estimate (KDE) curve for numerical columns. It
    automatically handles layout calculations and subplot arrangement.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns to visualize.
    n_cols : int, optional
        Number of subplot columns in the grid, by default 3.

    Returns
    -------
    None
        Displays the plot using matplotlib's pyplot.show().

    Examples
    --------
    >>> plot_numerical_distributions(df, n_cols=2)
    """

    # Select only numeric columns from the dataframe using numpy's number type.
    numeric_cols = df.select_dtypes(include=np.number).columns

    # If no numerical columns are present, print a warning and exit the function early.
    if len(numeric_cols) == 0:
        print("No numerical columns found in the dataframe.")
        return

    # Calculate the number of rows required for the subplot grid by rounding up the division.
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    # Create a figure object and an axes grid with the calculated number of rows and columns.
    # figsize sets the overall figure size: width = n_cols * 5 inches, height = n_rows * 4 inches.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    # Flatten the 2D axes array into a 1D list for easier iteration when multiple subplots exist.
    # If only one subplot is created, wrap the single axis object in a list for consistency.
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Iterate through each numeric column with its index and column name.
    for i, col in enumerate(numeric_cols):
        # Create a histogram with KDE overlay for the current column, removing NaN values.
        # kde=True overlays a kernel density estimate curve on the histogram.
        # bins=30 sets the number of histogram bars, color='royalblue' sets the bar color.
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], bins=30, color='royalblue')
        # Set the title of the subplot to display the column name with fontsize 14.
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        # Remove the x-axis label for a cleaner appearance (empty string removes default label).
        axes[i].set_xlabel('')
        # Set the y-axis label to 'Frequency' to indicate what the bars represent.
        axes[i].set_ylabel('Frequency')

    # Hide any unused axes in the grid (when total subplots > numeric columns).
    # This prevents empty subplot frames from appearing in the figure.
    for ax in axes[len(numeric_cols):]:
        ax.set_visible(False)

    # Adjust the layout to prevent titles and labels from overlapping with other elements.
    fig.tight_layout()
    # Display the complete figure in the output window or notebook.
    plt.show()


def plot_categorical_distributions(df, n_cols=3):
    """Plot bar charts for all categorical dataframe columns.

    This function creates a grid of subplots, each containing a bar chart
    for categorical columns. It automatically handles layout calculations
    and subplot arrangement.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns to visualize.
    n_cols : int, optional
        Number of subplot columns in the grid, by default 3.

    Returns
    -------
    None
        Displays the plot using matplotlib's pyplot.show().

    Examples
    --------
    >>> plot_categorical_distributions(df, n_cols=2)
    """

    # Select only categorical columns from the dataframe using pandas' select_dtypes method with 'object' and 'category' types.
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # If no categorical columns are present, print a warning message and exit the function early.
    if len(categorical_cols) == 0:
        print("No categorical columns found in the dataframe.")
        return

    # Calculate the number of rows required for the subplot grid by rounding up the division of the number of categorical columns by n_cols.
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))

    # Create a figure object and an axes grid with the calculated number of rows and columns.
    # figsize sets the overall figure size: width = n_cols * 5 inches, height = n_rows * 4 inches.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    # Flatten the 2D axes array into a 1D list for easier iteration when multiple subplots exist.
    # If only one subplot is created, wrap the single axis object in a list for consistency.
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Iterate through each categorical column with its index and column name.
    for i, col in enumerate(categorical_cols):
        # Count the frequency of each category in the current column using value_counts method.
        counts = df[col].value_counts()
        # Plot the bar chart on the current axis using the category names as x and their counts as y, with color 'royalblue'.
        axes[i].bar(counts.index, counts.values, color='royalblue')
        # Set the title of the subplot to display the column name with fontsize 14.
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        # Remove the x-axis label for a cleaner appearance (empty string removes default label).
        axes[i].set_xlabel('')
        # Set the y-axis label to 'Frequency' to indicate what the bars represent.
        axes[i].set_ylabel('Frequency')
        # Rotate x-ticks by 45 degrees if there are many categories to prevent overlapping.
        axes[i].tick_params(axis='x', rotation=45)

    # Hide any unused axes in the grid (when total subplots > categorical columns).
    # This prevents empty subplot frames from appearing in the figure.
    for ax in axes[len(categorical_cols):]:
        ax.set_visible(False)

    # Adjust the layout to prevent titles and labels from overlapping with other elements.
    fig.tight_layout()
    # Display the complete figure in the output window or notebook.
    plt.show()