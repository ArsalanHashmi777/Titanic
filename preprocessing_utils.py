"""Plotting utilities for Titanic dataset visualizations."""

import pandas as pd
import numpy as np

def full_preprocess(df):
    """
    Preprocess and engineer features for Titanic dataset.
    
    This function performs comprehensive data preprocessing including:
    - Feature engineering (Family Size, Title extraction, Deck extraction)
    - Handling missing values with appropriate imputation strategies
    
    Args:
        df (pd.DataFrame): Raw Titanic dataset as a pandas DataFrame
    Returns:
        pd.DataFrame: Processed dataframe with engineered features and no missing values
    """
    
    # Make a copy so we don't change the original data by accident
    data = df.copy()
    
    # 1. Feature Engineering: Family Size
    # Calculate total family members (siblings + spouse + parents + children + self)
    if 'SibSp' in data.columns and 'Parch' in data.columns:
        # Check that both required columns exist before processing
        # SibSp = number of siblings/spouses, Parch = number of parents/children, +1 for self
        # Create new FamilySize feature by summing SibSp (siblings/spouse) + Parch (parents/children) + 1 (self)
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # 2. Feature Engineering: Extract Title from Name
    if 'Name' in data.columns:
        # Check that Name column exists before extracting title
        # Extract title (Mr., Mrs., Miss., etc.) using regex pattern matching
        # Pattern ' ([A-Za-z]+)\.' matches a space, followed by letters, followed by a period
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Consolidate rare titles into a single 'Rare' category to reduce cardinality
        # This groups infrequent titles that would not provide meaningful signal for the model
        data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        
        # Standardize titles: combine similar variations (Mlle and Ms to Miss, Mme to Mrs)
        # This further reduces cardinality by merging semantically similar categories
        data['Title'] = data['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')

    # 3. Handling Deck from Cabin (as we discussed before)
    # Extract the deck letter from cabin number (first character indicates the deck)
    if 'Cabin' in data.columns:
        # Check that Cabin column exists before extracting deck information
        # For non-null cabin values, take first character; for null values, assign 'U' (Unknown)
        # Lambda function converts each cabin value to string and takes the first character
        # If cabin value is null, the ternary operator assigns 'U' instead
        data['Deck'] = data['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

    # 4. Handle Missing Values
    # Fill missing Age values with the median age to preserve the distribution
    # Median is preferred over mean as it's more robust to outliers in age data
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    # Fill missing Embarked values with the most common port of embarkation (mode)
    # Using mode preserves the most frequent embarkation port, which is realistic
    # [0] is used because mode() returns a Series; we take the first (most frequent) value
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Fill missing Fare values with the median fare to avoid bias
    # Median is used for the same reason as Age: robustness against outliers
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Return the preprocessed dataframe with all features engineered and missing values handled
    return data