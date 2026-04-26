"""
Module for creating a universal preprocessing pipeline for machine learning models.

This module provides a function to automatically build a preprocessing pipeline
that handles numeric and categorical features using scikit-learn components.
"""

from sklearn.pipeline import Pipeline  # Import Pipeline for chaining steps
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Import StandardScaler for scaling and OneHotEncoder for encoding
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for applying different preprocessing to different columns

def get_preprocessing_pipeline(df, model):
    """
    Create a universal preprocessing pipeline for a given DataFrame and model.

    This function automatically detects numeric and categorical columns in the DataFrame,
    applies appropriate preprocessing steps (imputation, scaling for numeric; imputation,
    one-hot encoding for categorical), and combines them with the specified model into a Pipeline.

    Parameters:
    df (pd.DataFrame): The input DataFrame to analyze for column types.
    model: The machine learning model to append to the preprocessing steps.

    Returns:
    Pipeline: A scikit-learn Pipeline object with preprocessing and model steps.
    """
    # 1. Automatically detect which columns are numbers and which are strings
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns  # Select columns with numeric data types (integers or floats)
    categorical_features = df.select_dtypes(include=['object']).columns  # Select columns with object data type (typically strings)

    # 2. Standard steps for ANY numbers
    num_transformer = Pipeline(steps=[  # Create a pipeline for numeric features
        ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with the median
        ('scaler', StandardScaler())  # Scale features to have mean 0 and variance 1
    ])

    # 3. Standard steps for ANY categories
    cat_transformer = Pipeline(steps=[  # Create a pipeline for categorical features
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical variables as one-hot vectors, ignoring unknown categories
    ])

    # 4. Combine based on the detected columns
    preprocessor = ColumnTransformer(  # Create a column transformer to apply different preprocessing
        transformers=[  # List of transformers
            ('num', num_transformer, numeric_features),  # Apply num_transformer to numeric features
            ('cat', cat_transformer, categorical_features)  # Apply cat_transformer to categorical features
        ])

    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])  # Return a pipeline with preprocessor and model

"""3. The "Feature Engineering" Catch
There is one thing a universal pipeline cannot do: create features like FamilySize or Title.

These are "domain-specific."

FamilySize makes sense for the Titanic, but it makes zero sense for a dataset about Weather or Stock Prices.

The Solution:
Keep your preprocessing_utils_pipeline.py for your general steps (Scaling, Imputing, One-Hot Encoding)
and use a separate function or a Custom Transformer for the Titanic-specific "math" like adding SibSp+ Parch.

How to handle Feature Engineering (The "Titanic-Specific" Logic)
Since you want to keep your pipeline universal, you should perform your "math" (like SibSp + Parch) in a simple
function inside your titanic_model.ipynb or a small specific module.

The Workflow:

Raw Data → 2. Titanic-Specific Function (Add FamilySize, Title, Deck) → 3. Universal Pipeline 
(Impute, Scale, One-Hot, Model).

Example in your notebook:

Python
def add_titanic_features(df):
    df_copy = df.copy()
    if 'SibSp' in df_copy.columns and 'Parch' in df_copy.columns:
        df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
    # Add other Titanic-specific logic here...
    return df_copy

# Use it like this:
X_train_engineered = add_titanic_features(X_train)
pipeline = get_universal_pipeline(X_train_engineered, LogisticRegression())
pipeline.fit(X_train_engineered, y_train)
"""