"""
Module for creating a universal preprocessing pipeline for machine learning models.

This module provides a function to automatically build a preprocessing pipeline
that handles numeric and categorical features using scikit-learn components.
"""

from sklearn.pipeline import Pipeline  # Import Pipeline for chaining steps
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Import StandardScaler for scaling and OneHotEncoder for encoding
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for applying different preprocessing to different columns

def add_titanic_features(df):
    """
    Creates new features to help the model understand social status 
    and family dynamics on the Titanic.
    """
    df_copy = df.copy()
    
    # 1. Handle Cabin 'U' (Unknown)
    # We do this BEFORE encoding so 'U' becomes its own category
    if 'Cabin' in df_copy.columns:
        df_copy['Cabin'] = df_copy['Cabin'].fillna('U')
        # Extract the first letter to create the 'Deck' feature
        # e.g., 'C22' becomes 'C', 'U' stays 'U'
        # We simplify 'Cabin' into 'Deck' (first letter) to reduce noise and overfitting risk.
        df_copy['Deck'] = df_copy['Cabin'].astype(str).str[0]
        # One last check: Since your get_preprocessing_pipeline handles categorical data, 
        # it will see Deck and automatically apply One-Hot Encoding to it.
        
        # We drop the high-cardinality 'Cabin' column to prevent overfitting
        df_copy = df_copy.drop(['Cabin'], axis=1)
        
    # 2. FamilySize: Total people in the family (including the passenger)
    if 'SibSp' in df_copy.columns and 'Parch' in df_copy.columns:
        df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
        
        # 3. IsAlone: A binary flag (1 if alone, 0 if with family)
        # Often, solo travelers had lower survival rates in 3rd class.
        df_copy['IsAlone'] = 0
        df_copy.loc[df_copy['FamilySize'] == 1, 'IsAlone'] = 1
    
    # 4. Title Extraction: Extract 'Mr', 'Mrs', 'Miss', etc., from the Name column
    # Titles are a proxy for both age and social standing.
    if 'Name' in df_copy.columns:
        # Extracts the string that ends with a period (e.g., "Braund, Mr. Owen")
        df_copy['Title'] = df_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles to prevent the model from over-fitting to unique cases
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df_copy['Title'] = df_copy['Title'].replace(rare_titles, 'Rare')
        df_copy['Title'] = df_copy['Title'].replace('Mlle', 'Miss')
        df_copy['Title'] = df_copy['Title'].replace('Ms', 'Miss')
        df_copy['Title'] = df_copy['Title'].replace('Mme', 'Mrs')

    return df_copy

def get_house_features(df):
    """
    Create feature engineering for the House Prices dataset.
    Handles ordinal mapping and feature combinations.
    """
    df_copy = df.copy()
    
    # 1. Map Ordinal Categories (Quality/Condition)
    # Most quality features use this scale
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    
    for col in qual_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(qual_map).fillna(0)

    # 2. Total Square Footage (Feature Combination)
    # A house's total size is often more important than individual floors
    df_copy['TotalSF'] = df_copy['TotalBsmtSF'] + df_copy['1stFlrSF'] + df_copy['2ndFlrSF']
    
    # 3. Total Bathrooms
    df_copy['TotalBath'] = (df_copy['FullBath'] + (0.5 * df_copy['HalfBath']) + 
                            df_copy['BsmtFullBath'] + (0.5 * df_copy['BsmtHalfBath']))

    # 4. House Age (Transforming YearBuilt into something more useful)
    df_copy['HouseAge'] = df_copy['YrSold'] - df_copy['YearBuilt']

    return df_copy

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