import pandas as pd
import sqlite3
import os

# 1. Load the CSVs
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# 2. Connect to (or create) the SQLite database
# This creates a file named 'house_prices.db' in your folder
conn = sqlite3.connect('house_prices.db')

# 3. Push the data into SQL tables
train_df.to_sql('train_records', conn, if_exists='replace', index=False)
test_df.to_sql('test_records', conn, if_exists='replace', index=False)

print("Database 'house_prices.db' created successfully.")
conn.close()

"""
Run this script once. You will see a new file appear called house_prices.db. This file is what 
you will upload to GitHub later. It allows anyone to use SQL on your data without installing a 
heavy database like PostgreSQL.
"""
