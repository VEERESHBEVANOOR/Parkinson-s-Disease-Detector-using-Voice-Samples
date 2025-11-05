# check_data.py
import pandas as pd

try:
    df = pd.read_csv('parkinsons_features.csv')
    print("Count of samples per class in parkinsons_features.csv:")
    print(df['status'].value_counts())
except FileNotFoundError:
    print("Error: parkinsons_features.csv does not exist!")