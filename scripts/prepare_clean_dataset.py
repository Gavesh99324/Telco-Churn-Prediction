"""Data cleaning script"""
import pandas as pd


def clean_dataset():
    """Clean the churn dataset"""
    df = pd.read_csv('data/ChurnModelling.csv')
    # Add cleaning logic
    df_clean = df.dropna()
    df_clean.to_csv('data/ChurnModelling_Clean.csv', index=False)
    print("Dataset cleaned successfully!")


if __name__ == '__main__':
    clean_dataset()
