"""Categorical encoding"""
from sklearn.preprocessing import LabelEncoder


class FeatureEncoder:
    """Encode categorical features"""

    def __init__(self):
        self.encoders = {}

    def fit_transform(self, df, columns):
        """Fit and transform categorical columns"""
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        return df
