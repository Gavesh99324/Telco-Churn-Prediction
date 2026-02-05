"""Feature normalization"""
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Scale features using StandardScaler"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        """Fit and transform features"""
        return self.scaler.fit_transform(X)

    def transform(self, X):
        """Transform features"""
        return self.scaler.transform(X)
