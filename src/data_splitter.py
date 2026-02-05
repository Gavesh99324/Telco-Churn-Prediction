"""Train/test split"""
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Split data into train and test sets"""

    def split(self, X, y, test_size=0.2, random_state=42):
        """Split data"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
