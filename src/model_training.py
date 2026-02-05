"""Training logic"""


class ModelTrainer:
    """Train ML models"""

    def train(self, model, X_train, y_train):
        """Train the model"""
        model.fit(X_train, y_train)
        return model
