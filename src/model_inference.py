"""Inference logic"""


class ModelInference:
    """Run model inference"""

    def predict(self, model, X):
        """Make predictions"""
        return model.predict(X)

    def predict_proba(self, model, X):
        """Get prediction probabilities"""
        return model.predict_proba(X)
