"""Model architecture"""
from sklearn.ensemble import RandomForestClassifier


class ModelBuilder:
    """Build ML models"""

    def build_model(self):
        """Build a Random Forest model"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
