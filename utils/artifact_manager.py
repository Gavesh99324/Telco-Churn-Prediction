"""Artifact utilities"""
import pickle


class ArtifactManager:
    """Manage ML artifacts locally"""

    @staticmethod
    def save_pickle(obj, filepath):
        """Save object as pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(filepath):
        """Load pickle object"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
