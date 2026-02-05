"""Feature binning"""
import pandas as pd


class FeatureBinner:
    """Create bins for continuous features"""

    def bin_feature(self, df, column, bins, labels=None):
        """Bin a feature into categories"""
        return pd.cut(df[column], bins=bins, labels=labels)
