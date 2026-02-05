"""Test feature engineering"""
import pytest
from src.feature_scaling import FeatureScaler


def test_feature_scaling():
    """Test feature scaling"""
    scaler = FeatureScaler()
    assert scaler is not None
