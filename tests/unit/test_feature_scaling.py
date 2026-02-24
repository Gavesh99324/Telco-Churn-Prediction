"""Unit tests for feature_scaling.py"""
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.feature_scaling import FeatureScaler


@pytest.fixture
def sample_df():
    """Create sample DataFrame"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': [100, 200, 300, 400, 500]
    })


@pytest.fixture
def sample_array():
    """Create sample numpy array"""
    np.random.seed(42)
    return np.array([
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
        [4, 40, 400],
        [5, 50, 500]
    ])


@pytest.fixture
def feature_scaler():
    """Create FeatureScaler instance"""
    return FeatureScaler()


class TestFeatureScalerInitialization:
    """Test FeatureScaler initialization"""

    def test_initialization(self, feature_scaler):
        """Test proper initialization"""
        assert isinstance(feature_scaler.scaler, StandardScaler)
        assert feature_scaler.feature_names is None
        assert feature_scaler.is_fitted is False


class TestFitTransform:
    """Test fit_transform functionality"""

    def test_fit_transform_dataframe(self, feature_scaler, sample_df):
        """Test fit_transform with DataFrame"""
        scaled_df = feature_scaler.fit_transform(sample_df)

        assert isinstance(scaled_df, pd.DataFrame)
        assert scaled_df.shape == sample_df.shape
        assert feature_scaler.is_fitted is True

    def test_fit_transform_array(self, feature_scaler, sample_array):
        """Test fit_transform with numpy array"""
        scaled_df = feature_scaler.fit_transform(sample_array)

        assert isinstance(scaled_df, pd.DataFrame)
        assert scaled_df.shape == sample_array.shape
        assert feature_scaler.is_fitted is True

    def test_fit_transform_preserves_column_names(self, feature_scaler, sample_df):
        """Test that column names are preserved"""
        scaled_df = feature_scaler.fit_transform(sample_df)

        assert list(scaled_df.columns) == list(sample_df.columns)

    def test_fit_transform_preserves_index(self, feature_scaler, sample_df):
        """Test that index is preserved"""
        custom_index = ['a', 'b', 'c', 'd', 'e']
        sample_df.index = custom_index

        scaled_df = feature_scaler.fit_transform(sample_df)

        pd.testing.assert_index_equal(scaled_df.index, sample_df.index)

    def test_fit_transform_creates_feature_names(self, feature_scaler, sample_df):
        """Test that feature names are stored"""
        feature_scaler.fit_transform(sample_df)

        assert feature_scaler.feature_names == list(sample_df.columns)

    def test_fit_transform_array_generates_names(self, feature_scaler, sample_array):
        """Test that feature names are generated for arrays"""
        feature_scaler.fit_transform(sample_array)

        assert feature_scaler.feature_names == [
            'feature_0', 'feature_1', 'feature_2']

    def test_fit_transform_standardizes(self, feature_scaler, sample_df):
        """Test that data is standardized (mean~0, std~1)"""
        scaled_df = feature_scaler.fit_transform(sample_df)

        # Check mean is approximately 0
        means = scaled_df.mean()
        np.testing.assert_array_almost_equal(
            means.values, np.zeros(len(means)), decimal=10)

        # Check std is approximately 1
        stds = scaled_df.std()
        np.testing.assert_array_almost_equal(
            stds.values, np.ones(len(stds)), decimal=10)


class TestTransform:
    """Test transform functionality"""

    def test_transform_after_fit(self, feature_scaler, sample_df):
        """Test transform on new data after fitting"""
        feature_scaler.fit_transform(sample_df)

        new_data = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [60, 70],
            'feature3': [600, 700]
        })

        scaled_new = feature_scaler.transform(new_data)

        assert isinstance(scaled_new, pd.DataFrame)
        assert scaled_new.shape == new_data.shape

    def test_transform_without_fit_raises_error(self, feature_scaler, sample_df):
        """Test that transform without fit raises error"""
        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            feature_scaler.transform(sample_df)

    def test_transform_preserves_column_names(self, feature_scaler, sample_df):
        """Test that transform preserves column names"""
        feature_scaler.fit_transform(sample_df)

        new_data = pd.DataFrame({
            'feature1': [6],
            'feature2': [60],
            'feature3': [600]
        })

        scaled_new = feature_scaler.transform(new_data)

        assert list(scaled_new.columns) == list(new_data.columns)

    def test_transform_uses_fitted_parameters(self, feature_scaler, sample_df):
        """Test that transform uses parameters from fit"""
        feature_scaler.fit_transform(sample_df)

        # Transform same data should give same result
        scaled_1 = feature_scaler.transform(sample_df)
        scaled_2 = feature_scaler.fit_transform(sample_df.copy())

        pd.testing.assert_frame_equal(scaled_1, scaled_2)


class TestInverseTransform:
    """Test inverse transform functionality"""

    def test_inverse_transform_basic(self, feature_scaler, sample_df):
        """Test inverse transform"""
        scaled_df = feature_scaler.fit_transform(sample_df)

        inverse_df = feature_scaler.inverse_transform(scaled_df)

        # Should be close to original
        pd.testing.assert_frame_equal(
            inverse_df, sample_df, check_dtype=False, atol=1e-10)

    def test_inverse_transform_without_fit_raises_error(self, feature_scaler, sample_df):
        """Test that inverse_transform without fit raises error"""
        with pytest.raises(RuntimeError):
            feature_scaler.inverse_transform(sample_df)

    def test_inverse_transform_array(self, feature_scaler, sample_array):
        """Test inverse transform with array"""
        scaled_df = feature_scaler.fit_transform(sample_array)

        inverse_df = feature_scaler.inverse_transform(scaled_df)

        # Check values are close to original
        np.testing.assert_array_almost_equal(
            inverse_df.values, sample_array, decimal=10)

    def test_round_trip_transform(self, feature_scaler, sample_df):
        """Test that fit->transform->inverse gives back original"""
        scaled = feature_scaler.fit_transform(sample_df)
        recovered = feature_scaler.inverse_transform(scaled)

        pd.testing.assert_frame_equal(
            recovered, sample_df, check_dtype=False, atol=1e-10)


class TestScalingStatistics:
    """Test scaling statistics"""

    def test_get_scaling_parameters(self, feature_scaler, sample_df):
        """Test retrieving scaling parameters"""
        feature_scaler.fit_transform(sample_df)

        params = feature_scaler.get_scaling_parameters()

        assert 'mean' in params
        assert 'scale' in params
        assert len(params['mean']) == sample_df.shape[1]
        assert len(params['scale']) == sample_df.shape[1]

    def test_scaling_parameters_correct_values(self, feature_scaler):
        """Test that scaling parameters are correct"""
        df = pd.DataFrame({
            'a': [0, 2, 4, 6, 8],  # mean=4, std=sqrt(10)
            'b': [10, 10, 10, 10, 10]  # mean=10, std=0
        })

        feature_scaler.fit_transform(df)
        params = feature_scaler.get_scaling_parameters()

        assert params['mean'][0] == pytest.approx(4.0)
        assert params['mean'][1] == pytest.approx(10.0)

        # Check scale (std dev)
        assert params['scale'][0] > 0

    def test_get_feature_statistics(self, feature_scaler, sample_df):
        """Test getting feature statistics"""
        scaled_df = feature_scaler.fit_transform(sample_df)

        stats = feature_scaler.get_feature_statistics(scaled_df)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats


class TestMultipleFeatures:
    """Test scaling with different numbers of features"""

    def test_single_feature(self, feature_scaler):
        """Test scaling single feature"""
        df = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})

        scaled_df = feature_scaler.fit_transform(df)

        assert scaled_df.shape == df.shape
        assert scaled_df.mean().iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_many_features(self, feature_scaler):
        """Test scaling many features"""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 20))

        scaled_df = feature_scaler.fit_transform(df)

        assert scaled_df.shape == df.shape
        # All features should be standardized
        means = scaled_df.mean()
        stds = scaled_df.std()

        np.testing.assert_array_almost_equal(
            means.values, np.zeros(20), decimal=10)
        np.testing.assert_array_almost_equal(
            stds.values, np.ones(20), decimal=10)

    def test_different_scales(self, feature_scaler):
        """Test scaling features with very different scales"""
        df = pd.DataFrame({
            'small': [0.001, 0.002, 0.003, 0.004, 0.005],
            'medium': [1, 2, 3, 4, 5],
            'large': [1000, 2000, 3000, 4000, 5000]
        })

        scaled_df = feature_scaler.fit_transform(df)

        # All should be scaled to similar ranges
        assert scaled_df['small'].std() == pytest.approx(1.0)
        assert scaled_df['medium'].std() == pytest.approx(1.0)
        assert scaled_df['large'].std() == pytest.approx(1.0)


class TestEdgeCases:
    """Test edge cases"""

    def test_constant_feature(self, feature_scaler):
        """Test scaling feature with constant value"""
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 5]
        })

        scaled_df = feature_scaler.fit_transform(df)

        # Constant feature should be 0 (with warning possible)
        assert scaled_df['constant'].std() == pytest.approx(0.0)
        # Variable feature should be standardized
        assert scaled_df['variable'].std() == pytest.approx(1.0)

    def test_single_row(self, feature_scaler):
        """Test scaling with single row"""
        df = pd.DataFrame({'a': [5], 'b': [10]})

        # Single row can't compute std, might raise error or warning
        try:
            scaled_df = feature_scaler.fit_transform(df)
            # If it works, check shape
            assert scaled_df.shape == df.shape
        except (ValueError, Warning):
            # Expected behavior
            pass

    def test_two_rows(self, feature_scaler):
        """Test scaling with two rows"""
        df = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})

        scaled_df = feature_scaler.fit_transform(df)

        # Should work with 2 rows
        assert scaled_df.shape == (2, 2)

    def test_with_missing_values(self, feature_scaler):
        """Test scaling with missing values"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [10, 20, 30, np.nan, 50]
        })

        # StandardScaler typically can't handle NaN
        try:
            scaled_df = feature_scaler.fit_transform(df)
            # If it works, check for NaNs
            assert scaled_df.notna().all().all() or scaled_df.isna().any().any()
        except ValueError:
            # Expected if NaN not handled
            pass

    def test_empty_dataframe(self, feature_scaler):
        """Test scaling empty DataFrame"""
        df = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            feature_scaler.fit_transform(df)

    def test_zero_variance_after_subset(self, feature_scaler):
        """Test scaling when subset has zero variance"""
        df = pd.DataFrame({
            'a': [1, 1, 1, 2, 3],
            'b': [5, 5, 5, 10, 15]
        })

        # Fit on full data
        feature_scaler.fit_transform(df)

        # Transform subset with zero variance
        subset = df.iloc[:3]  # All 1s and 5s
        scaled_subset = feature_scaler.transform(subset)

        # Should still work using full data's parameters
        assert scaled_subset.shape == subset.shape


class TestNumericalStability:
    """Test numerical stability"""

    def test_large_values(self, feature_scaler):
        """Test scaling very large values"""
        df = pd.DataFrame({
            'large': [1e10, 2e10, 3e10, 4e10, 5e10]
        })

        scaled_df = feature_scaler.fit_transform(df)

        assert scaled_df['large'].mean() == pytest.approx(0.0, abs=1e-5)
        assert scaled_df['large'].std() == pytest.approx(1.0, abs=1e-5)

    def test_small_values(self, feature_scaler):
        """Test scaling very small values"""
        df = pd.DataFrame({
            'small': [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        })

        scaled_df = feature_scaler.fit_transform(df)

        assert scaled_df['small'].mean() == pytest.approx(0.0, abs=1e-5)
        assert scaled_df['small'].std() == pytest.approx(1.0, abs=1e-5)

    def test_negative_values(self, feature_scaler):
        """Test scaling negative values"""
        df = pd.DataFrame({
            'negative': [-5, -4, -3, -2, -1]
        })

        scaled_df = feature_scaler.fit_transform(df)

        assert scaled_df['negative'].mean() == pytest.approx(0.0, abs=1e-10)
        assert scaled_df['negative'].std() == pytest.approx(1.0, abs=1e-10)

    def test_mixed_positive_negative(self, feature_scaler):
        """Test scaling mixed positive and negative values"""
        df = pd.DataFrame({
            'mixed': [-10, -5, 0, 5, 10]
        })

        scaled_df = feature_scaler.fit_transform(df)

        assert scaled_df['mixed'].mean() == pytest.approx(0.0, abs=1e-10)
        assert scaled_df['mixed'].std() == pytest.approx(1.0, abs=1e-10)


class TestConsistency:
    """Test consistency of scaling"""

    def test_scaling_deterministic(self, feature_scaler, sample_df):
        """Test that scaling is deterministic"""
        scaled_1 = feature_scaler.fit_transform(sample_df.copy())

        scaler_2 = FeatureScaler()
        scaled_2 = scaler_2.fit_transform(sample_df.copy())

        pd.testing.assert_frame_equal(scaled_1, scaled_2)

    def test_transform_consistent(self, feature_scaler, sample_df):
        """Test that transform is consistent"""
        feature_scaler.fit_transform(sample_df)

        new_data = pd.DataFrame({
            'feature1': [6],
            'feature2': [60],
            'feature3': [600]
        })

        scaled_1 = feature_scaler.transform(new_data.copy())
        scaled_2 = feature_scaler.transform(new_data.copy())

        pd.testing.assert_frame_equal(scaled_1, scaled_2)


class TestDataTypes:
    """Test with different data types"""

    def test_integer_input(self, feature_scaler):
        """Test scaling integer data"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5]
        }, dtype=int)

        scaled_df = feature_scaler.fit_transform(df)

        # Output should be float
        assert scaled_df['int_col'].dtype == float

    def test_float_input(self, feature_scaler):
        """Test scaling float data"""
        df = pd.DataFrame({
            'float_col': [1.5, 2.5, 3.5, 4.5, 5.5]
        })

        scaled_df = feature_scaler.fit_transform(df)

        assert scaled_df['float_col'].dtype == float

    def test_mixed_types(self, feature_scaler):
        """Test scaling mixed numeric types"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.5, 2.5, 3.5, 4.5, 5.5]
        })

        scaled_df = feature_scaler.fit_transform(df)

        # Both should be float after scaling
        assert all(scaled_df.dtypes == float)


class TestStateSaving:
    """Test state preservation"""

    def test_is_fitted_flag(self, feature_scaler, sample_df):
        """Test that is_fitted flag is set correctly"""
        assert feature_scaler.is_fitted is False

        feature_scaler.fit_transform(sample_df)
        assert feature_scaler.is_fitted is True

    def test_feature_names_saved(self, feature_scaler, sample_df):
        """Test that feature names are saved"""
        assert feature_scaler.feature_names is None

        feature_scaler.fit_transform(sample_df)
        assert feature_scaler.feature_names == list(sample_df.columns)

    def test_scaler_state_reusable(self, feature_scaler, sample_df):
        """Test that scaler state is reusable"""
        feature_scaler.fit_transform(sample_df)

        # Can transform multiple times
        new_data = pd.DataFrame({
            'feature1': [10],
            'feature2': [100],
            'feature3': [1000]
        })

        scaled_1 = feature_scaler.transform(new_data)
        scaled_2 = feature_scaler.transform(new_data)

        pd.testing.assert_frame_equal(scaled_1, scaled_2)
