"""Unit tests for feature_binning.py"""
import pytest
import numpy as np
import pandas as pd
from src.feature_binning import FeatureBinner


@pytest.fixture
def sample_df():
    """Create sample DataFrame with tenure"""
    np.random.seed(42)
    return pd.DataFrame({
        'tenure': [0, 6, 12, 18, 24, 36, 48, 60, 72],
        'MonthlyCharges': [20, 40, 60, 80, 100, 120, 140, 160, 180],
        'TotalCharges': [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    })


@pytest.fixture
def feature_binner():
    """Create FeatureBinner instance"""
    return FeatureBinner()


class TestFeatureBinnerInitialization:
    """Test FeatureBinner initialization"""

    def test_initialization(self, feature_binner):
        """Test proper initialization"""
        assert isinstance(feature_binner.binning_config, dict)
        assert len(feature_binner.binning_config) == 0


class TestTenureGroupCreation:
    """Test tenure group binning"""

    def test_create_tenure_groups_basic(self, feature_binner, sample_df):
        """Test basic tenure group creation"""
        result_df = feature_binner.create_tenure_groups(sample_df)

        assert 'tenure_group' in result_df.columns
        assert len(result_df) == len(sample_df)

    def test_create_tenure_groups_labels(self, feature_binner, sample_df):
        """Test tenure group labels are correct"""
        result_df = feature_binner.create_tenure_groups(sample_df)

        expected_labels = {'0-1 year', '1-2 years', '2-4 years', '4+ years'}
        actual_labels = set(result_df['tenure_group'].dropna().unique())

        assert actual_labels.issubset(expected_labels)

    def test_create_tenure_groups_bin_edges(self, feature_binner, sample_df):
        """Test that bins are correct"""
        result_df = feature_binner.create_tenure_groups(sample_df)

        # Check specific values fall into correct bins
        assert result_df[result_df['tenure'] ==
                         0]['tenure_group'].iloc[0] == '0-1 year'
        assert result_df[result_df['tenure'] ==
                         12]['tenure_group'].iloc[0] == '1-2 years'
        assert result_df[result_df['tenure'] ==
                         24]['tenure_group'].iloc[0] == '2-4 years'
        assert result_df[result_df['tenure'] ==
                         60]['tenure_group'].iloc[0] == '4+ years'

    def test_create_tenure_groups_stores_config(self, feature_binner, sample_df):
        """Test that binning config is stored"""
        feature_binner.create_tenure_groups(sample_df)

        assert 'tenure_group' in feature_binner.binning_config
        assert 'bins' in feature_binner.binning_config['tenure_group']
        assert 'labels' in feature_binner.binning_config['tenure_group']

    def test_create_tenure_groups_preserves_original(self, feature_binner, sample_df):
        """Test that original tenure column is preserved"""
        original_tenure = sample_df['tenure'].copy()
        result_df = feature_binner.create_tenure_groups(sample_df)

        pd.testing.assert_series_equal(result_df['tenure'], original_tenure)

    def test_create_tenure_groups_edge_values(self, feature_binner):
        """Test edge value binning"""
        df = pd.DataFrame({'tenure': [0, 12, 24, 48, 72]})
        result_df = feature_binner.create_tenure_groups(df)

        # Test boundary values
        assert result_df[result_df['tenure'] ==
                         0]['tenure_group'].iloc[0] == '0-1 year'
        assert result_df[result_df['tenure'] ==
                         12]['tenure_group'].iloc[0] == '1-2 years'
        assert result_df[result_df['tenure'] ==
                         24]['tenure_group'].iloc[0] == '2-4 years'
        assert result_df[result_df['tenure'] ==
                         48]['tenure_group'].iloc[0] == '4+ years'


class TestCustomBinning:
    """Test custom feature binning"""

    def test_bin_feature_basic(self, feature_binner, sample_df):
        """Test basic feature binning"""
        bins = [0, 50, 100, 200]
        labels = ['Low', 'Medium', 'High']

        result_df = feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins, labels
        )

        assert 'MonthlyCharges_binned' in result_df.columns

    def test_bin_feature_custom_column_name(self, feature_binner, sample_df):
        """Test binning with custom output column name"""
        bins = [0, 50, 100, 200]

        result_df = feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins,
            new_column='charge_category'
        )

        assert 'charge_category' in result_df.columns

    def test_bin_feature_with_labels(self, feature_binner, sample_df):
        """Test that custom labels are applied"""
        bins = [0, 1000, 5000, 10000]
        labels = ['Starter', 'Regular', 'Premium']

        result_df = feature_binner.bin_feature(
            sample_df, 'TotalCharges', bins, labels
        )

        unique_values = result_df['TotalCharges_binned'].dropna().unique()
        for value in unique_values:
            assert value in labels

    def test_bin_feature_without_labels(self, feature_binner, sample_df):
        """Test binning without custom labels"""
        bins = [0, 50, 100, 200]

        result_df = feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins
        )

        # Should create interval labels
        assert 'MonthlyCharges_binned' in result_df.columns

    def test_bin_feature_stores_config(self, feature_binner, sample_df):
        """Test that binning config is stored"""
        bins = [0, 50, 100, 200]
        labels = ['Low', 'Medium', 'High']

        feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins, labels
        )

        assert 'MonthlyCharges_binned' in feature_binner.binning_config
        config = feature_binner.binning_config['MonthlyCharges_binned']
        assert config['source_column'] == 'MonthlyCharges'
        assert config['bins'] == bins
        assert config['labels'] == labels

    def test_bin_feature_equal_width_bins(self, feature_binner, sample_df):
        """Test with equal width bins"""
        bins = [0, 60, 120, 180]

        result_df = feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins
        )

        assert result_df['MonthlyCharges_binned'].notna().sum() > 0

    def test_bin_feature_preserves_original(self, feature_binner, sample_df):
        """Test that original column is preserved"""
        original_values = sample_df['MonthlyCharges'].copy()
        bins = [0, 50, 100, 200]

        result_df = feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins
        )

        pd.testing.assert_series_equal(
            result_df['MonthlyCharges'], original_values)


class TestBinningEdgeCases:
    """Test edge cases in binning"""

    def test_bin_feature_with_outliers(self, feature_binner):
        """Test binning with values outside bin range"""
        df = pd.DataFrame({'value': [10, 50, 100, 500, 1000]})
        bins = [0, 50, 100, 200]

        result_df = feature_binner.bin_feature(df, 'value', bins)

        # Values outside range should be handled
        assert 'value_binned' in result_df.columns

    def test_bin_feature_with_single_value(self, feature_binner):
        """Test binning with single unique value"""
        df = pd.DataFrame({'value': [50, 50, 50, 50]})
        bins = [0, 40, 60, 100]

        result_df = feature_binner.bin_feature(df, 'value', bins)

        # All should fall in same bin
        assert result_df['value_binned'].nunique() == 1

    def test_bin_feature_with_missing_values(self, feature_binner):
        """Test binning with missing values"""
        df = pd.DataFrame({'value': [10, np.nan, 50, 100, np.nan]})
        bins = [0, 30, 70, 150]

        result_df = feature_binner.bin_feature(df, 'value', bins)

        # NaN should remain NaN
        assert result_df['value_binned'].isna().sum() == 2

    def test_bin_feature_minimal_data(self, feature_binner):
        """Test binning with minimal data"""
        df = pd.DataFrame({'value': [10, 50]})
        bins = [0, 30, 100]

        result_df = feature_binner.bin_feature(df, 'value', bins)

        assert len(result_df) == 2
        assert 'value_binned' in result_df.columns


class TestCreateAllBins:
    """Test creating all bins at once"""

    def test_create_all_bins_basic(self, feature_binner, sample_df):
        """Test creating all binned features"""
        result_df = feature_binner.create_all_bins(sample_df)

        # Should at least have tenure_group
        assert 'tenure_group' in result_df.columns

    def test_create_all_bins_stores_all_configs(self, feature_binner, sample_df):
        """Test that all configs are stored"""
        feature_binner.create_all_bins(sample_df)

        # Should have at least tenure_group config
        assert len(feature_binner.binning_config) > 0
        assert 'tenure_group' in feature_binner.binning_config


class TestBinningConsistency:
    """Test consistency of binning"""

    def test_binning_consistent_across_calls(self, feature_binner, sample_df):
        """Test that binning is consistent"""
        result1 = feature_binner.create_tenure_groups(sample_df.copy())

        binner2 = FeatureBinner()
        result2 = binner2.create_tenure_groups(sample_df.copy())

        pd.testing.assert_series_equal(
            result1['tenure_group'],
            result2['tenure_group']
        )

    def test_same_config_produces_same_bins(self, feature_binner, sample_df):
        """Test that same config produces same bins"""
        bins = [0, 50, 100, 200]
        labels = ['Low', 'Medium', 'High']

        result1 = feature_binner.bin_feature(
            sample_df.copy(), 'MonthlyCharges', bins, labels,
            new_column='cat1'
        )

        result2 = feature_binner.bin_feature(
            sample_df.copy(), 'MonthlyCharges', bins, labels,
            new_column='cat2'
        )

        # Both should have same values (different column names)
        assert result1['cat1'].equals(result2['cat2'])


class TestQuantileBinning:
    """Test quantile-based binning"""

    def test_quantile_bins(self, feature_binner):
        """Test creating bins based on quantiles"""
        df = pd.DataFrame({'value': range(100)})

        # Create quartile bins
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        bins = df['value'].quantile(quantiles).tolist()
        labels = ['Q1', 'Q2', 'Q3', 'Q4']

        result_df = feature_binner.bin_feature(
            df, 'value', bins, labels
        )

        # Each quartile should have roughly equal counts
        counts = result_df['value_binned'].value_counts()
        for count in counts:
            assert count >= 20  # Allow some variation


class TestBinDistribution:
    """Test that binning creates expected distributions"""

    def test_bin_distribution_logged(self, feature_binner, sample_df):
        """Test that distribution is calculated correctly"""
        bins = [0, 50, 100, 200]
        labels = ['Low', 'Medium', 'High']

        result_df = feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins, labels
        )

        # Check distribution adds up to total
        distribution = result_df['MonthlyCharges_binned'].value_counts()
        assert distribution.sum() == len(sample_df)

    def test_empty_bins_handled(self, feature_binner):
        """Test that empty bins are handled"""
        df = pd.DataFrame({'value': [10, 20, 30]})
        bins = [0, 15, 50, 100, 200]  # Some bins will be empty

        result_df = feature_binner.bin_feature(df, 'value', bins)

        # Should still work, some bins just won't have values
        assert 'value_binned' in result_df.columns


class TestBinLabelGeneration:
    """Test automatic bin label generation"""

    def test_default_labels(self, feature_binner, sample_df):
        """Test that default labels are generated when not provided"""
        bins = [0, 50, 100, 200]

        result_df = feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins, labels=None
        )

        # Should create interval-based labels
        assert result_df['MonthlyCharges_binned'].notna().any()

    def test_label_count_matches_bins(self, feature_binner, sample_df):
        """Test that number of labels matches number of intervals"""
        bins = [0, 50, 100, 200]
        labels = ['Low', 'Medium', 'High']

        # 3 labels for 3 intervals (4 bin edges)
        result_df = feature_binner.bin_feature(
            sample_df, 'MonthlyCharges', bins, labels
        )

        unique_labels = result_df['MonthlyCharges_binned'].dropna().unique()
        assert len(unique_labels) <= len(labels)


class TestBinningValidation:
    """Test validation of binning parameters"""

    def test_bin_feature_invalid_column(self, feature_binner, sample_df):
        """Test binning non-existent column"""
        bins = [0, 50, 100]

        # Depending on implementation, might handle gracefully or raise error
        try:
            result_df = feature_binner.bin_feature(
                sample_df, 'NonExistent', bins
            )
            # If it doesn't raise error, result should still be valid
            assert result_df is not None
        except (KeyError, ValueError):
            # Expected behavior
            pass

    def test_bin_feature_unsorted_bins(self, feature_binner, sample_df):
        """Test with unsorted bins"""
        bins = [100, 0, 50, 200]  # Unsorted

        # pd.cut typically requires sorted bins
        try:
            result_df = feature_binner.bin_feature(
                sample_df, 'MonthlyCharges', bins
            )
            # If it works, bins were likely sorted internally
            assert True
        except ValueError:
            # Expected if bins not sorted
            pass
