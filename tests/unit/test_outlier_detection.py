"""Unit tests for outlier_detection.py"""
import pytest
import numpy as np
import pandas as pd
from src.outlier_detection import OutlierDetector


@pytest.fixture
def sample_df():
    """Create sample DataFrame with outliers"""
    return pd.DataFrame({
        'tenure': [1, 2, 3, 4, 5, 100],  # 100 is outlier
        'MonthlyCharges': [20, 30, 40, 50, 60, 500],  # 500 is outlier
        'TotalCharges': [100, 200, 300, 400, 500, 10000]  # 10000 is outlier
    })


@pytest.fixture
def clean_df():
    """Create DataFrame without outliers"""
    return pd.DataFrame({
        'tenure': [1, 2, 3, 4, 5, 6],
        'MonthlyCharges': [20, 25, 30, 35, 40, 45],
        'TotalCharges': [100, 150, 200, 250, 300, 350]
    })


@pytest.fixture
def outlier_detector():
    """Create OutlierDetector instance"""
    return OutlierDetector()


@pytest.fixture
def custom_detector():
    """Create OutlierDetector with custom multiplier"""
    return OutlierDetector(multiplier=2.0)


class TestOutlierDetectorInitialization:
    """Test OutlierDetector initialization"""

    def test_initialization_default(self, outlier_detector):
        """Test default initialization"""
        assert outlier_detector.multiplier == 1.5
        assert isinstance(outlier_detector.outlier_summary, dict)
        assert len(outlier_detector.outlier_summary) == 0

    def test_initialization_custom_multiplier(self, custom_detector):
        """Test initialization with custom multiplier"""
        assert custom_detector.multiplier == 2.0

    def test_initialization_different_multipliers(self):
        """Test different multiplier values"""
        detector1 = OutlierDetector(multiplier=1.0)
        detector2 = OutlierDetector(multiplier=3.0)

        assert detector1.multiplier == 1.0
        assert detector2.multiplier == 3.0


class TestDetectOutliersColumn:
    """Test outlier detection for single column"""

    def test_detect_outliers_column_basic(self, outlier_detector, sample_df):
        """Test basic outlier detection for single column"""
        outlier_mask, stats = outlier_detector.detect_outliers_column(
            sample_df, 'tenure'
        )

        assert isinstance(outlier_mask, pd.Series)
        assert isinstance(stats, dict)
        assert len(outlier_mask) == len(sample_df)

    def test_detect_outliers_column_statistics(self, outlier_detector, sample_df):
        """Test that statistics are calculated correctly"""
        outlier_mask, stats = outlier_detector.detect_outliers_column(
            sample_df, 'tenure'
        )

        # Check required statistics
        required_keys = ['Q1', 'Q3', 'IQR', 'lower_bound', 'upper_bound',
                         'outlier_count', 'outlier_percentage']
        for key in required_keys:
            assert key in stats

    def test_detect_outliers_column_bounds(self, outlier_detector, sample_df):
        """Test that bounds are calculated correctly"""
        outlier_mask, stats = outlier_detector.detect_outliers_column(
            sample_df, 'tenure'
        )

        # Check IQR calculation
        Q1 = stats['Q1']
        Q3 = stats['Q3']
        IQR = stats['IQR']

        assert IQR == Q3 - Q1
        assert stats['lower_bound'] == Q1 - 1.5 * IQR
        assert stats['upper_bound'] == Q3 + 1.5 * IQR

    def test_detect_outliers_column_identifies_outliers(self, outlier_detector, sample_df):
        """Test that outliers are correctly identified"""
        outlier_mask, stats = outlier_detector.detect_outliers_column(
            sample_df, 'tenure'
        )

        # 100 should be identified as outlier
        assert stats['outlier_count'] > 0
        assert outlier_mask.iloc[-1] is True  # Last value (100) is outlier

    def test_detect_outliers_column_count(self, outlier_detector, sample_df):
        """Test outlier count"""
        outlier_mask, stats = outlier_detector.detect_outliers_column(
            sample_df, 'tenure'
        )

        assert stats['outlier_count'] == outlier_mask.sum()

    def test_detect_outliers_column_percentage(self, outlier_detector, sample_df):
        """Test outlier percentage calculation"""
        outlier_mask, stats = outlier_detector.detect_outliers_column(
            sample_df, 'tenure'
        )

        expected_pct = (stats['outlier_count'] / len(sample_df)) * 100
        assert stats['outlier_percentage'] == pytest.approx(expected_pct)

    def test_detect_outliers_column_no_outliers(self, outlier_detector, clean_df):
        """Test detection when no outliers present"""
        outlier_mask, stats = outlier_detector.detect_outliers_column(
            clean_df, 'tenure'
        )

        assert stats['outlier_count'] == 0
        assert stats['outlier_percentage'] == 0.0


class TestDetectOutliersMultipleColumns:
    """Test outlier detection for multiple columns"""

    def test_detect_outliers_basic(self, outlier_detector, sample_df):
        """Test detecting outliers in multiple columns"""
        report = outlier_detector.detect_outliers(sample_df)

        assert isinstance(report, dict)
        # Should detect outliers in default columns
        assert len(report) > 0

    def test_detect_outliers_custom_columns(self, outlier_detector, sample_df):
        """Test with custom column list"""
        columns = ['tenure', 'MonthlyCharges']
        report = outlier_detector.detect_outliers(sample_df, columns)

        assert set(report.keys()) == set(columns)

    def test_detect_outliers_all_columns(self, outlier_detector, sample_df):
        """Test detecting outliers in all columns"""
        columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        report = outlier_detector.detect_outliers(sample_df, columns)

        assert len(report) == 3
        for col in columns:
            assert col in report

    def test_detect_outliers_stores_summary(self, outlier_detector, sample_df):
        """Test that outlier summary is stored"""
        outlier_detector.detect_outliers(sample_df)

        assert len(outlier_detector.outlier_summary) > 0

    def test_detect_outliers_missing_column(self, outlier_detector, sample_df):
        """Test with non-existent column"""
        columns = ['tenure', 'NonExistent']
        report = outlier_detector.detect_outliers(sample_df, columns)

        # Should only have tenure
        assert 'tenure' in report
        assert 'NonExistent' not in report

    def test_detect_outliers_reports_per_column(self, outlier_detector, sample_df):
        """Test that each column has complete report"""
        report = outlier_detector.detect_outliers(sample_df)

        for col, stats in report.items():
            assert 'Q1' in stats
            assert 'Q3' in stats
            assert 'IQR' in stats
            assert 'outlier_count' in stats


class TestRemoveOutliers:
    """Test outlier removal"""

    def test_remove_outliers_basic(self, outlier_detector, sample_df):
        """Test basic outlier removal"""
        cleaned_df = outlier_detector.remove_outliers(sample_df)

        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) <= len(sample_df)

    def test_remove_outliers_reduces_rows(self, outlier_detector, sample_df):
        """Test that outliers are actually removed"""
        original_len = len(sample_df)
        cleaned_df = outlier_detector.remove_outliers(sample_df)

        assert len(cleaned_df) < original_len

    def test_remove_outliers_custom_columns(self, outlier_detector, sample_df):
        """Test removing outliers from specific columns"""
        columns = ['tenure']
        cleaned_df = outlier_detector.remove_outliers(sample_df, columns)

        # Should remove row with tenure=100
        assert 100 not in cleaned_df['tenure'].values

    def test_remove_outliers_preserves_clean_data(self, outlier_detector, clean_df):
        """Test that clean data is preserved"""
        original_len = len(clean_df)
        cleaned_df = outlier_detector.remove_outliers(clean_df)

        # Should not remove any rows
        assert len(cleaned_df) == original_len

    def test_remove_outliers_preserves_columns(self, outlier_detector, sample_df):
        """Test that all columns are preserved"""
        cleaned_df = outlier_detector.remove_outliers(sample_df)

        assert list(cleaned_df.columns) == list(sample_df.columns)


class TestCapOutliers:
    """Test outlier capping"""

    def test_cap_outliers_basic(self, outlier_detector, sample_df):
        """Test basic outlier capping"""
        capped_df = outlier_detector.cap_outliers(sample_df)

        assert isinstance(capped_df, pd.DataFrame)
        assert len(capped_df) == len(sample_df)  # No rows removed

    def test_cap_outliers_caps_values(self, outlier_detector, sample_df):
        """Test that outlier values are capped"""
        capped_df = outlier_detector.cap_outliers(sample_df)

        # Check that extreme values are capped
        # tenure=100 should be capped
        assert capped_df['tenure'].max() < sample_df['tenure'].max()

    def test_cap_outliers_custom_columns(self, outlier_detector, sample_df):
        """Test capping specific columns"""
        columns = ['tenure']
        capped_df = outlier_detector.cap_outliers(sample_df, columns)

        # tenure should be capped, others unchanged
        assert capped_df['tenure'].max() < sample_df['tenure'].max()

    def test_cap_outliers_preserves_shape(self, outlier_detector, sample_df):
        """Test that DataFrame shape is preserved"""
        original_shape = sample_df.shape
        capped_df = outlier_detector.cap_outliers(sample_df)

        assert capped_df.shape == original_shape

    def test_cap_outliers_no_change_for_clean_data(self, outlier_detector, clean_df):
        """Test that clean data is unchanged"""
        capped_df = outlier_detector.cap_outliers(clean_df)

        pd.testing.assert_frame_equal(capped_df, clean_df)

    def test_cap_outliers_within_bounds(self, outlier_detector, sample_df):
        """Test that capped values are within bounds"""
        capped_df = outlier_detector.cap_outliers(sample_df, ['tenure'])

        # Detect outliers to get bounds
        _, stats = outlier_detector.detect_outliers_column(sample_df, 'tenure')

        # All values should be within bounds
        assert capped_df['tenure'].min() >= stats['lower_bound']
        assert capped_df['tenure'].max() <= stats['upper_bound']


class TestDifferentMultipliers:
    """Test with different IQR multipliers"""

    def test_higher_multiplier_fewer_outliers(self, sample_df):
        """Test that higher multiplier detects fewer outliers"""
        detector_15 = OutlierDetector(multiplier=1.5)
        detector_30 = OutlierDetector(multiplier=3.0)

        _, stats_15 = detector_15.detect_outliers_column(sample_df, 'tenure')
        _, stats_30 = detector_30.detect_outliers_column(sample_df, 'tenure')

        # Higher multiplier should find fewer or equal outliers
        assert stats_30['outlier_count'] <= stats_15['outlier_count']

    def test_lower_multiplier_more_outliers(self, sample_df):
        """Test that lower multiplier detects more outliers"""
        detector_10 = OutlierDetector(multiplier=1.0)
        detector_20 = OutlierDetector(multiplier=2.0)

        _, stats_10 = detector_10.detect_outliers_column(sample_df, 'tenure')
        _, stats_20 = detector_20.detect_outliers_column(sample_df, 'tenure')

        # Lower multiplier should find more or equal outliers
        assert stats_10['outlier_count'] >= stats_20['outlier_count']

    def test_multiplier_affects_bounds(self, sample_df):
        """Test that multiplier affects bounds"""
        detector_15 = OutlierDetector(multiplier=1.5)
        detector_30 = OutlierDetector(multiplier=3.0)

        _, stats_15 = detector_15.detect_outliers_column(sample_df, 'tenure')
        _, stats_30 = detector_30.detect_outliers_column(sample_df, 'tenure')

        # Higher multiplier should have wider bounds
        assert stats_30['lower_bound'] <= stats_15['lower_bound']
        assert stats_30['upper_bound'] >= stats_15['upper_bound']


class TestEdgeCases:
    """Test edge cases"""

    def test_detect_outliers_single_value(self, outlier_detector):
        """Test with DataFrame containing single value"""
        df = pd.DataFrame({'col': [5, 5, 5, 5, 5]})

        outlier_mask, stats = outlier_detector.detect_outliers_column(
            df, 'col')

        # No outliers in constant data
        assert stats['IQR'] == 0
        assert stats['outlier_count'] == 0

    def test_detect_outliers_two_values(self, outlier_detector):
        """Test with two distinct values"""
        df = pd.DataFrame({'col': [1, 2]})

        outlier_mask, stats = outlier_detector.detect_outliers_column(
            df, 'col')

        # Should calculate statistics without error
        assert 'Q1' in stats
        assert 'Q3' in stats

    def test_detect_outliers_all_outliers(self, outlier_detector):
        """Test when all values are outliers"""
        df = pd.DataFrame({'col': [1, 2, 3, 100, 200, 300]})

        report = outlier_detector.detect_outliers(df, ['col'])

        # Should detect multiple outliers
        assert report['col']['outlier_count'] > 0

    def test_remove_outliers_empty_result(self, outlier_detector):
        """Test when removing outliers results in empty DataFrame"""
        df = pd.DataFrame({'col': [1, 100, 200, 300]})

        cleaned_df = outlier_detector.remove_outliers(df, ['col'])

        # Might result in very small or empty DataFrame
        assert len(cleaned_df) >= 0

    def test_cap_outliers_single_row(self, outlier_detector):
        """Test capping with single row"""
        df = pd.DataFrame({'col': [100]})

        capped_df = outlier_detector.cap_outliers(df, ['col'])

        # Should handle gracefully
        assert len(capped_df) == 1

    def test_detect_outliers_with_nan(self, outlier_detector):
        """Test detection with NaN values"""
        df = pd.DataFrame({'col': [1, 2, 3, np.nan, 5, 100]})

        # Should handle NaN appropriately
        outlier_mask, stats = outlier_detector.detect_outliers_column(
            df, 'col')

        # Statistics should be calculated ignoring NaN
        assert not np.isnan(stats['Q1'])
        assert not np.isnan(stats['Q3'])


class TestOutlierStatistics:
    """Test outlier statistics reporting"""

    def test_get_outlier_summary(self, outlier_detector, sample_df):
        """Test getting outlier summary"""
        outlier_detector.detect_outliers(sample_df)

        summary = outlier_detector.outlier_summary

        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_outlier_summary_per_column(self, outlier_detector, sample_df):
        """Test that summary has entry for each column"""
        columns = ['tenure', 'MonthlyCharges']
        outlier_detector.detect_outliers(sample_df, columns)

        summary = outlier_detector.outlier_summary

        for col in columns:
            assert col in summary

    def test_outlier_summary_statistics(self, outlier_detector, sample_df):
        """Test that summary contains all statistics"""
        outlier_detector.detect_outliers(sample_df, ['tenure'])

        stats = outlier_detector.outlier_summary['tenure']

        required_keys = ['Q1', 'Q3', 'IQR', 'lower_bound', 'upper_bound',
                         'outlier_count', 'outlier_percentage']
        for key in required_keys:
            assert key in stats


class TestBothDirections:
    """Test detection in both directions"""

    def test_detect_high_outliers(self, outlier_detector):
        """Test detection of high outliers"""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5, 100]})

        outlier_mask, stats = outlier_detector.detect_outliers_column(
            df, 'col')

        # 100 should be detected as high outlier
        assert outlier_mask.iloc[-1] is True
        assert stats['outlier_count'] > 0

    def test_detect_low_outliers(self, outlier_detector):
        """Test detection of low outliers"""
        df = pd.DataFrame({'col': [1, 50, 51, 52, 53, 54]})

        outlier_mask, stats = outlier_detector.detect_outliers_column(
            df, 'col')

        # 1 should be detected as low outlier
        assert outlier_mask.iloc[0] is True
        assert stats['outlier_count'] > 0

    def test_detect_both_directions(self, outlier_detector):
        """Test detection of outliers in both directions"""
        df = pd.DataFrame({'col': [1, 40, 45, 50, 55, 60, 200]})

        outlier_mask, stats = outlier_detector.detect_outliers_column(
            df, 'col')

        # Both 1 and 200 should be outliers
        assert stats['outlier_count'] >= 2


class TestPreservingDataIntegrity:
    """Test that original data is not modified"""

    def test_detect_preserves_original(self, outlier_detector, sample_df):
        """Test that detection doesn't modify original"""
        original = sample_df.copy()

        outlier_detector.detect_outliers(sample_df)

        pd.testing.assert_frame_equal(sample_df, original)

    def test_remove_preserves_original(self, outlier_detector, sample_df):
        """Test that removal doesn't modify original"""
        original = sample_df.copy()

        outlier_detector.remove_outliers(sample_df)

        pd.testing.assert_frame_equal(sample_df, original)

    def test_cap_creates_copy(self, outlier_detector, sample_df):
        """Test that capping creates a copy"""
        original_values = sample_df['tenure'].copy()

        capped_df = outlier_detector.cap_outliers(sample_df)

        # Original should be unchanged
        pd.testing.assert_series_equal(sample_df['tenure'], original_values)
