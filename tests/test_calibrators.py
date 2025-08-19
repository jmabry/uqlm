# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uqlm.utils.score_calibrator import ScoreCalibrator, fit_and_evaluate_calibrators, _plot_original_vs_transformed


class TestScoreCalibrator:
    """Test suite for ScoreCalibrator class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample calibration data for testing."""
        np.random.seed(42)
        n_samples = 100
        scores = np.random.beta(2, 2, n_samples)  # Scores between 0 and 1
        # Create labels with some correlation to scores
        correct_labels = np.random.binomial(1, scores * 0.7 + 0.15, n_samples)
        return scores, correct_labels

    @pytest.fixture
    def perfect_calibration_data(self):
        """Generate perfectly calibrated data for testing."""
        np.random.seed(123)
        n_samples = 100
        scores = np.random.uniform(0, 1, n_samples)
        correct_labels = np.random.binomial(1, scores, n_samples)
        return scores, correct_labels

    def test_init_default(self):
        """Test ScoreCalibrator initialization with default parameters."""
        calibrator = ScoreCalibrator()
        assert calibrator.method == "platt"
        assert calibrator.calibrator_ is None
        assert calibrator.is_fitted_ is False

    def test_init_invalid_method(self):
        """Test ScoreCalibrator initialization with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            calibrator = ScoreCalibrator(method="invalid")
            calibrator.fit([0.1, 0.9], [0, 1])

    def test_fit_and_transform_platt(self, sample_data):
        """Test fitting and transforming with Platt scaling method."""
        scores, labels = sample_data
        calibrator = ScoreCalibrator(method="platt")

        # Test fit
        result = calibrator.fit(scores, labels)
        assert result is calibrator  # Should return self
        assert calibrator.is_fitted_ is True
        assert calibrator.calibrator_ is not None
        assert hasattr(calibrator.calibrator_, "predict_proba")

        # Test transform
        transformed = calibrator.transform(scores)
        assert len(transformed) == len(scores)
        assert np.all((transformed >= 0) & (transformed <= 1))
        assert isinstance(transformed, np.ndarray)

    def test_fit_and_transform_isotonic(self, sample_data):
        """Test fitting and transforming with isotonic regression method."""
        scores, labels = sample_data
        calibrator = ScoreCalibrator(method="isotonic")

        # Test fit
        result = calibrator.fit(scores, labels)
        assert result is calibrator  # Should return self
        assert calibrator.is_fitted_ is True
        assert calibrator.calibrator_ is not None
        assert hasattr(calibrator.calibrator_, "predict")

        # Test transform
        transformed = calibrator.transform(scores)
        assert len(transformed) == len(scores)
        assert np.all((transformed >= 0) & (transformed <= 1))
        assert isinstance(transformed, np.ndarray)

    def test_fit_mismatched_lengths(self):
        """Test fit with mismatched input lengths."""
        calibrator = ScoreCalibrator()
        scores = [0.1, 0.5, 0.9]
        labels = [0, 1]  # Different length

        with pytest.raises(ValueError, match="scores and correct_labels must have the same length"):
            calibrator.fit(scores, labels)

    def test_fit_invalid_labels(self):
        """Test fit with invalid label values."""
        calibrator = ScoreCalibrator()
        scores = [0.1, 0.5, 0.9]
        labels = [0, 1, 2]  # Invalid label value

        with pytest.raises(ValueError, match="correct_labels must be binary"):
            calibrator.fit(scores, labels)

    def test_fit_invalid_scores(self):
        """Test fit with invalid score values."""
        calibrator = ScoreCalibrator()
        scores = [0.1, 0.5, 1.5]  # Score > 1
        labels = [0, 1, 1]

        with pytest.raises(ValueError, match="scores must be between 0 and 1 inclusive"):
            calibrator.fit(scores, labels)

    def test_fit_negative_scores(self):
        """Test fit with negative score values."""
        calibrator = ScoreCalibrator()
        scores = [-0.1, 0.5, 0.9]  # Negative score
        labels = [0, 1, 1]

        with pytest.raises(ValueError, match="scores must be between 0 and 1 inclusive"):
            calibrator.fit(scores, labels)

    def test_transform_not_fitted(self):
        """Test transform before fitting."""
        calibrator = ScoreCalibrator()
        scores = [0.1, 0.5, 0.9]

        with pytest.raises(ValueError, match="Calibrator must be fitted before transform"):
            calibrator.transform(scores)

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        scores, labels = sample_data
        calibrator = ScoreCalibrator()

        transformed = calibrator.fit_transform(scores, labels)

        assert calibrator.is_fitted_ is True
        assert len(transformed) == len(scores)
        assert np.all((transformed >= 0) & (transformed <= 1))

    def test_input_types(self, sample_data):
        """Test different input types (list, numpy array)."""
        scores, labels = sample_data
        calibrator = ScoreCalibrator()

        # Test with lists
        scores_list = scores.tolist()
        labels_list = labels.tolist()
        calibrator.fit(scores_list, labels_list)
        transformed = calibrator.transform(scores_list)

        assert len(transformed) == len(scores_list)
        assert isinstance(transformed, np.ndarray)

    def test_boolean_labels(self, sample_data):
        """Test with boolean labels."""
        scores, labels = sample_data
        bool_labels = labels.astype(bool)
        calibrator = ScoreCalibrator()

        calibrator.fit(scores, bool_labels)
        transformed = calibrator.transform(scores)

        assert len(transformed) == len(scores)
        assert calibrator.is_fitted_ is True

    def test_edge_case_single_sample(self):
        """Test with single sample - should raise error for insufficient data."""
        calibrator = ScoreCalibrator()
        scores = [0.5]
        labels = [1]

        # Single sample with only one class should raise an error
        with pytest.raises(ValueError):
            calibrator.fit(scores, labels)

    def test_minimal_valid_dataset(self):
        """Test with minimal valid dataset (2 samples, 2 classes)."""
        calibrator = ScoreCalibrator()
        scores = [0.3, 0.7]
        labels = [0, 1]

        calibrator.fit(scores, labels)
        transformed = calibrator.transform(scores)

        assert len(transformed) == 2
        assert np.all((transformed >= 0) & (transformed <= 1))


class TestFitAndEvaluateCalibrators:
    """Test suite for fit_and_evaluate_calibrators function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample calibration data for testing."""
        np.random.seed(42)
        n_samples = 200
        scores = np.random.beta(2, 2, n_samples)
        correct_labels = np.random.binomial(1, scores * 0.7 + 0.15, n_samples)
        return scores, correct_labels

    def test_default_parameters(self, sample_data):
        """Test with default parameters."""
        scores, labels = sample_data

        calibrators, metrics_df, figures = fit_and_evaluate_calibrators(scores, labels)

        # Check calibrators
        assert isinstance(calibrators, dict)
        assert "platt" in calibrators
        assert "isotonic" in calibrators
        assert all(cal.is_fitted_ for cal in calibrators.values())

        # Check metrics
        assert isinstance(metrics_df, pd.DataFrame)
        assert len(metrics_df) == 2  # platt and isotonic
        assert "brier_score" in metrics_df.columns
        assert "ece" in metrics_df.columns
        assert "mce" in metrics_df.columns

        # Check figures
        assert isinstance(figures, dict)
        assert "transformation_comparison" in figures
        assert "platt_calibration" in figures
        assert "isotonic_calibration" in figures

    def test_custom_random_state(self, sample_data):
        """Test with custom random state."""
        scores, labels = sample_data

        # Run twice with same random state
        result1 = fit_and_evaluate_calibrators(scores, labels, random_state=42)
        result2 = fit_and_evaluate_calibrators(scores, labels, random_state=42)

        # Results should be identical
        pd.testing.assert_frame_equal(result1[1], result2[1])

    def test_single_method(self, sample_data):
        """Test with single calibration method."""
        scores, labels = sample_data

        calibrators, metrics_df, figures = fit_and_evaluate_calibrators(scores, labels, methods=["platt"])

        assert len(calibrators) == 1
        assert "platt" in calibrators
        assert len(metrics_df) == 1
        assert "platt_calibration" in figures
        assert "isotonic_calibration" not in figures

    def test_invalid_method(self, sample_data):
        """Test with invalid method."""
        scores, labels = sample_data

        with pytest.raises(ValueError):
            fit_and_evaluate_calibrators(scores, labels, methods=["invalid"])

    def test_small_dataset(self):
        """Test with very small dataset."""
        scores = [0.1, 0.9, 0.3, 0.7]
        labels = [0, 1, 0, 1]

        calibrators, metrics_df, _ = fit_and_evaluate_calibrators(scores, labels, test_size=0.5)

        assert len(calibrators) == 2
        assert len(metrics_df) == 2


class TestPlotOriginalVsTransformed:
    """Test suite for _plot_original_vs_transformed function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for plotting tests."""
        np.random.seed(42)
        original = np.random.uniform(0, 1, 50)
        transformed_platt = original * 0.8 + 0.1  # Simple transformation
        transformed_isotonic = np.clip(original * 1.2 - 0.1, 0, 1)
        return original, {"platt": transformed_platt, "isotonic": transformed_isotonic}

    def test_plot_with_default_params(self, sample_data):
        """Test plotting with default parameters."""
        original, transformed = sample_data

        fig, ax = plt.subplots()
        _plot_original_vs_transformed(original, transformed, ax=ax)

        assert len(ax.collections) == 2  # Two scatter plots
        assert ax.get_xlabel() == "Original Scores"
        assert ax.get_ylabel() == "Transformed Scores"
        assert ax.legend_ is not None

        plt.close(fig)

    def test_plot_with_custom_kwargs(self, sample_data):
        """Test plotting with custom keyword arguments."""
        original, transformed = sample_data

        fig, ax = plt.subplots()
        _plot_original_vs_transformed(original, transformed, ax=ax, title="Custom Title", alpha=0.8, s=20)

        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_plot_single_method(self, sample_data):
        """Test plotting with single transformation method."""
        original, transformed = sample_data

        fig, ax = plt.subplots()
        single_transformed = {"platt": transformed["platt"]}
        _plot_original_vs_transformed(original, single_transformed, ax=ax)

        assert len(ax.collections) == 1  # One scatter plot
        plt.close(fig)


class TestIntegration:
    """Integration tests for the calibration module."""

    def test_end_to_end_workflow(self):
        """Test complete calibration workflow."""
        # Generate realistic overconfident data
        np.random.seed(42)
        n_samples = 500
        raw_scores = np.random.beta(0.5, 0.5, n_samples)
        raw_scores = 0.3 + 0.7 * raw_scores  # Shift to be overconfident
        correct_labels = np.random.binomial(1, 0.2 + 0.4 * raw_scores, n_samples)

        # Fit calibrators
        calibrators, metrics_df, figures = fit_and_evaluate_calibrators(raw_scores, correct_labels, test_size=0.4, random_state=42)

        # Check that calibration improves metrics
        # Both methods should have reasonable performance
        assert len(calibrators) == 2
        assert all(0 <= row["brier_score"] <= 1 for _, row in metrics_df.iterrows())
        assert all(0 <= row["ece"] <= 1 for _, row in metrics_df.iterrows())

        # Test individual calibrator usage
        platt_calibrator = calibrators["platt"]
        new_scores = [0.1, 0.5, 0.9]
        calibrated = platt_calibrator.transform(new_scores)

        assert len(calibrated) == 3
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_perfect_calibration_data(self):
        """Test with perfectly calibrated data."""
        np.random.seed(123)
        n_samples = 300
        scores = np.random.uniform(0, 1, n_samples)
        correct_labels = np.random.binomial(1, scores, n_samples)

        calibrators, metrics_df, figures = fit_and_evaluate_calibrators(scores, correct_labels, random_state=42)

        # With perfect calibration, ECE should be relatively low
        assert all(row["ece"] < 0.2 for _, row in metrics_df.iterrows())

    def test_extreme_overconfidence(self):
        """Test with extremely overconfident scores."""
        n_samples = 200
        scores = np.full(n_samples, 0.95)  # Very high confidence
        correct_labels = np.random.binomial(1, 0.3, n_samples)  # Low accuracy

        calibrators, metrics_df, figures = fit_and_evaluate_calibrators(scores, correct_labels, random_state=42)

        # Calibration should help reduce overconfidence
        platt_calibrator = calibrators["platt"]
        calibrated_scores = platt_calibrator.transform(scores)

        # Calibrated scores should be lower than original for this case
        assert np.mean(calibrated_scores) < np.mean(scores)

    def test_isotonic_monotonicity(self):
        """Test that isotonic regression produces monotonic transformation."""
        np.random.seed(42)
        scores = np.linspace(0, 1, 100)
        # Create labels with some noise but generally increasing with scores
        correct_labels = np.random.binomial(1, np.clip(scores + np.random.normal(0, 0.1, 100), 0, 1))

        calibrator = ScoreCalibrator(method="isotonic")
        calibrator.fit(scores, correct_labels)

        # Test on sorted scores
        test_scores = np.linspace(0, 1, 50)
        calibrated = calibrator.transform(test_scores)

        # Should be monotonic (non-decreasing)
        assert np.all(np.diff(calibrated) >= -1e-10)  # Allow for small numerical errors
