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
from unittest.mock import patch, MagicMock

from uqlm.utils.score_evaluator import ScoreEvaluator


class TestScoreEvaluator:
    """Test suite for ScoreEvaluator class."""

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
        n_samples = 1000  # Larger sample for more stable perfect calibration
        scores = np.random.uniform(0, 1, n_samples)
        correct_labels = np.random.binomial(1, scores, n_samples)
        return scores, correct_labels

    @pytest.fixture
    def overconfident_data(self):
        """Generate overconfident scores (high confidence, low accuracy)."""
        scores = np.array([0.9, 0.95, 0.8, 0.85, 0.9, 0.92, 0.88, 0.93, 0.87, 0.91])
        correct_labels = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])  # 30% accuracy
        return scores, correct_labels

    @pytest.fixture
    def underconfident_data(self):
        """Generate underconfident scores (low confidence, high accuracy)."""
        scores = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.12, 0.18, 0.22, 0.28, 0.16])
        correct_labels = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1])  # 90% accuracy
        return scores, correct_labels

    def test_evaluate_calibration_basic_metrics(self, sample_data):
        """Test basic calibration metrics calculation."""
        scores, correct_labels = sample_data

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        # Check that all expected metrics are present
        expected_keys = {"average_confidence", "average_accuracy", "calibration_gap", "brier_score", "log_loss", "ece", "mce"}
        assert set(metrics.keys()) == expected_keys

        # Check metric ranges
        assert 0 <= metrics["average_confidence"] <= 1
        assert 0 <= metrics["average_accuracy"] <= 1
        assert metrics["calibration_gap"] >= 0
        assert metrics["brier_score"] >= 0
        assert metrics["log_loss"] >= 0
        assert 0 <= metrics["ece"] <= 1
        assert 0 <= metrics["mce"] <= 1

    def test_evaluate_calibration_perfect_case(self, perfect_calibration_data):
        """Test calibration metrics with perfectly calibrated data."""
        scores, correct_labels = perfect_calibration_data

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        # Perfect calibration should have small calibration gap and ECE
        # Note: Due to randomness, we allow some tolerance
        assert metrics["calibration_gap"] < 0.05
        assert metrics["ece"] < 0.05

    def test_evaluate_calibration_overconfident(self, overconfident_data):
        """Test calibration metrics with overconfident scores."""
        scores, correct_labels = overconfident_data

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        # Overconfident data should have high calibration gap
        assert metrics["average_confidence"] > metrics["average_accuracy"]
        assert metrics["calibration_gap"] > 0.5  # Should be significantly overconfident

    def test_evaluate_calibration_underconfident(self, underconfident_data):
        """Test calibration metrics with underconfident scores."""
        scores, correct_labels = underconfident_data

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        # Underconfident data should have high calibration gap
        assert metrics["average_confidence"] < metrics["average_accuracy"]
        assert metrics["calibration_gap"] > 0.5  # Should be significantly underconfident

    def test_input_type_validation(self):
        """Test that function handles various input types correctly."""
        # Test with different numeric types: lists, tuples, arrays
        scores_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        correct_list = [0, 0, 1, 1, 1]
        scores_tuple = (0.1, 0.5, 0.9)
        correct_tuple = (0, 1, 1)
        scores_array = np.array([0.2, 0.6, 0.8])
        correct_array = np.array([0, 1, 1])

        # All should work without error
        for scores, correct_labels in [(scores_list, correct_list), (scores_tuple, correct_tuple), (scores_array, correct_array)]:
            metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)
            assert isinstance(metrics, dict)
            assert 0 <= metrics["average_confidence"] <= 1

    def test_single_class_scenarios(self):
        """Test edge cases with single class data and extreme values."""
        # All correct, high confidence
        scores = np.array([0.9, 0.95, 0.99])
        correct_labels = np.array([1, 1, 1])

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        assert metrics["average_accuracy"] == 1.0
        assert metrics["calibration_gap"] >= 0
        assert np.isfinite(metrics["log_loss"])
        assert metrics["log_loss"] > 0

        # All incorrect, low confidence
        scores = np.array([0.1, 0.05, 0.01])
        correct_labels = np.array([0, 0, 0])

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        assert metrics["average_accuracy"] == 0.0
        assert np.isfinite(metrics["log_loss"])
        assert metrics["log_loss"] > 0

    def test_evaluate_calibration_single_sample(self):
        """Test with single sample."""
        scores = np.array([0.7])
        correct_labels = np.array([1])

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        assert metrics["average_confidence"] == 0.7
        assert metrics["average_accuracy"] == 1.0
        assert abs(metrics["calibration_gap"] - 0.3) < 1e-10  # Use tolerance for floating point
        # Log loss should be finite for single sample case
        assert np.isfinite(metrics["log_loss"])

    def test_evaluate_calibration_boundary_scores(self):
        """Test with boundary confidence scores (0 and 1)."""
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        correct_labels = np.array([0, 1, 0, 1])

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        # Should handle boundary cases without error
        assert isinstance(metrics, dict)
        assert 0 <= metrics["ece"] <= 1

    def test_evaluate_calibration_different_bin_counts(self, sample_data):
        """Test with different numbers of bins."""
        scores, correct_labels = sample_data

        for n_bins in [5, 10, 20]:
            metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, n_bins=n_bins, plot=False)

            # Should work with different bin counts
            assert isinstance(metrics, dict)
            assert 0 <= metrics["ece"] <= 1

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    def test_plotting_functionality(self, mock_show, mock_tight_layout, mock_subplots, sample_data):
        """Test plotting functionality with automatic and custom axes."""
        scores, correct_labels = sample_data

        # Test automatic plotting
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])

        _ = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=True)

        # Verify plotting functions were called
        mock_subplots.assert_called_once_with(1, 2, figsize=(12, 5))
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()
        assert mock_ax1.plot.called
        assert mock_ax1.bar.called
        assert mock_ax2.bar.called

        # Reset mocks for custom axes test
        mock_show.reset_mock()
        mock_subplots.reset_mock()
        mock_tight_layout.reset_mock()

        # Test custom axes
        custom_ax1 = MagicMock()
        custom_ax2 = MagicMock()
        custom_axes = (custom_ax1, custom_ax2)

        _ = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=True, axes=custom_axes)

        # When custom axes provided, show() should not be called
        mock_show.assert_not_called()
        mock_subplots.assert_not_called()
        mock_tight_layout.assert_not_called()

        # Verify custom axes were used
        assert custom_ax1.plot.called
        assert custom_ax1.bar.called
        assert custom_ax2.bar.called

    def test_ece_calculation_manual(self):
        """Test ECE calculation with manually verified data."""
        # Create simple data where we can manually calculate ECE
        scores = np.array([0.1, 0.1, 0.9, 0.9])  # Two bins: [0.0-0.5], [0.5-1.0]
        correct_labels = np.array([0, 1, 0, 1])  # 50% accuracy in each bin

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, n_bins=2, plot=False)

        # Manual ECE calculation:
        # Bin 1 [0.0-0.5]: avg_conf=0.1, accuracy=0.5, weight=0.5, contrib=|0.1-0.5|*0.5=0.2
        # Bin 2 [0.5-1.0]: avg_conf=0.9, accuracy=0.5, weight=0.5, contrib=|0.9-0.5|*0.5=0.2
        # ECE = 0.2 + 0.2 = 0.4
        expected_ece = 0.4

        assert abs(metrics["ece"] - expected_ece) < 0.01

    def test_mce_calculation_manual(self):
        """Test MCE calculation with manually verified data."""
        # Create data where one bin has much worse calibration than others
        scores = np.array([0.1, 0.1, 0.9, 0.9])
        correct_labels = np.array([0, 0, 0, 0])  # All wrong

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, n_bins=2, plot=False)

        # MCE should be the maximum calibration error across bins
        # Bin 2 [0.5-1.0]: |0.9 - 0.0| = 0.9 (worst bin)
        # Bin 1 [0.0-0.5]: |0.1 - 0.0| = 0.1
        expected_mce = 0.9

        assert abs(metrics["mce"] - expected_mce) < 0.01
        # Log loss should be finite even with all incorrect labels
        assert np.isfinite(metrics["log_loss"])

    def test_empty_bins_handling(self):
        """Test handling of empty bins in ECE/MCE calculation."""
        # Create scores that don't populate all bins
        scores = np.array([0.1, 0.2, 0.8, 0.9])  # No scores in middle bins
        correct_labels = np.array([0, 1, 0, 1])

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, n_bins=10, plot=False)

        # Should handle empty bins without error
        assert isinstance(metrics["ece"], float)
        assert isinstance(metrics["mce"], float)
        assert 0 <= metrics["ece"] <= 1
        assert 0 <= metrics["mce"] <= 1

    def test_plot_reliability_diagram_methods(self):
        """Test the _plot_reliability_diagram method with and without axes."""
        bin_boundaries = np.array([0, 0.5, 1.0])
        bin_counts = [50, 50]
        bin_accuracies = [0.3, 0.7]

        # Test with provided axes
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        axes = (mock_ax1, mock_ax2)

        ScoreEvaluator._plot_reliability_diagram(bin_boundaries, bin_counts, bin_accuracies, axes=axes)

        # Verify plotting methods were called
        mock_ax1.plot.assert_called()
        mock_ax1.bar.assert_called()
        mock_ax2.bar.assert_called()

        # Verify labels and titles were set
        mock_ax1.set_xlabel.assert_called()
        mock_ax1.set_ylabel.assert_called()
        mock_ax1.set_title.assert_called()
        mock_ax2.set_xlabel.assert_called()
        mock_ax2.set_ylabel.assert_called()
        mock_ax2.set_title.assert_called()

        # Test without axes (creates new figure)
        with patch("matplotlib.pyplot.subplots") as mock_subplots, patch("matplotlib.pyplot.tight_layout") as mock_tight_layout, patch("matplotlib.pyplot.show") as mock_show:
            mock_fig = MagicMock()
            mock_ax1_new = MagicMock()
            mock_ax2_new = MagicMock()
            mock_subplots.return_value = (mock_fig, [mock_ax1_new, mock_ax2_new])

            ScoreEvaluator._plot_reliability_diagram(bin_boundaries, bin_counts, bin_accuracies, axes=None)

            # Verify new figure was created and shown
            mock_subplots.assert_called_once_with(1, 2, figsize=(12, 5))
            mock_tight_layout.assert_called_once()
            mock_show.assert_called_once()

    def test_edge_cases_and_precision(self):
        """Test numeric precision and edge cases with boundary values."""
        # Test with very small differences
        scores = np.array([0.5000001, 0.5000002, 0.4999999])
        correct_labels = np.array([1, 0, 1])

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, plot=False)

        # Should handle small numeric differences without error
        assert isinstance(metrics["calibration_gap"], float)
        assert not np.isnan(metrics["calibration_gap"])

        # Test scores exactly at bin boundaries
        scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        correct_labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1])

        metrics = ScoreEvaluator.evaluate_calibration(scores, correct_labels, n_bins=10, plot=False)

        # Should handle boundary scores correctly
        assert isinstance(metrics["ece"], float)
        assert not np.isnan(metrics["ece"])
        assert not np.isinf(metrics["ece"])
