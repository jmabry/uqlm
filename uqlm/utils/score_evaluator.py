from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


class ScoreEvaluator:
    """
    A class for evaluating the calibration quality of confidence scores.

    This class provides methods to assess how well confidence scores reflect
    true probabilities using reliability diagrams and calibration metrics.
    """

    @staticmethod
    def evaluate_calibration(scores: Union[List[float], np.ndarray], correct_indicators: "Union[List[int], np.ndarray]", n_bins: int = 10, plot: bool = True, axes: "Union[tuple, None]" = None) -> dict:
        """
        Evaluate the calibration quality of scores.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,)
            Confidence scores (raw or calibrated).
        correct_indicators : np.ndarray of shape (n_samples,)
            Binary indicators (0/1) of whether each response was correct.
        n_bins : int, default=10
            Number of bins for reliability diagram.
        plot : bool, default=True
            Whether to plot the reliability diagram.
        axes : tuple of matplotlib.axes.Axes, optional
            Tuple of (reliability_ax, distribution_ax) for plotting.
            If None and plot=True, creates new figure.

        Returns
        -------
        metrics : dict
            Dictionary containing calibration metrics:
            - 'average_confidence': Mean of confidence scores
            - 'average_accuracy': Mean of correct indicators
            - 'calibration_gap': Absolute difference between average confidence and accuracy
            - 'brier_score': Brier score (lower is better)
            - 'log_loss': Log loss (lower is better)
            - 'ece': Expected Calibration Error
            - 'mce': Maximum Calibration Error
        """
        scores = np.array(scores)
        correct_indicators = np.array(correct_indicators)
        avg_confidence = scores.mean()
        avg_accuracy = correct_indicators.mean()
        calibration_gap = abs(avg_confidence - avg_accuracy)

        # Store metrics for table display
        metrics = {"average_confidence": avg_confidence, "average_accuracy": avg_accuracy, "calibration_gap": calibration_gap}

        # Calculate Brier score and log loss
        brier = brier_score_loss(correct_indicators, scores)

        # Handle edge case where all labels are the same (log_loss requires at least 2 classes)
        if len(np.unique(correct_indicators)) == 1:
            # For single class, log loss is not well-defined, so we calculate it manually
            # Log loss = -1/N * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
            eps = 1e-15  # Small epsilon to avoid log(0)
            scores_clipped = np.clip(scores, eps, 1 - eps)
            if correct_indicators[0] == 1:  # All correct
                logloss = -np.mean(np.log(scores_clipped))
            else:  # All incorrect
                logloss = -np.mean(np.log(1 - scores_clipped))
        else:
            logloss = log_loss(correct_indicators, scores)

        # Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)

        # Create bins enclosing the range [0, 1] of confidence scores
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_boundaries[0] = 0 - np.finfo(float).eps  # Ensure scores of exactly 0 are included
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        mce = 0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (scores > bin_lower) & (scores <= bin_upper)
            prob_in_bin = in_bin.mean()

            if prob_in_bin > 0:
                accuracy_in_bin = correct_indicators[in_bin].mean()
                avg_confidence_in_bin = scores[in_bin].mean()

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)

        metrics.update({"brier_score": brier, "log_loss": logloss, "ece": ece, "mce": mce})

        if plot:
            ScoreEvaluator._plot_reliability_diagram(bin_boundaries, bin_counts, bin_accuracies, axes=axes)

        return metrics

    @staticmethod
    def _plot_reliability_diagram(bin_boundaries: np.ndarray, bin_counts: list, bin_accuracies: list, axes: tuple = None):
        """Plot reliability diagram for calibration assessment."""
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(12, 5))
            show_plot = True
        else:
            show_plot = False

        ax1, ax2 = axes

        n_bins = len(bin_boundaries) - 1

        # Create bin boundary labels
        bin_labels = [f"({bin_boundaries[i]:.1f}, {bin_boundaries[i + 1]:.1f}]" for i in range(n_bins)]

        # Calculate bin midpoints for perfect calibration line
        bin_midpoints = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]

        # Reliability diagram
        # Perfect calibration line: where confidence = accuracy for each bin
        ax1.plot(range(n_bins), bin_midpoints, "k--", label="Perfect calibration")
        ax1.bar(range(n_bins), bin_accuracies, alpha=0.7, label="Actual accuracy", width=0.8)
        ax1.set_xlabel("Confidence bin")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Reliability Diagram")
        ax1.set_xticks(range(n_bins))
        ax1.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Score distribution
        ax2.bar(range(n_bins), bin_counts, alpha=0.7)
        ax2.set_xlabel("Confidence bin")
        ax2.set_ylabel("Number of samples")
        ax2.set_title("Score Distribution")
        ax2.set_xticks(range(n_bins))
        ax2.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        if show_plot:
            plt.tight_layout()
            plt.show()
