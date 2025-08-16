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

"""
Score calibration module for uncertainty quantification confidence scores.

This module provides calibration methods to transform raw confidence scores
into better-calibrated probabilities using Platt Scaling and Isotonic Regression.
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional, List, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
from copy import deepcopy


class ScoreCalibrator:
    """
    A class for calibrating confidence scores using Platt Scaling or Isotonic Regression.

    Confidence scores from uncertainty quantification methods may not be well-calibrated
    probabilities. This class provides methods to transform raw scores into calibrated
    probabilities that better reflect the true likelihood of correctness.

    Parameters
    ----------
    method : {'platt', 'isotonic'}, default='platt'
        The calibration method to use:
        - 'platt': Platt scaling using logistic regression
        - 'isotonic': Isotonic regression (non-parametric, monotonic)

    Attributes
    ----------
    method : str
        The calibration method used.
    calibrator_ : sklearn estimator
        The fitted calibration model.
    is_fitted_ : bool
        Whether the calibrator has been fitted.
    """

    def __init__(self, method: Literal["platt", "isotonic"] = "platt"):
        self.method = method
        self.calibrator_ = None
        self.is_fitted_ = False

    def fit(self, scores: Union[List[float], np.ndarray], correct_labels: Union[List[bool], List[int], np.ndarray]) -> "ScoreCalibrator":
        """
        Fit the calibration model using scores and binary correctness labels.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Raw confidence scores to be calibrated (0-1 range expected).
        correct_labels : array-like of shape (n_samples,)
            Binary labels indicating correctness (True/False or 1/0).

        Returns
        -------
        self : ScoreCalibrator
            The fitted calibrator instance.
        """
        scores = np.array(scores)
        correct_labels = np.array(correct_labels, dtype=int)

        if len(scores) != len(correct_labels):
            raise ValueError("scores and correct_labels must have the same length")

        if not np.all(np.isin(correct_labels, [0, 1])):
            raise ValueError("correct_labels must be binary (True/False or 1/0)")
        if not np.all((scores >= 0) & (scores <= 1)):
            raise ValueError("scores must be between 0 and 1 inclusive")

        if self.method == "platt":
            from sklearn.calibration import _SigmoidCalibration

            self.calibrator_ = _SigmoidCalibration()
        elif self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit the calibrator directly on scores and labels
        self.calibrator_.fit(scores, correct_labels)
        self.is_fitted_ = True
        return self

    def transform(self, scores: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Transform raw scores into calibrated probabilities.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Raw confidence scores to be calibrated.

        Returns
        -------
        calibrated_scores : np.ndarray of shape (n_samples,)
            Calibrated probability scores.
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before transform")

        scores = np.array(scores)
        return self.calibrator_.predict(scores)

    def fit_transform(self, scores: Union[List[float], np.ndarray], correct_labels: Union[List[bool], List[int], np.ndarray]) -> np.ndarray:
        """
        Fit the calibrator and transform the scores in one step.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Raw confidence scores to be calibrated.
        correct_labels : array-like of shape (n_samples,)
            Binary labels indicating correctness (True/False or 1/0).

        Returns
        -------
        calibrated_scores : np.ndarray of shape (n_samples,)
            Calibrated probability scores.
        """
        return self.fit(scores, correct_labels).transform(scores)

    def evaluate_calibration(self, scores: Union[List[float], np.ndarray], correct_indicators: Union[List[int], np.ndarray], n_bins: int = 10, plot: bool = True) -> dict:
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

        Returns
        -------
        metrics : dict
            Dictionary containing calibration metrics:
            - 'brier_score': Brier score (lower is better)
            - 'log_loss': Log loss (lower is better)
            - 'ece': Expected Calibration Error
            - 'mce': Maximum Calibration Error
        """
        scores = np.array(scores)
        correct_indicators = np.array(correct_indicators)

        # Calculate Brier score and log loss
        brier = brier_score_loss(correct_indicators, scores)
        logloss = log_loss(correct_indicators, scores)

        # Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
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

        metrics = {"brier_score": brier, "log_loss": logloss, "ece": ece, "mce": mce}

        if plot:
            self._plot_reliability_diagram(bin_confidences, bin_accuracies, bin_counts, bin_boundaries)

        return metrics

    def _plot_reliability_diagram(self, bin_confidences: list, bin_accuracies: list, bin_counts: list, bin_boundaries: np.ndarray):
        """Plot reliability diagram for calibration assessment."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

        # Sample distribution
        ax2.bar(range(n_bins), bin_counts, alpha=0.7)
        ax2.set_xlabel("Confidence bin")
        ax2.set_ylabel("Number of samples")
        ax2.set_title("Sample Distribution")
        ax2.set_xticks(range(n_bins))
        ax2.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def compare_calibration_methods(scores: Union[List[float], np.ndarray], correct_labels: Union[List[bool], List[int], np.ndarray], test_scores: Optional[Union[List[float], np.ndarray]] = None, test_correct_labels: Optional[Union[List[bool], List[int], np.ndarray]] = None) -> pd.DataFrame:
    """
    Compare different calibration methods on the same data.

    Parameters
    ----------
    scores : array-like
        Training confidence scores.
    correct_labels : array-like
        Training binary correctness labels (True/False or 1/0).
    test_scores : array-like, optional
        Test confidence scores. If None, evaluates on training data.
    test_correct_labels : array-like, optional
        Test binary correctness labels.

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame comparing calibration metrics across methods.
    """
    methods = ["uncalibrated", "platt", "isotonic"]
    results = []

    # Convert to numpy arrays
    scores = np.array(scores)
    correct_labels = np.array(correct_labels, dtype=int)

    for method in methods:
        if method == "uncalibrated":
            if test_scores is not None:
                eval_scores = np.array(test_scores)
                eval_correct = np.array(test_correct_labels, dtype=int)
            else:
                eval_scores = scores
                eval_correct = correct_labels
        else:
            calibrator = ScoreCalibrator(method=method)
            calibrator.fit(scores, correct_labels)

            if test_scores is not None:
                eval_scores = calibrator.transform(test_scores)
                eval_correct = np.array(test_correct_labels, dtype=int)
            else:
                eval_scores = calibrator.transform(scores)
                eval_correct = correct_labels

        # Evaluate calibration
        temp_calibrator = ScoreCalibrator()
        metrics = temp_calibrator.evaluate_calibration(eval_scores, eval_correct, plot=False)
        metrics["method"] = method
        results.append(metrics)

    return pd.DataFrame(results).set_index("method")


def calibrate_uq_results(uq_results, correct_indicators: Union[List[int], np.ndarray], calibration_method: str = "platt", cv: int = 5, scorer_names: Optional[List[str]] = None) -> tuple:
    """
    Calibrate confidence scores from UQ results.

    Parameters
    ----------
    uq_results : UQResult
        Results from UQLM uncertainty quantification.
    correct_indicators : np.ndarray
        Binary indicators of response correctness.
    calibration_method : str, default='platt'
        Calibration method ('platt' or 'isotonic').
    cv : int, default=5
        Cross-validation folds for calibration.
    scorer_names : List[str], optional
        Specific scorers to calibrate. If None, calibrates all.

    Returns
    -------
    calibrated_results : UQResult
        UQ results with calibrated scores.
    calibrators : dict
        Dictionary of fitted calibrators for each scorer.
    """
    # Convert to DataFrame for easier manipulation
    df = uq_results.to_df()

    if scorer_names is None:
        # Get all confidence score columns (exclude metadata columns)
        score_columns = [col for col in df.columns if col not in ["prompt", "response", "log_probs"]]
    elif isinstance(scorer_names, str):
        score_columns = [scorer_names]
    else:
        score_columns = scorer_names

    calibrators = {}
    calibrated_results = deepcopy(uq_results)

    for scorer in score_columns:
        if scorer not in df.columns:
            print(f"Warning: Scorer '{scorer}' not found in results")
            continue

        # Fit calibrator
        calibrator = ScoreCalibrator(method=calibration_method)
        calibrator.fit(df[scorer], correct_indicators)

        # Transform scores
        calibrated_scores = calibrator.transform(df[scorer])

        # Update results with calibrated scores
        calibrated_results.confidence_scores[f"{scorer}_calibrated"] = calibrated_scores
        calibrators[scorer] = calibrator

    return calibrated_results, calibrators
