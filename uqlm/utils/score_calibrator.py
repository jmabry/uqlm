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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from uqlm.utils.score_evaluator import ScoreEvaluator


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
            # Reshape scores to 2D array for LogisticRegression
            scores_2d = scores.reshape(-1, 1)
            self.calibrator_ = LogisticRegression()
            self.calibrator_.fit(scores_2d, correct_labels)
        elif self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(scores, correct_labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit the calibrator directly on scores and labels
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

        if self.method == "platt":
            # LogisticRegression needs 2D input and returns probabilities for class 1
            scores_2d = scores.reshape(-1, 1)
            return self.calibrator_.predict_proba(scores_2d)[:, 1]
        elif self.method == "isotonic":
            # IsotonicRegression can handle 1D arrays
            return self.calibrator_.predict(scores)
        else:
            raise ValueError(f"Unknown method: {self.method}")

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


def fit_and_evaluate_calibrators(scores: Union[List[float], np.ndarray], correct_labels: Union[List[bool], List[int], np.ndarray], test_size: float = 0.2, random_state: Optional[int] = None, methods: Optional[List[str]] = None) -> tuple[dict, pd.DataFrame, dict]:
    """
    Fit and evaluate multiple calibration methods on the same data.

    Parameters
    ----------
    scores : array-like
        Training confidence scores.
    correct_labels : array-like
        Training binary correctness labels (True/False or 1/0).
    test_size : float, default=0.2
        Proportion of data to use for testing.
    random_state : int, optional
        Random state for train-test split.
    methods : list of str, optional
        List of calibration methods to compare. Default is ['platt', 'isotonic'].

    Returns
    -------
    calibrators : dict
        Dictionary containing fitted ScoreCalibrator instances for each method.
        Keys are method names ('platt', 'isotonic'), values are fitted calibrators.

    metrics_df : pd.DataFrame
        DataFrame comparing calibration metrics across methods with columns:

        - 'brier_score' : float
            Brier score (lower is better) - measures accuracy of probabilistic predictions
        - 'log_loss' : float
            Logarithmic loss (lower is better) - penalizes confident wrong predictions
        - 'ece' : float
            Expected Calibration Error - average difference between confidence and accuracy
        - 'mce' : float
            Maximum Calibration Error - worst-case calibration error across bins

        Index contains method names: 'platt', 'isotonic'.

    figures : dict
        Dictionary containing matplotlib figures:
        - 'transformation_comparison' : Figure showing original vs transformed scores
        - '{method}_calibration' : Figure showing calibration plots for each method

    See Also
    --------
    ScoreEvaluator.evaluate_calibration : Detailed metric definitions and computation
    """
    if methods is None:
        methods = ["platt", "isotonic"]
    results = []
    calibrated_scores_dict = {}
    figures = {}
    calibrators = {}

    # Convert to numpy arrays
    scores = np.array(scores)
    correct_labels = np.array(correct_labels, dtype=int)

    # Train-test split
    train_scores, test_scores, train_labels, test_labels = train_test_split(scores, correct_labels, test_size=test_size, random_state=random_state, stratify=correct_labels)

    # Create evaluator instance
    evaluator = ScoreEvaluator()
    transformation_data = {}
    for method in methods:
        calibrator = ScoreCalibrator(method=method)
        calibrator.fit(train_scores, train_labels)
        calibrators[method] = calibrator
        eval_scores = calibrator.transform(test_scores)
        calibrated_scores_dict[method] = eval_scores
        transformation_data[method] = eval_scores

        metrics = evaluator.evaluate_calibration(eval_scores, test_labels, plot=False, axes=None)
        metrics["method"] = method
        results.append(metrics)

    # Create figures
    # 1. Transformation comparison
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3.5))
    _plot_original_vs_transformed(test_scores, transformation_data, ax=axes)
    plt.close(fig)
    figures["transformation_comparison"] = fig

    # 2. Calibration plots for each method
    for method in methods:
        eval_scores = calibrated_scores_dict[method]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Calibration Plots for {method} Scores")
        evaluator.evaluate_calibration(eval_scores, test_labels, plot=True, axes=axes)
        plt.close(fig)
        figures[f"{method}_calibration"] = fig

    return calibrators, pd.DataFrame(results).set_index("method"), figures


def _plot_original_vs_transformed(original_scores: np.ndarray, transformed_data: Union[np.ndarray, dict], ax: plt.Axes = None, **kwargs):
    """
    Plot original vs transformed scores with a perfect calibration reference line.

    Parameters
    ----------
    original_scores : np.ndarray
        The original (uncalibrated) scores.
    transformed_data : dict
        multiple transformations with method names as keys.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    kwargs : dict
        Optional keyword arguments for customization.
    """
    title = kwargs.get("title", "Score Transformation Comparison")
    alpha = kwargs.get("alpha", 0.5)
    s = kwargs.get("s", 10)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Handle both single transformation and multiple transformations
    colors = {"platt": "blue", "isotonic": "orange"}
    for method, transformed_scores in transformed_data.items():
        color = colors.get(method, None)
        ax.scatter(original_scores, transformed_scores, alpha=alpha, s=s, label=f"{method.title()} Calibration", color=color)

    ax.set_xlabel("Original Scores")
    ax.set_ylabel("Transformed Scores")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
