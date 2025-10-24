"""Model validation metrics and evaluation functions.

This module provides standalone functions for model validation using Polars DataFrames.
All functions accept a DataFrame as the first argument.
"""

import polars as pl
import numpy as np
from typing import Optional, Union, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import (
    auc, roc_curve, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix,
    precision_recall_curve
)


@dataclass
class BootstrapResult:
    """Container for bootstrap results."""
    estimate: float
    ci_lower: float
    ci_upper: float
    bootstrap_samples: np.ndarray
    
    def to_polars(self) -> pl.DataFrame:
        """Convert bootstrap results to a Polars DataFrame.
        
        Returns:
            Single-row DataFrame with estimate and confidence intervals.
            Bootstrap samples are not included in the DataFrame.
        """
        return pl.DataFrame({
            'estimate': [self.estimate],
            'ci_lower': [self.ci_lower],
            'ci_upper': [self.ci_upper],
            'n_samples': [len(self.bootstrap_samples)]
        })


@dataclass
class LiftResult:
    """Container for lift curve results."""
    percentiles: np.ndarray
    score_lift_values: np.ndarray
    target_lift_values: np.ndarray
    score_cumulative_lift: np.ndarray
    target_cumulative_lift: np.ndarray
    baseline: float
    auc_score_lift: float
    auc_target_lift: float
    
    def to_polars(self) -> pl.DataFrame:
        """Convert lift results to a Polars DataFrame.
        
        Returns:
            DataFrame with one row per percentile containing lift values.
        """
        return pl.DataFrame({
            'percentile': self.percentiles,
            'score_lift': self.score_lift_values,
            'target_lift': self.target_lift_values,
            'score_cumulative_lift': self.score_cumulative_lift,
            'target_cumulative_lift': self.target_cumulative_lift
        }).with_columns([
            pl.lit(self.baseline).alias('baseline'),
            pl.lit(self.auc_score_lift).alias('auc_score_lift'),
            pl.lit(self.auc_target_lift).alias('auc_target_lift')
        ])


@dataclass
class ROCResult:
    """Container for ROC curve results."""
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc_score: float
    optimal_threshold: float
    optimal_point: Tuple[float, float]
    
    def to_polars(self) -> pl.DataFrame:
        """Convert ROC results to a Polars DataFrame.
        
        Returns:
            DataFrame with one row per threshold containing FPR and TPR values.
        """
        return pl.DataFrame({
            'threshold': self.thresholds,
            'fpr': self.fpr,
            'tpr': self.tpr
        }).with_columns([
            pl.lit(self.auc_score).alias('auc_score'),
            pl.lit(self.optimal_threshold).alias('optimal_threshold'),
            pl.lit(self.optimal_point[0]).alias('optimal_fpr'),
            pl.lit(self.optimal_point[1]).alias('optimal_tpr')
        ])


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    adj_r2: float
    n_samples: int
    n_features: Optional[int] = None
    
    def to_polars(self) -> pl.DataFrame:
        """Convert regression metrics to a Polars DataFrame.
        
        Returns:
            Single-row DataFrame with all regression metrics.
        """
        return pl.DataFrame({
            'mse': [self.mse],
            'rmse': [self.rmse],
            'mae': [self.mae],
            'mape': [self.mape],
            'r2': [self.r2],
            'adj_r2': [self.adj_r2],
            'n_samples': [self.n_samples],
            'n_features': [self.n_features]
        })


@dataclass
class DoubleLiftResult:
    """Container for double lift analysis results."""
    lift1: LiftResult
    lift2: LiftResult
    correlation: float
    joint_lift: float
    conditional_lift: float
    
    def to_polars(self) -> pl.DataFrame:
        """Convert double lift results to a Polars DataFrame.
        
        Returns:
            DataFrame with lift curves from both models and comparison metrics.
        """
        # Get lift DataFrames
        df1 = self.lift1.to_polars().select([
            'percentile',
            pl.col('score_lift').alias('model1_score_lift'),
            pl.col('target_lift').alias('model1_target_lift'),
            pl.col('score_cumulative_lift').alias('model1_cumulative_lift')
        ])
        
        df2 = self.lift2.to_polars().select([
            pl.col('score_lift').alias('model2_score_lift'),
            pl.col('target_lift').alias('model2_target_lift'),
            pl.col('score_cumulative_lift').alias('model2_cumulative_lift')
        ])
        
        # Combine and add comparison metrics
        return pl.concat([df1, df2], how='horizontal').with_columns([
            pl.lit(self.correlation).alias('correlation'),
            pl.lit(self.joint_lift).alias('joint_lift'),
            pl.lit(self.conditional_lift).alias('conditional_lift')
        ])


@dataclass
class BinaryClassificationMetrics:
    """Container for binary classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    specificity: float
    npv: float
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    
    def to_polars(self) -> pl.DataFrame:
        """Convert classification metrics to a Polars DataFrame.
        
        Returns:
            Single-row DataFrame with all classification metrics.
            Confusion matrix is flattened into separate columns.
        """
        return pl.DataFrame({
            'accuracy': [self.accuracy],
            'precision': [self.precision],
            'recall': [self.recall],
            'f1_score': [self.f1_score],
            'true_positives': [self.true_positives],
            'false_positives': [self.false_positives],
            'true_negatives': [self.true_negatives],
            'false_negatives': [self.false_negatives],
            'specificity': [self.specificity],
            'npv': [self.npv],
            'pr_auc': [self.pr_auc],
            'roc_auc': [self.roc_auc]
        })

# ============================================================================
# Utility Functions
# ============================================================================

def get_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """Return information about missing values.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        pl.DataFrame: Missing value information
    """
    total_rows = df.height
    null_counts = df.null_count()
    
    return pl.DataFrame({
        'column': df.columns,
        'missing_count': null_counts,
        'missing_percentage': (null_counts / total_rows * 100)
    })


# ============================================================================
# Lorenz Curve and Gini Coefficient Functions
# ============================================================================

def calculate_lorenz_curve(df: pl.DataFrame, column: str, 
                          exposure_column: Optional[str] = None,
                          observed_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculate the Lorenz curve coordinates and Gini coefficient.
    
    Args:
        df: Polars DataFrame
        column: Name of the numeric column to order by (e.g., predicted values)
        exposure_column: Optional name of the column containing exposure/weight values
        observed_column: Optional name of the column to calculate Lorenz curve for.
                        If provided, data is ordered by 'column' but Lorenz curve
                        is calculated using 'observed_column' values.
        
    Returns:
        Tuple containing (x_coords, y_coords, gini_coefficient)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    if observed_column is not None and observed_column not in df.columns:
        raise ValueError(f"Observed column '{observed_column}' not found in data")
        
    try:
        if exposure_column is not None:
            if exposure_column not in df.columns:
                raise ValueError(f"Exposure column '{exposure_column}' not found in data")
            
            # Select columns based on whether observed_column is provided
            select_cols = [
                pl.col(column).alias('sort_value'),
                pl.col(exposure_column).alias('exposure')
            ]
            if observed_column is not None:
                select_cols.append(pl.col(observed_column).alias('value'))
            else:
                select_cols.append(pl.col(column).alias('value'))
                
            data_df = df.select(select_cols).drop_nulls()
            
            sort_values = data_df.select('sort_value').to_numpy().flatten()
            values = data_df.select('value').to_numpy().flatten()
            exposures = data_df.select('exposure').to_numpy().flatten()
            
            if not np.all(exposures >= 0):
                raise ValueError("Exposure values must be non-negative")
            
            ratio = sort_values / (exposures + np.finfo(float).eps)
            sort_idx = np.argsort(ratio)
            values = values[sort_idx]
            exposures = exposures[sort_idx]
            
            cum_values = np.cumsum(values)
            cum_exposures = np.cumsum(exposures)
            
            if cum_values[-1] == 0 or cum_exposures[-1] == 0:
                return np.linspace(0, 1, 100), np.linspace(0, 1, 100), 0.0
            
            x = np.insert(cum_exposures / cum_exposures[-1], 0, 0)
            y = np.insert(cum_values / cum_values[-1], 0, 0)
            
        else:
            # Select columns based on whether observed_column is provided
            if observed_column is not None:
                data_df = df.select([
                    pl.col(column).alias('sort_value'),
                    pl.col(observed_column).alias('value')
                ]).drop_nulls()
                
                sort_values = data_df.select('sort_value').to_numpy().flatten()
                values = data_df.select('value').to_numpy().flatten()
                
                # Sort values by sort_value (predicted) but use values (observed) for Lorenz
                sort_idx = np.argsort(sort_values)
                values = values[sort_idx]
            else:
                values = df.select(pl.col(column)).drop_nulls().to_numpy().flatten()
                values.sort()
            
            if len(values) == 0:
                raise ValueError(f"Column '{column}' has no valid numeric data")
            
            n = len(values)
            cum_values = np.cumsum(values)
            x = np.insert(np.arange(1, n + 1) / n, 0, 0)
            y = np.insert(cum_values / cum_values[-1], 0, 0)
        
        area_under_curve = np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2
        gini = 1 - 2 * area_under_curve
        
        return x, y, gini
        
    except Exception as e:
        if 'exposure_column' in str(e):
            raise ValueError(f"Exposure column '{exposure_column}' must contain numeric data") from e
        raise ValueError(f"Column '{column}' must contain numeric data") from e


def calculate_gini(df: pl.DataFrame, column: str, 
                  exposure_column: Optional[str] = None,
                  observed_column: Optional[str] = None) -> float:
    """Calculate the Gini coefficient for a numeric column.
    
    Args:
        df: Polars DataFrame
        column: Name of the numeric column to order by (e.g., predicted values)
        exposure_column: Optional exposure/weight column
        observed_column: Optional name of the column to calculate Gini for.
                        If provided, data is ordered by 'column' but Gini
                        is calculated using 'observed_column' values.
        
    Returns:
        float: Gini coefficient between 0 and 1
    """
    _, _, gini = calculate_lorenz_curve(df, column, exposure_column, observed_column)
    return gini


def bootstrap_gini(df: pl.DataFrame, column: str, exposure_column: Optional[str] = None,
                   observed_column: Optional[str] = None,
                   n_iterations: int = 1000, confidence_level: float = 0.95,
                   random_seed: Optional[int] = None) -> BootstrapResult:
    """Calculate bootstrap confidence intervals for the Gini coefficient.
    
    Args:
        df: Polars DataFrame
        column: Name of the numeric column to order by (e.g., predicted values)
        exposure_column: Optional exposure/weight column
        observed_column: Optional name of the column to calculate Gini for.
                        If provided, data is ordered by 'column' but Gini
                        is calculated using 'observed_column' values.
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level for the interval
        random_seed: Optional random seed
        
    Returns:
        BootstrapResult with estimate and confidence intervals
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
        
    if n_iterations < 100:
        raise ValueError("Number of iterations should be at least 100")
        
    point_estimate = calculate_gini(df, column, exposure_column, observed_column)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if exposure_column is not None:
        # Select columns based on whether observed_column is provided
        select_cols = [
            pl.col(column).alias('sort_value'),
            pl.col(exposure_column).alias('exposure')
        ]
        if observed_column is not None:
            select_cols.append(pl.col(observed_column).alias('value'))
        else:
            select_cols.append(pl.col(column).alias('value'))
            
        data_df = df.select(select_cols).drop_nulls()
        sort_values = data_df.select('sort_value').to_numpy().flatten()
        values = data_df.select('value').to_numpy().flatten()
        exposures = data_df.select('exposure').to_numpy().flatten()
        n_samples = len(values)
        
        bootstrap_samples = np.zeros(n_iterations)
        for i in range(n_iterations):
            indices = np.random.randint(0, n_samples, size=n_samples)
            boot_sort_values = sort_values[indices]
            boot_values = values[indices]
            boot_exposures = exposures[indices]
            
            ratio = boot_sort_values / (boot_exposures + np.finfo(float).eps)
            sort_idx = np.argsort(ratio)
            sorted_values = boot_values[sort_idx]
            sorted_exposures = boot_exposures[sort_idx]
            
            cum_values = np.cumsum(sorted_values)
            cum_exposures = np.cumsum(sorted_exposures)
            
            if cum_values[-1] == 0 or cum_exposures[-1] == 0:
                bootstrap_samples[i] = 0
                continue
            
            x = np.insert(cum_exposures / cum_exposures[-1], 0, 0)
            y = np.insert(cum_values / cum_values[-1], 0, 0)
            
            area_under_curve = np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2
            bootstrap_samples[i] = 1 - 2 * area_under_curve
    else:
        # Select columns based on whether observed_column is provided
        if observed_column is not None:
            data_df = df.select([
                pl.col(column).alias('sort_value'),
                pl.col(observed_column).alias('value')
            ]).drop_nulls()
            sort_values = data_df.select('sort_value').to_numpy().flatten()
            values = data_df.select('value').to_numpy().flatten()
        else:
            values = df.select(pl.col(column)).drop_nulls().to_numpy().flatten()
            sort_values = values.copy()
            
        n_samples = len(values)
        
        bootstrap_samples = np.zeros(n_iterations)
        for i in range(n_iterations):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_sort_values = sort_values[indices]
            boot_values = values[indices]
            
            # Sort by sort_values but use values for Gini calculation
            sort_idx = np.argsort(boot_sort_values)
            boot_values = boot_values[sort_idx]
            
            n = len(boot_values)
            cum_values = np.cumsum(boot_values)
            x = np.insert(np.arange(1, n + 1) / n, 0, 0)
            y = np.insert(cum_values / cum_values[-1], 0, 0)
            
            area_under_curve = np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2
            bootstrap_samples[i] = 1 - 2 * area_under_curve
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_samples, alpha * 100 / 2)
    ci_upper = np.percentile(bootstrap_samples, 100 - alpha * 100 / 2)
    
    return BootstrapResult(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        bootstrap_samples=bootstrap_samples
    )


# ============================================================================
# Lift Curve Functions
# ============================================================================

def calculate_lift_curve(df: pl.DataFrame, target_column: str, score_column: str, 
                         n_bins: int = 10) -> LiftResult:
    """Calculate lift curve coordinates and metrics.
    
    Args:
        df: Polars DataFrame
        target_column: Name of the column containing actual values
        score_column: Name of the column containing model scores
        n_bins: Number of bins to divide the data into
        
    Returns:
        LiftResult with percentiles, lift values, and metrics
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
        
    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in data")
        
    data_df = df.select([
        pl.col(target_column).alias('target'),
        pl.col(score_column).alias('score')
    ]).drop_nulls()
    
    targets = data_df.select('target').to_numpy().flatten()
    scores = data_df.select('score').to_numpy().flatten()
    
    baseline = np.mean(targets)
    
    if baseline == 0:
        raise ValueError("Target column has zero mean, lift cannot be calculated")
        
    sort_idx = np.argsort(scores)[::-1]
    sorted_targets = targets[sort_idx]        
    sorted_scores = scores[sort_idx]
    
    n_samples = len(sorted_targets)
    step_size = n_samples // n_bins
    
    percentiles = np.linspace(0, 100, n_bins + 1)[1:]
    score_lift_values = np.zeros(n_bins)
    target_lift_values = np.zeros(n_bins)
    score_cumulative_lift = np.zeros(n_bins)
    target_cumulative_lift = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_start = i * step_size
        bin_end = (i + 1) * step_size if i < n_bins - 1 else n_samples
        bin_mean_score = np.mean(sorted_scores[bin_start:bin_end])
        bin_mean_target = np.mean(sorted_targets[bin_start:bin_end])
        score_lift_values[i] = bin_mean_score / baseline
        target_lift_values[i] = bin_mean_target / baseline
        
        cumulative_mean_score = np.mean(sorted_scores[:bin_end])
        cumulative_mean_target = np.mean(sorted_targets[:bin_end])
        score_cumulative_lift[i] = cumulative_mean_score / baseline
        target_cumulative_lift[i] = cumulative_mean_target / baseline
    
    auc_score_lift = auc([0] + list(percentiles/100), [1] + list(score_cumulative_lift))
    auc_target_lift = auc([0] + list(percentiles/100), [1] + list(target_cumulative_lift))
    
    return LiftResult(
        percentiles=percentiles,
        score_lift_values=score_lift_values,
        target_lift_values=target_lift_values,
        score_cumulative_lift=score_cumulative_lift,
        target_cumulative_lift=target_cumulative_lift,
        baseline=baseline,
        auc_score_lift=auc_score_lift,
        auc_target_lift=auc_target_lift
    )


# ============================================================================
# ROC Curve Functions
# ============================================================================

def calculate_roc_curve(df: pl.DataFrame, target_column: str, score_column: str) -> ROCResult:
    """Calculate ROC curve coordinates and metrics.
    
    Args:
        df: Polars DataFrame
        target_column: Name of the column containing actual binary values (0 or 1)
        score_column: Name of the column containing model scores
        
    Returns:
        ROCResult with fpr, tpr, thresholds, and metrics
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
        
    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in data")
        
    data_df = df.select([
        pl.col(target_column).alias('target'),
        pl.col(score_column).alias('score')
    ]).drop_nulls()
    
    y_true = data_df.select('target').to_numpy().flatten()
    y_score = data_df.select('score').to_numpy().flatten()
    
    unique_values = np.unique(y_true)
    if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [1, 0]):
        raise ValueError("Target column must contain binary values (0 and 1)")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_point = (fpr[optimal_idx], tpr[optimal_idx])
    
    return ROCResult(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc_score=auc_score,
        optimal_threshold=optimal_threshold,
        optimal_point=optimal_point
    )


def bootstrap_roc_curve(df: pl.DataFrame, target_column: str, score_column: str,
                       n_iterations: int = 1000, confidence_level: float = 0.95,
                       random_seed: Optional[int] = None) -> Tuple[ROCResult, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ROC curve with bootstrap confidence intervals.
    
    Args:
        df: Polars DataFrame
        target_column: Name of the column containing actual binary values
        score_column: Name of the column containing model scores
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
        random_seed: Optional random seed
        
    Returns:
        Tuple of (ROCResult, ci_lower, ci_upper, fpr_points)
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
        
    if n_iterations < 100:
        raise ValueError("Number of iterations should be at least 100")
    
    roc_result = calculate_roc_curve(df, target_column, score_column)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data_df = df.select([
        pl.col(target_column).alias('target'),
        pl.col(score_column).alias('score')
    ]).drop_nulls()
    
    y_true = data_df.select('target').to_numpy().flatten()
    y_score = data_df.select('score').to_numpy().flatten()
    
    fpr_points = np.linspace(0, 1, 100)
    bootstrap_tprs = np.zeros((n_iterations, len(fpr_points)))
    
    n_samples = len(y_true)
    for i in range(n_iterations):
        indices = np.random.randint(0, n_samples, size=n_samples)
        boot_y_true = y_true[indices]
        boot_y_score = y_score[indices]
        
        fpr, tpr, _ = roc_curve(boot_y_true, boot_y_score)
        bootstrap_tprs[i] = np.interp(fpr_points, fpr, tpr)
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_tprs, alpha * 100 / 2, axis=0)
    ci_upper = np.percentile(bootstrap_tprs, 100 - alpha * 100 / 2, axis=0)
    
    return roc_result, ci_lower, ci_upper, fpr_points


# ============================================================================
# Double Lift Functions
# ============================================================================

def calculate_double_lift(df: pl.DataFrame, target_column: str, score1_column: str, 
                         score2_column: str, n_bins: int = 10) -> DoubleLiftResult:
    """Calculate double lift analysis comparing two scoring variables.
    
    Args:
        df: Polars DataFrame
        target_column: Name of the column containing actual values
        score1_column: Name of the first score/prediction column
        score2_column: Name of the second score/prediction column
        n_bins: Number of bins for lift calculation
        
    Returns:
        DoubleLiftResult with lift curves and comparison metrics
    """
    for col in [target_column, score1_column, score2_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")
    
    lift1 = calculate_lift_curve(df, target_column, score1_column, n_bins)
    lift2 = calculate_lift_curve(df, target_column, score2_column, n_bins)
    
    data_df = df.select([
        pl.col(target_column).alias('target'),
        pl.col(score1_column).alias('score1'),
        pl.col(score2_column).alias('score2')
    ]).drop_nulls()
    
    score1 = data_df.select('score1').to_numpy().flatten()
    score2 = data_df.select('score2').to_numpy().flatten()
    targets = data_df.select('target').to_numpy().flatten()
    
    correlation = np.corrcoef(score1, score2)[0, 1]
    
    combined_score = (score1 + score2) / 2
    sort_idx = np.argsort(combined_score)[::-1]
    sorted_targets = targets[sort_idx]
    
    baseline = np.mean(targets)
    if baseline == 0:
        raise ValueError("Target column has zero mean, lift cannot be calculated")
        
    n_samples = len(targets)
    step_size = n_samples // n_bins
    joint_lift = np.mean(sorted_targets[:step_size]) / baseline
    
    sort_idx1 = np.argsort(score1)[::-1]
    top_idx1 = sort_idx1[:step_size]
    remaining_targets = targets[top_idx1]
    remaining_score2 = score2[top_idx1]
    
    sort_idx2 = np.argsort(remaining_score2)[::-1]
    conditional_targets = remaining_targets[sort_idx2]
    conditional_lift = np.mean(conditional_targets[:len(conditional_targets)//2]) / np.mean(remaining_targets)
    
    return DoubleLiftResult(
        lift1=lift1,
        lift2=lift2,
        correlation=correlation,
        joint_lift=joint_lift,
        conditional_lift=conditional_lift
    )


# ============================================================================
# Regression Metrics Functions
# ============================================================================

def calculate_regression_metrics(df: pl.DataFrame, actual_column: str, predicted_column: str,
                                 n_features: Optional[int] = None) -> RegressionMetrics:
    """Calculate comprehensive regression metrics.
    
    Args:
        df: Polars DataFrame
        actual_column: Name of the column containing actual values
        predicted_column: Name of the column containing predicted values
        n_features: Optional number of features (for adjusted R-squared)
        
    Returns:
        RegressionMetrics with mse, rmse, mae, mape, r2, etc.
    """
    if actual_column not in df.columns:
        raise ValueError(f"Actual column '{actual_column}' not found in data")
        
    if predicted_column not in df.columns:
        raise ValueError(f"Predicted column '{predicted_column}' not found in data")
        
    data_df = df.select([
        pl.col(actual_column).alias('actual'),
        pl.col(predicted_column).alias('predicted')
    ]).drop_nulls()
    
    y_true = data_df.select('actual').to_numpy().flatten()
    y_pred = data_df.select('predicted').to_numpy().flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n_samples = len(y_true)
    
    if n_features is not None:
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    else:
        adj_r2 = None
        
    return RegressionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        adj_r2=adj_r2,
        n_samples=n_samples,
        n_features=n_features
    )


# ============================================================================
# Classification Metrics Functions
# ============================================================================

def calculate_classification_metrics(df: Optional[pl.DataFrame], 
                                    true_labels: Union[str, np.ndarray], 
                                    predicted_labels: Union[str, np.ndarray],
                                    pos_label: Any = 1,
                                    threshold: Optional[float] = None,
                                    probability_input: bool = False) -> BinaryClassificationMetrics:
    """Calculate comprehensive metrics for binary classification.
    
    Args:
        df: Optional Polars DataFrame (required if labels are column names)
        true_labels: Column name or numpy array of true labels
        predicted_labels: Column name or numpy array of predicted labels/probabilities
        pos_label: Value to be considered as positive class
        threshold: Classification threshold for probability inputs
        probability_input: If True, predicted_labels contains probabilities
        
    Returns:
        BinaryClassificationMetrics with accuracy, precision, recall, etc.
    """
    if df is not None and isinstance(true_labels, str) and isinstance(predicted_labels, str):
        if true_labels not in df.columns or predicted_labels not in df.columns:
            raise ValueError(f"Columns {true_labels} and/or {predicted_labels} not found in data")
        y_true = df.select(pl.col(true_labels)).drop_nulls().to_numpy().flatten()
        y_pred = df.select(pl.col(predicted_labels)).drop_nulls().to_numpy().flatten()
    else:
        y_true = np.asarray(true_labels)
        y_pred = np.asarray(predicted_labels)

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted labels must have the same shape")

    if probability_input:
        threshold = 0.5 if threshold is None else threshold
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, pos_label=pos_label)
        recall = recall_score(y_true, y_pred_binary, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred_binary, pos_label=pos_label)
        
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall_curve, precision_curve)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=pos_label)
        recall = recall_score(y_true, y_pred, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        pr_auc = None
        roc_auc = None

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=cm,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        specificity=specificity,
        npv=npv,
        pr_auc=pr_auc,
        roc_auc=roc_auc
    )
