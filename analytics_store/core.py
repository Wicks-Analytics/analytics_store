from cProfile import label
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass
from sklearn.metrics import (auc, roc_curve, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, 
                         precision_score, recall_score, f1_score, accuracy_score, confusion_matrix,
                         precision_recall_curve)

@dataclass
class BootstrapResult:
    """Container for bootstrap results."""
    estimate: float
    ci_lower: float
    ci_upper: float
    bootstrap_samples: np.ndarray

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
    
@dataclass
class ROCResult:
    """Container for ROC curve results."""
    fpr: np.ndarray  # False Positive Rate
    tpr: np.ndarray  # True Positive Rate
    thresholds: np.ndarray
    auc_score: float
    optimal_threshold: float
    optimal_point: Tuple[float, float]

@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    mse: float  # Mean Square Error
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
    mape: float   # Mean Absolute Percentage Error
    r2: float    # R-squared score
    adj_r2: float  # Adjusted R-squared
    n_samples: int  # Number of samples
    n_features: Optional[int] = None  # Number of features (for adjusted R-squared)

@dataclass
class DoubleLiftResult:
    """Container for double lift analysis results."""
    lift1: LiftResult
    lift2: LiftResult
    correlation: float
    joint_lift: float
    conditional_lift: float

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
    npv: float  # Negative Predictive Value
    pr_auc: Optional[float] = None  # Precision-Recall AUC (only for probability inputs)
    roc_auc: Optional[float] = None  # ROC AUC (only for probability inputs)

@dataclass
class PopulationTestResult:
    """Container for population comparison test results."""
    statistic: float
    p_value: float
    effect_size: float
    test_type: str
    is_significant: bool
    confidence_interval: Optional[Tuple[float, float]] = None

class DataAnalyser:
    """Main class for data analysis operations using Polars."""
    
    def __init__(self):
        self.data = None
        
    def __getattr__(self, name: str) -> Any:
        """
        Forward any unknown attribute/method calls to the underlying Polars DataFrame.
        This allows using Polars methods directly on the DataAnalyser instance.
        
        Args:
            name: Name of the attribute/method to access
            
        Returns:
            The result of the method call on the underlying DataFrame
            
        Raises:
            ValueError: If no data is loaded
            AttributeError: If the method doesn't exist in Polars DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        if hasattr(self.data, name):
            attr = getattr(self.data, name)
            if callable(attr):
                # If it's a method, return a wrapper that maintains method chaining
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if isinstance(result, pl.DataFrame):
                        # If result is a DataFrame, wrap it in a new DataAnalyser
                        new_analyser = DataAnalyser()
                        new_analyser.data = result
                        return new_analyser
                    return result
                return wrapper
            return attr
            
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def load_data(self, data: Union[str, pl.DataFrame]) -> None:
        """
        Load data from a file or Polars DataFrame.
        
        Args:
            data: Path to file or Polars DataFrame
        """
        if isinstance(data, str):
            self.data = pl.read_csv(data)
        elif isinstance(data, pl.DataFrame):
            self.data = data.clone()
        else:
            raise TypeError("Data must be a file path or Polars DataFrame")
            
    def get_summary_stats(self) -> pl.DataFrame:
        """Return basic statistical summary of the data."""
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.describe()
    
    def get_missing_values(self) -> pl.DataFrame:
        """Return information about missing values."""
        if self.data is None:
            raise ValueError("No data loaded")
        total_rows = self.data.height
        null_counts = self.data.null_count()
        
        return pl.DataFrame({
            'column': self.data.columns,
            'missing_count': null_counts,
            'missing_percentage': (null_counts / total_rows * 100)
        })

    def calculate_lorenz_curve(self, column: str, exposure_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate the Lorenz curve coordinates and Gini coefficient.
        
        The Lorenz curve shows the cumulative proportion of a variable (e.g., income)
        plotted against the cumulative proportion of the population. The diagonal line
        y=x represents perfect equality.
        
        Args:
            column: Name of the numeric column to calculate Lorenz curve for
            exposure_column: Optional name of the column containing exposure/weight values
            
        Returns:
            Tuple containing:
            - np.ndarray: Cumulative proportion of population (x coordinates)
            - np.ndarray: Cumulative proportion of variable (y coordinates)
            - float: Gini coefficient
            
        Raises:
            ValueError: If columns don't exist or contain non-numeric data
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
            
        try:
            if exposure_column is not None:
                if exposure_column not in self.data.columns:
                    raise ValueError(f"Exposure column '{exposure_column}' not found in data")
                    
                # Get both value and exposure data, dropping rows where either is null
                df = self.data.select([
                    pl.col(column).alias('value'),
                    pl.col(exposure_column).alias('exposure')
                ]).drop_nulls()
                
                # Convert to numpy arrays
                values = df.select('value').to_numpy().flatten()
                exposures = df.select('exposure').to_numpy().flatten()
                
                if not np.all(exposures >= 0):
                    raise ValueError("Exposure values must be non-negative")
                
                # Sort by value/exposure ratio
                ratio = values / (exposures + np.finfo(float).eps)
                sort_idx = np.argsort(ratio)
                values = values[sort_idx]
                exposures = exposures[sort_idx]
                
                # Calculate cumulative sums
                cum_values = np.cumsum(values)
                cum_exposures = np.cumsum(exposures)
                
                # Normalize to get proportions
                if cum_values[-1] == 0 or cum_exposures[-1] == 0:
                    return np.linspace(0, 1, 100), np.linspace(0, 1, 100), 0.0
                
                x = np.insert(cum_exposures / cum_exposures[-1], 0, 0)
                y = np.insert(cum_values / cum_values[-1], 0, 0)
                
                # Calculate Gini coefficient
                area_under_curve = np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2
                gini = 1 - 2 * area_under_curve
                
            else:
                # Standard Lorenz curve calculation without exposure
                values = self.data.select(pl.col(column)).drop_nulls().to_numpy().flatten()
                values.sort()
                
                if len(values) == 0:
                    raise ValueError(f"Column '{column}' has no valid numeric data")
                
                # Calculate cumulative sums
                n = len(values)
                cum_values = np.cumsum(values)
                
                # Create x and y coordinates for Lorenz curve
                x = np.insert(np.arange(1, n + 1) / n, 0, 0)
                y = np.insert(cum_values / cum_values[-1], 0, 0)
                
                # Calculate Gini coefficient
                area_under_curve = np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2
                gini = 1 - 2 * area_under_curve
                
            return x, y, gini
            
        except Exception as e:
            if 'exposure_column' in str(e):
                raise ValueError(f"Exposure column '{exposure_column}' must contain numeric data") from e
            raise ValueError(f"Column '{column}' must contain numeric data") from e

    def plot_lorenz_curve(self, column: str, exposure_column: Optional[str] = None, 
                         title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the Lorenz curve for a given column.
        
        Args:
            column: Name of the numeric column to plot Lorenz curve for
            exposure_column: Optional name of the column containing exposure/weight values
            title: Optional title for the plot
            figsize: Tuple of (width, height) for the plot
            
        Raises:
            ValueError: If columns don't exist or contain non-numeric data
        """
        # Calculate Lorenz curve coordinates
        x, y, gini = self.calculate_lorenz_curve(column, exposure_column)
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-dark')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the Lorenz curve
        ax.plot(x, y, 'b-', label='Lorenz Curve', linewidth=2)
        ax.plot([0, 1], [0, 1], 'r--', label='Line of Perfect Equality', linewidth=1)
        
        # Add labels and title
        ax.set_xlabel('Cumulative Proportion of Population')
        ax.set_ylabel(f'Cumulative Proportion of {column}')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Lorenz Curve for {column}\nGini Coefficient: {gini:.3f}')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set aspect ratio to make the plot square
        ax.set_aspect('equal')
        
        # Add text box with Gini coefficient
        textstr = f'Gini Coefficient: {gini:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig

    def calculate_gini(self, column: str, exposure_column: Optional[str] = None) -> float:
        """
        Calculate the Gini coefficient for a numeric column, optionally weighted by exposure.
        
        The Gini coefficient measures inequality among values, with 0 representing
        perfect equality and 1 representing perfect inequality.
        
        Args:
            column: Name of the numeric column to calculate Gini coefficient for
            exposure_column: Optional name of the column containing exposure/weight values
            
        Returns:
            float: Gini coefficient between 0 and 1
            
        Raises:
            ValueError: If columns don't exist or contain non-numeric data
        """
        _, _, gini = self.calculate_lorenz_curve(column, exposure_column)
        return gini

    def bootstrap_gini(self, column: str, exposure_column: Optional[str] = None, 
                      n_iterations: int = 1000, confidence_level: float = 0.95,
                      random_seed: Optional[int] = None) -> BootstrapResult:
        """
        Calculate bootstrap confidence intervals for the Gini coefficient.
        
        Args:
            column: Name of the numeric column to calculate Gini coefficient for
            exposure_column: Optional name of the column containing exposure/weight values
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for the interval (between 0 and 1)
            random_seed: Optional random seed for reproducibility
            
        Returns:
            BootstrapResult containing:
            - estimate: Point estimate of Gini coefficient
            - ci_lower: Lower bound of confidence interval
            - ci_upper: Upper bound of confidence interval
            - bootstrap_samples: Array of bootstrap Gini estimates
            
        Raises:
            ValueError: If parameters are invalid or data is missing
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
            
        if n_iterations < 100:
            raise ValueError("Number of iterations should be at least 100")
            
        # Calculate point estimate
        point_estimate = self.calculate_gini(column, exposure_column)
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Prepare data for bootstrapping
        if exposure_column is not None:
            df = self.data.select([
                pl.col(column).alias('value'),
                pl.col(exposure_column).alias('exposure')
            ]).drop_nulls()
            values = df.select('value').to_numpy().flatten()
            exposures = df.select('exposure').to_numpy().flatten()
            n_samples = len(values)
            
            # Perform bootstrap iterations
            bootstrap_samples = np.zeros(n_iterations)
            for i in range(n_iterations):
                # Sample with replacement
                indices = np.random.randint(0, n_samples, size=n_samples)
                boot_values = values[indices]
                boot_exposures = exposures[indices]
                
                # Calculate Gini for this bootstrap sample
                ratio = boot_values / (boot_exposures + np.finfo(float).eps)
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
            values = self.data.select(pl.col(column)).drop_nulls().to_numpy().flatten()
            n_samples = len(values)
            
            # Perform bootstrap iterations
            bootstrap_samples = np.zeros(n_iterations)
            for i in range(n_iterations):
                # Sample with replacement
                boot_values = np.random.choice(values, size=n_samples, replace=True)
                boot_values.sort()
                
                n = len(boot_values)
                cum_values = np.cumsum(boot_values)
                x = np.insert(np.arange(1, n + 1) / n, 0, 0)
                y = np.insert(cum_values / cum_values[-1], 0, 0)
                
                area_under_curve = np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2
                bootstrap_samples[i] = 1 - 2 * area_under_curve
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_samples, alpha * 100 / 2)
        ci_upper = np.percentile(bootstrap_samples, 100 - alpha * 100 / 2)
        
        return BootstrapResult(
            estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bootstrap_samples=bootstrap_samples
        )

    def plot_lorenz_curve_with_ci(self, column: str, exposure_column: Optional[str] = None,
                                 n_iterations: int = 1000, confidence_level: float = 0.95,
                                 title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6),
                                 random_seed: Optional[int] = None) -> plt.Figure:
        """
        Plot the Lorenz curve with bootstrap confidence intervals.
        
        Args:
            column: Name of the numeric column to plot
            exposure_column: Optional name of the column containing exposure/weight values
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for the interval
            title: Optional title for the plot
            figsize: Tuple of (width, height) for the plot
            random_seed: Optional random seed for reproducibility
            
        Raises:
            ValueError: If parameters are invalid or data is missing
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
            
        # Calculate main Lorenz curve
        x, y, gini = self.calculate_lorenz_curve(column, exposure_column)
        
        # Get bootstrap Gini results
        bootstrap_result = self.bootstrap_gini(
            column, exposure_column, n_iterations, confidence_level, random_seed
        )
        
        # Set up the plot
        plt.style.use('seaborn-v0_8-dark')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the main Lorenz curve
        ax.plot(x, y, 'b-', label='Lorenz Curve', linewidth=2)
        ax.plot([0, 1], [0, 1], 'r--', label='Line of Perfect Equality', linewidth=1)
        
        # Calculate and plot confidence intervals for the curve
        if exposure_column is not None:
            df = self.data.select([
                pl.col(column).alias('value'),
                pl.col(exposure_column).alias('exposure')
            ]).drop_nulls()
            values = df.select('value').to_numpy().flatten()
            exposures = df.select('exposure').to_numpy().flatten()
        else:
            values = self.data.select(pl.col(column)).drop_nulls().to_numpy().flatten()
            exposures = None
            
        n_points = 100
        bootstrap_curves = np.zeros((n_iterations, n_points))
        
        # Calculate bootstrap Lorenz curves
        x_points = np.linspace(0, 1, n_points)
        for i in range(n_iterations):
            if exposures is not None:
                indices = np.random.randint(0, len(values), size=len(values))
                boot_values = values[indices]
                boot_exposures = exposures[indices]
                
                ratio = boot_values / (boot_exposures + np.finfo(float).eps)
                sort_idx = np.argsort(ratio)
                boot_values = boot_values[sort_idx]
                boot_exposures = boot_exposures[sort_idx]
                
                cum_values = np.cumsum(boot_values)
                cum_exposures = np.cumsum(boot_exposures)
                
                if cum_values[-1] == 0 or cum_exposures[-1] == 0:
                    bootstrap_curves[i] = x_points
                    continue
                
                x_boot = np.insert(cum_exposures / cum_exposures[-1], 0, 0)
                y_boot = np.insert(cum_values / cum_values[-1], 0, 0)
                
                # Interpolate to get consistent x points
                bootstrap_curves[i] = np.interp(x_points, x_boot, y_boot)
            else:
                boot_values = np.random.choice(values, size=len(values), replace=True)
                boot_values.sort()
                
                cum_values = np.cumsum(boot_values)
                x_boot = np.insert(np.arange(1, len(boot_values) + 1) / len(boot_values), 0, 0)
                y_boot = np.insert(cum_values / cum_values[-1], 0, 0)
                
                # Interpolate to get consistent x points
                bootstrap_curves[i] = np.interp(x_points, x_boot, y_boot)
        
        # Calculate confidence intervals for the curves
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_curves, alpha * 100 / 2, axis=0)
        ci_upper = np.percentile(bootstrap_curves, 100 - alpha * 100 / 2, axis=0)
        
        # Plot confidence intervals
        ax.fill_between(x_points, ci_lower, ci_upper, alpha=0.2, color='blue',
                       label=f'{confidence_level*100:.0f}% Confidence Interval')
        
        # Add labels and title
        ax.set_xlabel('Cumulative Proportion of Population')
        ax.set_ylabel(f'Cumulative Proportion of {column}')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Lorenz Curve for {column}\nGini: {gini:.3f} [{bootstrap_result.ci_lower:.3f}, {bootstrap_result.ci_upper:.3f}]')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set aspect ratio to make the plot square
        ax.set_aspect('equal')
        
        # Add text box with Gini coefficient and CI
        textstr = (f'Gini: {gini:.3f}\n'
                  f'95% CI: [{bootstrap_result.ci_lower:.3f}, {bootstrap_result.ci_upper:.3f}]')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig

    def calculate_lift_curve(self, target_column: str, score_column: str, 
                           n_bins: int = 10) -> LiftResult:
        """
        Calculate lift curve coordinates and metrics.
        
        The lift curve shows how much better your model performs compared to a random selection.
        It plots the ratio of the target variable mean at each percentile to the overall mean.
        
        Args:
            target_column: Name of the column containing actual values (binary or continuous)
            score_column: Name of the column containing model scores or predictions
            n_bins: Number of bins to divide the data into (default: 10)
            
        Returns:
            LiftResult containing:
            - percentiles: Points at which lift is calculated
            - lift_values: Lift value at each percentile
            - cumulative_lift: Cumulative lift at each percentile
            - baseline: Overall mean of the target variable
            - auc_lift: Area under the lift curve
            
        Raises:
            ValueError: If columns don't exist or contain invalid data
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        if score_column not in self.data.columns:
            raise ValueError(f"Score column '{score_column}' not found in data")
            
        # Get data as numpy arrays
        df = self.data.select([
            pl.col(target_column).alias('target'),
            pl.col(score_column).alias('score')
        ]).drop_nulls()
        
        targets = df.select('target').to_numpy().flatten()
        scores = df.select('score').to_numpy().flatten()
        
        # Calculate baseline (overall mean)
        baseline = np.mean(targets)
        
        if baseline == 0:
            raise ValueError("Target column has zero mean, lift cannot be calculated")
            
        # Sort by scores in descending order
        sort_idx = np.argsort(scores)[::-1]
        sorted_targets = targets[sort_idx]        
        sorted_scores = scores[sort_idx]
        
        # Calculate points for the lift curve
        n_samples = len(sorted_targets)
        step_size = n_samples // n_bins
        
        percentiles = np.linspace(0, 100, n_bins + 1)[1:]  # exclude 0
        score_lift_values = np.zeros(n_bins)
        target_lift_values = np.zeros(n_bins)
        score_cumulative_lift = np.zeros(n_bins)
        target_cumulative_lift = np.zeros(n_bins)
        
        for i in range(n_bins):
            # Calculate lift for this bin
            bin_start = i * step_size
            bin_end = (i + 1) * step_size if i < n_bins - 1 else n_samples
            bin_mean_score = np.mean(sorted_scores[bin_start:bin_end])
            bin_mean_target = np.mean(sorted_targets[bin_start:bin_end])
            score_lift_values[i] = bin_mean_score / baseline
            target_lift_values[i] = bin_mean_target / baseline
            
            # Calculate cumulative lift up to this point
            cumulative_mean_score = np.mean(sorted_scores[:bin_end])
            cumulative_mean_target = np.mean(sorted_targets[:bin_end])
            score_cumulative_lift[i] = cumulative_mean_score / baseline
            target_cumulative_lift[i] = cumulative_mean_target / baseline
        
        # Calculate area under the lift curve
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
    
    def plot_lift_curve(self, target_column: str, score_column: str, n_bins: int = 10,
                       title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot lift curve showing both point-wise and cumulative lift.
        
        Args:
            target_column: Name of the column containing actual values
            score_column: Name of the column containing model scores
            n_bins: Number of bins to divide the data into
            title: Optional title for the plot
            figsize: Tuple of (width, height) for the plot
            
        Raises:
            ValueError: If columns don't exist or contain invalid data
        """
        # Calculate lift curve
        lift_result = self.calculate_lift_curve(target_column, score_column, n_bins)
        
        # Create figure with two subplots
        plt.style.use('seaborn-v0_8-dark')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot point-wise lift
        ax1.plot(lift_result.percentiles, lift_result.score_lift_values, 'red', marker='o', label='Predicted')
        ax1.plot(lift_result.percentiles, lift_result.target_lift_values, 'blue', marker='o', label='Actual')
        ax1.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax1.set_xlabel('Percentile')
        ax1.set_ylabel('Lift')
        ax1.set_title('Point-wise Lift by Percentile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot cumulative lift
        ax2.plot(lift_result.percentiles, lift_result.score_cumulative_lift, 'red', marker='o', label='Predicted')
        ax2.plot(lift_result.percentiles, lift_result.target_cumulative_lift, 'blue', marker='o', label='Actual')
        ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('Cumulative Lift')
        ax2.set_title('Cumulative Lift by Percentile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add overall title if provided
        if title:
            fig.suptitle(title)
            
        # Add text box with metrics
        textstr = (f'Baseline: {lift_result.baseline:.3f}\n'
                  f'AUC Lift: {lift_result.auc_score_lift:.3f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, textstr, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig

    def calculate_roc_curve(self, target_column: str, score_column: str) -> ROCResult:
        """
        Calculate ROC curve coordinates and metrics.
        
        The ROC (Receiver Operating Characteristic) curve shows the trade-off between
        the true positive rate and false positive rate at various classification thresholds.
        
        Args:
            target_column: Name of the column containing actual binary values (0 or 1)
            score_column: Name of the column containing model scores or probabilities
            
        Returns:
            ROCResult containing:
            - fpr: False Positive Rate at each threshold
            - tpr: True Positive Rate at each threshold
            - thresholds: Classification thresholds
            - auc_score: Area Under the ROC Curve
            - optimal_threshold: Threshold that minimizes distance to (0,1)
            - optimal_point: (FPR, TPR) at optimal threshold
            
        Raises:
            ValueError: If columns don't exist or contain invalid data
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        if score_column not in self.data.columns:
            raise ValueError(f"Score column '{score_column}' not found in data")
            
        # Get data as numpy arrays
        df = self.data.select([
            pl.col(target_column).alias('target'),
            pl.col(score_column).alias('score')
        ]).drop_nulls()
        
        y_true = df.select('target').to_numpy().flatten()
        y_score = df.select('score').to_numpy().flatten()
        
        # Validate binary targets
        unique_values = np.unique(y_true)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [1, 0]):
            raise ValueError("Target column must contain binary values (0 and 1)")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)
        
        # Find optimal threshold using Youden's J statistic
        # This maximizes the distance to the diagonal line
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
    
    def bootstrap_roc_curve(self, target_column: str, score_column: str,
                          n_iterations: int = 1000, confidence_level: float = 0.95,
                          random_seed: Optional[int] = None) -> Tuple[ROCResult, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve with bootstrap confidence intervals.
        
        Args:
            target_column: Name of the column containing actual binary values
            score_column: Name of the column containing model scores
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for intervals
            random_seed: Optional random seed for reproducibility
            
        Returns:
            Tuple containing:
            - ROCResult for the original data
            - Lower bound TPR values
            - Upper bound TPR values
            - FPR points at which bounds are calculated
            
        Raises:
            ValueError: If parameters are invalid or data is missing
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
            
        if n_iterations < 100:
            raise ValueError("Number of iterations should be at least 100")
        
        # Calculate original ROC curve
        roc_result = self.calculate_roc_curve(target_column, score_column)
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Get data for bootstrapping
        df = self.data.select([
            pl.col(target_column).alias('target'),
            pl.col(score_column).alias('score')
        ]).drop_nulls()
        
        y_true = df.select('target').to_numpy().flatten()
        y_score = df.select('score').to_numpy().flatten()
        
        # Define fixed FPR points for interpolation
        fpr_points = np.linspace(0, 1, 100)
        
        # Store interpolated TPR values for each bootstrap
        bootstrap_tprs = np.zeros((n_iterations, len(fpr_points)))
        
        n_samples = len(y_true)
        for i in range(n_iterations):
            # Sample with replacement
            indices = np.random.randint(0, n_samples, size=n_samples)
            boot_y_true = y_true[indices]
            boot_y_score = y_score[indices]
            
            # Calculate ROC curve for this bootstrap sample
            fpr, tpr, _ = roc_curve(boot_y_true, boot_y_score)
            
            # Interpolate TPR values at fixed FPR points
            bootstrap_tprs[i] = np.interp(fpr_points, fpr, tpr)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_tprs, alpha * 100 / 2, axis=0)
        ci_upper = np.percentile(bootstrap_tprs, 100 - alpha * 100 / 2, axis=0)
        
        return roc_result, ci_lower, ci_upper, fpr_points
    
    def plot_roc_curve(self, target_column: str, score_column: str,
                      with_ci: bool = True, n_iterations: int = 1000,
                      confidence_level: float = 0.95, title: Optional[str] = None,
                      figsize: Tuple[int, int] = (10, 8),
                      random_seed: Optional[int] = None) -> plt.Figure:
        """
        Plot ROC curve optionally with bootstrap confidence intervals.
        
        Args:
            target_column: Name of the column containing actual binary values
            score_column: Name of the column containing model scores
            with_ci: Whether to include bootstrap confidence intervals
            n_iterations: Number of bootstrap iterations if with_ci is True
            confidence_level: Confidence level for intervals if with_ci is True
            title: Optional title for the plot
            figsize: Tuple of (width, height) for the plot
            random_seed: Optional random seed for reproducibility
            
        Raises:
            ValueError: If parameters are invalid or data is missing
        """
        plt.style.use('seaborn-v0_8-dark')
        fig, ax = plt.subplots(figsize=figsize)
        
        if with_ci:
            # Get ROC curve with confidence intervals
            roc_result, ci_lower, ci_upper, fpr_points = self.bootstrap_roc_curve(
                target_column, score_column, n_iterations, confidence_level, random_seed
            )
            
            # Plot confidence intervals
            ax.fill_between(fpr_points, ci_lower, ci_upper, color='blue', alpha=0.2,
                          label=f'{confidence_level*100:.0f}% Confidence Interval')
        else:
            # Just calculate basic ROC curve
            roc_result = self.calculate_roc_curve(target_column, score_column)
        
        # Plot ROC curve
        ax.plot(roc_result.fpr, roc_result.tpr, 'b-', label=f'ROC (AUC = {roc_result.auc_score:.3f})',
                linewidth=2)
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier', linewidth=1)
        
        # Mark optimal point
        ax.plot(roc_result.optimal_point[0], roc_result.optimal_point[1], 'go',
                label=f'Optimal Point (threshold = {roc_result.optimal_threshold:.3f})')
        
        # Add labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('ROC Curve Analysis')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # Set aspect ratio to make the plot square
        ax.set_aspect('equal')
        
        # Add text box with metrics
        textstr = (f'AUC: {roc_result.auc_score:.3f}\n'
                  f'Optimal Threshold: {roc_result.optimal_threshold:.3f}\n'
                  f'Optimal Point FPR: {roc_result.optimal_point[0]:.3f}\n'
                  f'Optimal Point TPR: {roc_result.optimal_point[1]:.3f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig

    def calculate_double_lift(self, target_column: str, score1_column: str, 
                            score2_column: str, n_bins: int = 10) -> DoubleLiftResult:
        """
        Calculate double lift analysis comparing two scoring variables.
        
        Args:
            target_column: Name of the column containing actual values
            score1_column: Name of the first score/prediction column
            score2_column: Name of the second score/prediction column
            n_bins: Number of bins for lift calculation
            
        Returns:
            DoubleLiftResult containing:
            - lift1: LiftResult for first score
            - lift2: LiftResult for second score
            - correlation: Correlation between scores
            - joint_lift: Maximum achievable lift using both scores
            - conditional_lift: Additional lift from second score given first
            
        Raises:
            ValueError: If columns don't exist or contain invalid data
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        for col in [target_column, score1_column, score2_column]:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Calculate individual lift curves
        lift1 = self.calculate_lift_curve(target_column, score1_column, n_bins)
        lift2 = self.calculate_lift_curve(target_column, score2_column, n_bins)
        
        # Get data for correlation and joint analysis
        df = self.data.select([
            pl.col(target_column).alias('target'),
            pl.col(score1_column).alias('score1'),
            pl.col(score2_column).alias('score2')
        ]).drop_nulls()
        
        score1 = df.select('score1').to_numpy().flatten()
        score2 = df.select('score2').to_numpy().flatten()
        targets = df.select('target').to_numpy().flatten()
        
        # Calculate correlation between scores
        correlation = np.corrcoef(score1, score2)[0, 1]
        
        # Calculate joint lift (using both scores)
        # We'll use a simple average of scores for combination
        combined_score = (score1 + score2) / 2
        sort_idx = np.argsort(combined_score)[::-1]
        sorted_targets = targets[sort_idx]
        
        baseline = np.mean(targets)
        if baseline == 0:
            raise ValueError("Target column has zero mean, lift cannot be calculated")
            
        n_samples = len(targets)
        step_size = n_samples // n_bins
        joint_lift = np.mean(sorted_targets[:step_size]) / baseline
        
        # Calculate conditional lift
        # This shows how much additional lift we get from score2 after using score1
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
    
    def plot_double_lift(self, target_column: str, score1_column: str, score2_column: str,
                        n_bins: int = 10, title: Optional[str] = None, 
                        figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Create a double lift plot comparing two scoring variables.
        
        Args:
            target_column: Name of the column containing actual values
            score1_column: Name of the first score/prediction column
            score2_column: Name of the second score/prediction column
            n_bins: Number of bins for lift calculation
            title: Optional title for the plot
            figsize: Tuple of (width, height) for the plot
            
        Raises:
            ValueError: If columns don't exist or contain invalid data
        """
        # Calculate double lift results
        results = self.calculate_double_lift(target_column, score1_column, score2_column, n_bins)
        
        # Create figure with three subplots
        plt.style.use('seaborn-v0_8-dark')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Compare point-wise lift curves
        ax1.plot(results.lift1.percentiles, results.lift1.lift_values, 'b-', 
                marker='o', label=score1_column)
        ax1.plot(results.lift2.percentiles, results.lift2.lift_values, 'g-',
                marker='s', label=score2_column)
        ax1.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax1.set_xlabel('Percentile')
        ax1.set_ylabel('Lift')
        ax1.set_title('Point-wise Lift Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Compare cumulative lift curves
        ax2.plot(results.lift1.percentiles, results.lift1.cumulative_lift, 'b-',
                marker='o', label=score1_column)
        ax2.plot(results.lift2.percentiles, results.lift2.cumulative_lift, 'g-',
                marker='s', label=score2_column)
        ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('Cumulative Lift')
        ax2.set_title('Cumulative Lift Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Score correlation scatter plot
        df = self.data.select([
            pl.col(score1_column).alias('score1'),
            pl.col(score2_column).alias('score2')
        ]).drop_nulls()
        
        score1 = df.select('score1').to_numpy().flatten()
        score2 = df.select('score2').to_numpy().flatten()
        
        ax3.scatter(score1, score2, alpha=0.5, s=20)
        ax3.set_xlabel(score1_column)
        ax3.set_ylabel(score2_column)
        ax3.set_title('Score Correlation')
        ax3.grid(True, alpha=0.3)
        
        # Add text box with metrics
        textstr = (f'Correlation: {results.correlation:.3f}\n'
                  f'Joint Lift: {results.joint_lift:.3f}\n'
                  f'Conditional Lift: {results.conditional_lift:.3f}\n\n'
                  f'AUC Lift 1: {results.lift1.auc_lift:.3f}\n'
                  f'AUC Lift 2: {results.lift2.auc_lift:.3f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, y=1.05)
        
        plt.tight_layout()
        return fig

    def calculate_regression_metrics(self, actual_column: str, predicted_column: str,
                                   n_features: Optional[int] = None) -> RegressionMetrics:
        """
        Calculate comprehensive regression metrics including RMSE, MAE, and R-squared.
        
        Args:
            actual_column: Name of the column containing actual values
            predicted_column: Name of the column containing predicted values
            n_features: Optional number of features used in the model (for adjusted R-squared)
            
        Returns:
            RegressionMetrics containing:
            - rmse: Root Mean Square Error
            - mae: Mean Absolute Error
            - r2: R-squared score
            - adj_r2: Adjusted R-squared (if n_features provided)
            - n_samples: Number of samples
            - n_features: Number of features (if provided)
            
        Raises:
            ValueError: If columns don't exist or contain invalid data
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        if actual_column not in self.data.columns:
            raise ValueError(f"Actual column '{actual_column}' not found in data")
            
        if predicted_column not in self.data.columns:
            raise ValueError(f"Predicted column '{predicted_column}' not found in data")
            
        # Get data as numpy arrays
        df = self.data.select([
            pl.col(actual_column).alias('actual'),
            pl.col(predicted_column).alias('predicted')
        ]).drop_nulls()
        
        y_true = df.select('actual').to_numpy().flatten()
        y_pred = df.select('predicted').to_numpy().flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        n_samples = len(y_true)
        
        # Calculate adjusted R-squared if n_features is provided
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
    
    def plot_regression_diagnostics(self, actual_column: str, predicted_column: str,
                                  title: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a comprehensive set of regression diagnostic plots.
        
        Creates a 2x2 panel of plots:
        1. Actual vs Predicted scatter plot
        2. Residuals vs Predicted values
        3. Residual histogram
        4. Q-Q plot of residuals
        
        Args:
            actual_column: Name of the column containing actual values
            predicted_column: Name of the column containing predicted values
            title: Optional title for the plot
            figsize: Tuple of (width, height) for the plot
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            ValueError: If columns don't exist or contain non-numeric data
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        if actual_column not in self.data.columns:
            raise ValueError(f"Column '{actual_column}' not found in data")
            
        if predicted_column not in self.data.columns:
            raise ValueError(f"Column '{predicted_column}' not found in data")
            
        # Get data as numpy arrays
        df = self.data.select([
            pl.col(actual_column).alias('actual'),
            pl.col(predicted_column).alias('predicted')
        ]).drop_nulls()
        
        actual = df.select('actual').to_numpy().flatten()
        predicted = df.select('predicted').to_numpy().flatten()
        
        # Calculate residuals
        residuals = actual - predicted
        
        # Create figure and subplots
        plt.style.use('default')
        sns.set_theme()
        fig = plt.figure(figsize=figsize)
        
        # 1. Actual vs Predicted scatter plot
        ax1 = plt.subplot(221)
        ax1.scatter(predicted, actual, alpha=0.5)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Actual Values')
        ax1.set_title('Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        ax2 = plt.subplot(222)
        ax2.scatter(predicted, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residual histogram
        ax3 = plt.subplot(223)
        ax3.hist(residuals, bins=30, edgecolor='black')
        ax3.set_xlabel('Residual Value')
        ax3.set_ylabel('Count')
        ax3.set_title('Residual Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax4 = plt.subplot(224)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Normal Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title)
            
        # Add text box with metrics
        metrics = self.calculate_regression_metrics(actual_column, predicted_column)
        textstr = (f'RMSE: {metrics.rmse:.2f}\n'
                  f'MAE: {metrics.mae:.2f}\n'
                  f'R²: {metrics.r2:.3f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, textstr, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig

    def plot_actual_vs_expected_by_factor(self, actual_column: str, predicted_column: str, factor_column: str,
                                        exposure_column: Optional[str] = None, title: Optional[str] = None, 
                                        figsize: Optional[Tuple[int, int]] = None,
                                        n_bins: [int] = 20) -> plt.Figure:
        """
        Create an actual vs expected plot grouped by a factor on the x-axis.
        
        Args:
            actual_column: Name of the column containing actual values
            predicted_column: Name of the column containing predicted values
            factor_column: Name of the column containing the categorical factor to group by (x-axis)
            exposure_column: Optional column to show as bar chart on secondary y-axis
            title: Optional title for the plot
            figsize: Optional tuple of (width, height) for the plot
            n_bins: Optional number of bins to split numeric factor columns into
            
        Raises:
            ValueError: If columns don't exist or contain invalid data
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        required_cols = [actual_column, predicted_column, factor_column]
        if exposure_column is not None:
            required_cols.append(exposure_column)
            
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Get data and drop nulls
        select_cols = [
            pl.col(actual_column).alias('actual'),
            pl.col(predicted_column).alias('predicted'),
            pl.col(factor_column).alias('factor')
        ]
        if exposure_column is not None:
            select_cols.append(pl.col(exposure_column).alias('exposure'))
            
        df = self.data.select(select_cols).drop_nulls()
        
        # Check if factor is numeric and needs binning
        is_numeric = df.select(pl.col('factor')).dtypes[0] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        unique_factors = df.select('factor').unique().height
        needs_binning = is_numeric and unique_factors > n_bins
        
        if needs_binning:
            # Create bins for numeric factors
            quantiles = np.linspace(0, 1, n_bins + 1)  # n_bins + 1 edges for n_bins
            
            # Get quantile values
            factor_quantiles = df.select(pl.col('factor').quantile(quantiles)).row(0)
            
            # Create labels
            # bin_labels = [f'{factor_quantiles[i]:.1f} - {factor_quantiles[i+1]:.1f}' for i in range(len(factor_quantiles)-1)]
            
            # Add bin column using qcut for equal-sized bins
            df = df.with_columns([
                pl.col('factor')
                .qcut(n_bins, allow_duplicates=True)
                .alias('factor_bin')
            ])
            
            # Calculate statistics by bin
            agg_exprs = [
                pl.col('actual').mean().alias('actual_mean'),
                pl.col('predicted').mean().alias('predicted_mean'),
                pl.col('actual').count().alias('count')
            ]
            if exposure_column is not None:
                agg_exprs.append(pl.col('exposure').sum().alias('exposure_sum'))
                
            stats = df.group_by('factor_bin').agg(agg_exprs).sort('factor_bin')
            
            # Use bin labels directly for x-axis
            x_labels = bin_labels
        else:
            # Calculate statistics by original factor
            agg_exprs = [
                pl.col('actual').mean().alias('actual_mean'),
                pl.col('predicted').mean().alias('predicted_mean'),
                pl.col('actual').count().alias('count')
            ]
            if exposure_column is not None:
                agg_exprs.append(pl.col('exposure').sum().alias('exposure_sum'))
                
            stats = df.group_by('factor').agg(agg_exprs).sort('factor')
            x_labels = stats.select('factor').to_numpy().flatten()
        
        # Set up the plot
        if figsize is None:
            figsize = (12, 6)
            
        plt.style.use('default')
        sns.set_theme()
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get x positions for bars and points
        x = np.arange(len(stats))
        
        # Plot actual and predicted means as points with error bars
        ax.plot(x, stats.select('actual_mean').to_numpy().flatten(), 
                  color='blue', alpha=0.7, label='Actual')
        ax.plot(x, stats.select('predicted_mean').to_numpy().flatten(), 
                  color='red', alpha=0.7, label='Predicted')
        
        # Add connecting lines between actual and predicted for each factor
        for i in range(len(x)):
            ax.plot([x[i], x[i]], 
                   [stats.select('actual_mean').to_numpy().flatten()[i],
                    stats.select('predicted_mean').to_numpy().flatten()[i]],
                   color='gray', alpha=0.3, linestyle='--')
        
        # Add exposure/count bars on secondary axis
        ax2 = ax.twinx()
        if exposure_column is not None:
            bars = ax2.bar(x, stats.select('exposure_sum').to_numpy().flatten(),
                         alpha=0.2, color='gray', label=exposure_column)
            ax2.set_ylabel(f'Sum of {exposure_column}', color='gray')
        else:
            bars = ax2.bar(x, stats.select('count').to_numpy().flatten(),
                         alpha=0.2, color='gray', label='Count')
            ax2.set_ylabel('Count', color='gray')
        
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Calculate overall metrics
        metrics = self.calculate_regression_metrics(actual_column, predicted_column)
        
        # Add text box with overall metrics
        textstr = (f'Overall Metrics:\n'
                  f'N: {metrics.n_samples:,}\n'
                  f'R²: {metrics.r2:.3f}\n'
                  f'RMSE: {metrics.rmse:.2f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        # Customize plot
        ax.set_xlabel(factor_column)
        ax.set_ylabel('Value')
        if title:
            plt.title(title)
            
        # Set x-axis ticks
        ax.set_xticks(x)
        if needs_binning:
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_xlabel(f'{factor_column} (binned)')
        else:
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_xlabel(factor_column)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        return fig

    def calculate_classification_metrics(self, true_labels: Union[str, np.ndarray], 
                                     predicted_labels: Union[str, np.ndarray],
                                     pos_label: Any = 1,
                                     threshold: Optional[float] = None,
                                     probability_input: bool = False) -> BinaryClassificationMetrics:
        """
        Calculate comprehensive metrics for binary classification.

        Args:
            true_labels: Column name containing true labels or numpy array of true labels
            predicted_labels: Column name containing predicted labels/probabilities or numpy array
            pos_label: Value to be considered as positive class (default: 1)
            threshold: Classification threshold for probability inputs (default: 0.5)
            probability_input: If True, predicted_labels contains probabilities/likelihoods

        Returns:
            BinaryClassificationMetrics containing various classification metrics

        Raises:
            ValueError: If columns don't exist or data is invalid
        """
        if self.data is not None and isinstance(true_labels, str) and isinstance(predicted_labels, str):
            if true_labels not in self.data.columns or predicted_labels not in self.data.columns:
                raise ValueError(f"Columns {true_labels} and/or {predicted_labels} not found in data")
            y_true = self.data.select(pl.col(true_labels)).drop_nulls().to_numpy().flatten()
            y_pred = self.data.select(pl.col(predicted_labels)).drop_nulls().to_numpy().flatten()
        else:
            y_true = np.asarray(true_labels)
            y_pred = np.asarray(predicted_labels)

        if y_true.shape != y_pred.shape:
            raise ValueError("True and predicted labels must have the same shape")

        # Handle probability inputs
        if probability_input:
            threshold = 0.5 if threshold is None else threshold
            # Convert probabilities to binary predictions using threshold
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate metrics using thresholded predictions
            accuracy = accuracy_score(y_true, y_pred_binary)
            precision = precision_score(y_true, y_pred_binary, pos_label=pos_label)
            recall = recall_score(y_true, y_pred_binary, pos_label=pos_label)
            f1 = f1_score(y_true, y_pred_binary, pos_label=pos_label)
            
            # Calculate confusion matrix and derived metrics
            cm = confusion_matrix(y_true, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate additional probability-based metrics
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall_curve, precision_curve)
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
        else:
            # Original binary prediction metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, pos_label=pos_label)
            recall = recall_score(y_true, y_pred, pos_label=pos_label)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label)
            
            # Calculate confusion matrix and derived metrics
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Set probability-based metrics to None for binary predictions
            pr_auc = None
            roc_auc = None

        # Calculate additional metrics
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

    def plot_residual_ratios(self, actual_col: str, predicted_col: str, factor_col: str,
                            group_col: Optional[str] = None, rebase_means: bool = False,
                            n_bins: int = 20,
                            title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the residual ratio (actual/predicted) by a factor with separate lines for each group.
        
        Args:
            actual_col: Name of the column containing actual values
            predicted_col: Name of the column containing predicted values
            factor_col: Name of the column to plot on x-axis
            group_col: Optional name of the column to use for grouping (different lines)
            rebase_means: If True, divide each group's ratios by their mean to center them at 1.0
            title: Optional title for the plot
            figsize: Tuple of (width, height) for the plot
            
        Raises:
            ValueError: If required columns don't exist or contain invalid data
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        required_cols = [actual_col, predicted_col, factor_col]
        if group_col is not None:
            required_cols.append(group_col)
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Missing one or more required columns: {required_cols}")
            
        # Select required columns
        select_cols = [
            pl.col(actual_col).alias('actual'),
            pl.col(predicted_col).alias('predicted'),
            pl.col(factor_col).alias('factor')
        ]
        if group_col is not None:
            select_cols.append(pl.col(group_col).alias('group'))
        df = self.data.select(select_cols).drop_nulls()
        
        # Check if factor is numeric and needs binning
        is_numeric = df.select(pl.col('factor')).dtypes[0] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        unique_factors = df.select('factor').unique().height
        needs_binning = is_numeric and unique_factors > n_bins
        
        if needs_binning:
            # Create bins for numeric factors
            quantiles = np.linspace(0, 1, n_bins + 1)
                                    
            # Add bin column using qcut for equal-sized bins
            df = df.with_columns([
                pl.col('factor')
                .qcut(n_bins, allow_duplicates=True)
                .alias('factor_bin')
            ])
            
            # Use binned factor for plotting
            plot_factor = 'factor_bin'
        else:
            # Use original factor for plotting
            plot_factor = 'factor'
            
        # Calculate ratios
        df = df.with_columns([
            (pl.col('actual') / pl.col('predicted')).alias('ratio')
        ])
        
        if rebase_means and group_col is not None:
            # Calculate mean ratio for each group and divide through
            group_means = df.group_by('group').agg(
                pl.col('ratio').mean().alias('group_mean')
            )
            df = df.join(group_means, on='group')
            df = df.with_columns([
                (pl.col('ratio') / pl.col('group_mean')).alias('ratio')
            ])
        
        # Calculate mean ratios by factor and optionally group
        group_cols = [plot_factor]
        if group_col is not None:
            group_cols.append('group')
        stats = df.group_by(group_cols).agg([
            pl.col('ratio').mean().alias('ratio'),
            pl.col('ratio').count().alias('count')
        ]).sort(group_cols)
        
        # Create the plot
        fig = plt.figure(figsize=figsize)
        
        if group_col is not None:
            # Get unique groups and assign colors
            groups = stats.select('group').unique().to_series().to_list()
            colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
            
            for group, color in zip(groups, colors):
                group_data = stats.filter(pl.col('group') == group)
                x_vals = group_data.select(plot_factor).to_series().to_list()
                y_vals = group_data.select('ratio').to_series().to_list()
                counts = group_data.select('count').to_series().to_list()
                
                # Scale line widths based on counts
                min_count = min(counts)
                max_count = max(counts)
                widths = [1 + 3 * (c - min_count) / (max_count - min_count) if max_count > min_count else 2 for c in counts]
                
                plt.plot(x_vals, y_vals, '-o', label=str(group), color=color, 
                         linewidth=2, markersize=0, alpha=0.7)
            plt.legend(title=group_col)
            x_labels = group_data.select(plot_factor).to_numpy().flatten()
        else:
            x_vals = stats.select(plot_factor).to_series().to_list()
            y_vals = stats.select('ratio').to_series().to_list()
            counts = stats.select('count').to_series().to_list()
            
            # Scale line widths based on counts
            min_count = min(counts)
            max_count = max(counts)
            widths = [1 + 3 * (c - min_count) / (max_count - min_count) if max_count > min_count else 2 for c in counts]
            
            plt.plot(x_vals, y_vals, '-o', color='tab:blue',
                     linewidth=2, markersize=0, alpha=0.7)                     
            x_labels = stats.select(plot_factor).to_numpy().flatten()
        
        # Add reference line at 1.0
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Customize the plot
        default_title = f'Residual Ratios by {factor_col}'
        if group_col is not None:
            default_title += f' grouped by {group_col}'
        plt.title(title or default_title)
        if needs_binning:
            plt.xticks(rotation=45, ha='right')
        plt.xlabel(factor_col)
        plt.ylabel('Actual / Predicted' + (' (Rebased)' if rebase_means else ''))
        
        # Adjust layout and return figure
        fig.tight_layout()
        return fig

    def compare_populations(self, column1: str, column2: str, alpha: float = 0.05,
                          test_type: str = 'auto', equal_var: bool = False) -> PopulationTestResult:
        """
        Test if two datasets are from significantly different populations.

        Args:
            column1: Name of the first column to compare
            column2: Name of the second column to compare
            alpha: Significance level (default: 0.05)
            test_type: Type of test to perform ('t', 'mann_whitney', or 'auto')
            equal_var: Whether to assume equal variances for t-test (default: False)

        Returns:
            PopulationTestResult containing test statistics and interpretation

        Raises:
            ValueError: If columns don't exist or contain non-numeric data
        """
        if self.data is None:
            raise ValueError("No data loaded")

        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError(f"Columns {column1} and/or {column2} not found in data")

        # Extract data as numpy arrays
        data1 = self.data.select(pl.col(column1)).drop_nulls().to_numpy().flatten()
        data2 = self.data.select(pl.col(column2)).drop_nulls().to_numpy().flatten()

        # Determine test type if auto
        if test_type == 'auto':
            # Check for normality using Shapiro-Wilk test
            from scipy.stats import shapiro
            _, p1 = shapiro(data1[:min(5000, len(data1))])  # Limit sample size for speed
            _, p2 = shapiro(data2[:min(5000, len(data2))])
            
            # Use t-test if both samples appear normal (p > 0.05), otherwise Mann-Whitney
            test_type = 't' if (p1 > 0.05 and p2 > 0.05) else 'mann_whitney'

        if test_type == 't':
            from scipy.stats import ttest_ind, norm
            
            # Perform t-test
            statistic, p_value = ttest_ind(data1, data2, equal_var=equal_var)
            
            # Calculate Cohen's d effect size
            n1, n2 = len(data1), len(data2)
            pooled_std = np.sqrt(((n1 - 1) * np.std(data1, ddof=1) ** 2 + 
                                (n2 - 1) * np.std(data2, ddof=1) ** 2) / (n1 + n2 - 2))
            effect_size = abs(np.mean(data1) - np.mean(data2)) / pooled_std
            
            # Calculate confidence interval for difference in means
            if equal_var:
                # Pooled standard error
                se = pooled_std * np.sqrt(1/n1 + 1/n2)
            else:
                # Welch's t-test standard error
                se = np.sqrt(np.var(data1, ddof=1)/n1 + np.var(data2, ddof=1)/n2)
            
            ci_lower = (np.mean(data1) - np.mean(data2)) - se * norm.ppf(1 - alpha/2)
            ci_upper = (np.mean(data1) - np.mean(data2)) + se * norm.ppf(1 - alpha/2)
            
        else:  # Mann-Whitney U test
            from scipy.stats import mannwhitneyu
            
            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Calculate rank-biserial correlation as effect size
            n1, n2 = len(data1), len(data2)
            effect_size = 1 - (2 * statistic) / (n1 * n2)
            
            # For Mann-Whitney, we'll use bootstrap for confidence interval
            ci_lower, ci_upper = None, None

        return PopulationTestResult(
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            test_type=test_type,
            is_significant=p_value < alpha,
            confidence_interval=(ci_lower, ci_upper) if test_type == 't' else None
        )
