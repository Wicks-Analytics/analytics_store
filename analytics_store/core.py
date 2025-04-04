import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Tuple, Dict
from dataclasses import dataclass
from sklearn.metrics import auc, roc_curve, mean_squared_error, mean_absolute_error, r2_score

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
    lift_values: np.ndarray
    cumulative_lift: np.ndarray
    baseline: float
    auc_lift: float
    
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
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
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

class DataAnalyser:
    """Main class for data analysis operations using Polars."""
    
    def __init__(self):
        self.data = None
        
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
                         title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)) -> None:
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
        plt.show()

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
                                 random_seed: Optional[int] = None) -> None:
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
        plt.show()

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
        
        # Calculate points for the lift curve
        n_samples = len(sorted_targets)
        step_size = n_samples // n_bins
        
        percentiles = np.linspace(0, 100, n_bins + 1)[1:]  # exclude 0
        lift_values = np.zeros(n_bins)
        cumulative_lift = np.zeros(n_bins)
        
        for i in range(n_bins):
            # Calculate lift for this bin
            bin_start = i * step_size
            bin_end = (i + 1) * step_size if i < n_bins - 1 else n_samples
            bin_mean = np.mean(sorted_targets[bin_start:bin_end])
            lift_values[i] = bin_mean / baseline
            
            # Calculate cumulative lift up to this point
            cumulative_mean = np.mean(sorted_targets[:bin_end])
            cumulative_lift[i] = cumulative_mean / baseline
        
        # Calculate area under the lift curve
        auc_lift = auc([0] + list(percentiles/100), [1] + list(cumulative_lift))
        
        return LiftResult(
            percentiles=percentiles,
            lift_values=lift_values,
            cumulative_lift=cumulative_lift,
            baseline=baseline,
            auc_lift=auc_lift
        )
    
    def plot_lift_curve(self, target_column: str, score_column: str, n_bins: int = 10,
                       title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)) -> None:
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
        ax1.plot(lift_result.percentiles, lift_result.lift_values, 'b-', marker='o')
        ax1.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax1.set_xlabel('Percentile')
        ax1.set_ylabel('Lift')
        ax1.set_title('Point-wise Lift by Percentile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot cumulative lift
        ax2.plot(lift_result.percentiles, lift_result.cumulative_lift, 'g-', marker='o')
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
                  f'AUC Lift: {lift_result.auc_lift:.3f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, textstr, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()

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
                      random_seed: Optional[int] = None) -> None:
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
        plt.show()

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
                        figsize: Tuple[int, int] = (15, 5)) -> None:
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
        plt.show()

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
        r2 = r2_score(y_true, y_pred)
        
        n_samples = len(y_true)
        
        # Calculate adjusted R-squared if n_features is provided
        if n_features is not None:
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        else:
            adj_r2 = None
            
        return RegressionMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            adj_r2=adj_r2,
            n_samples=n_samples,
            n_features=n_features
        )
    
    def plot_regression_diagnostics(self, actual_column: str, predicted_column: str,
                                  title: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> None:
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
            
        Raises:
            ValueError: If columns don't exist or contain invalid data
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
        plt.show()
