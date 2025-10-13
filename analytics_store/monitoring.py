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
class PopulationTestResult:
    """Container for population comparison test results."""
    statistic: float
    p_value: float
    effect_size: float
    test_type: str
    is_significant: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_polars(self) -> pl.DataFrame:
        """Convert population test results to a Polars DataFrame.
        
        Returns:
            Single-row DataFrame with test statistics and results.
        """
        return pl.DataFrame({
            'statistic': [self.statistic],
            'p_value': [self.p_value],
            'effect_size': [self.effect_size],
            'test_type': [self.test_type],
            'is_significant': [self.is_significant],
            'ci_lower': [self.confidence_interval[0] if self.confidence_interval else None],
            'ci_upper': [self.confidence_interval[1] if self.confidence_interval else None]
        })

def compare_populations(df: pl.DataFrame, column1: str, column2: str, alpha: float = 0.05,
                       test_type: str = 'auto', equal_var: bool = False) -> PopulationTestResult:
    """Test if two datasets are from significantly different populations.

    Args:
        df: Polars DataFrame
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
    if column1 not in df.columns or column2 not in df.columns:
        raise ValueError(f"Columns {column1} and/or {column2} not found in data")

    # Extract data as numpy arrays
    data1 = df.select(pl.col(column1)).drop_nulls().to_numpy().flatten()
    data2 = df.select(pl.col(column2)).drop_nulls().to_numpy().flatten()

    # Determine test type if auto
    if test_type == 'auto':
        # Check for normality using Shapiro-Wilk test
        from scipy.stats import shapiro
        _, p1 = shapiro(data1[:min(5000, len(data1))])
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

def monitor_regression_drift(
        data: Optional[pl.DataFrame] = None,
        reference: Optional[pl.DataFrame] = None,
        current: Optional[pl.DataFrame] = None,
        target_col: str = None,
        predicted_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        n_bins: int = 10,
        psi_threshold: float = 0.2,
        alpha: float = 0.05,
        split_col: Optional[str] = None,
        split_value: Optional[Any] = None,
        split_percentile: Optional[float] = None,
        baseline_value: Optional[Any] = None,
        comparison_values: Optional[Union[List[Any], str]] = None,
    ) -> Union[Dict[str, pl.DataFrame], Dict[str, Dict[str, pl.DataFrame]]]:
    """Monitor data and feature drift for regression models using Polars DataFrames.

    Computes per-feature drift statistics and overall performance drift between a
    reference dataset and one or more current datasets.

    Feature drift (per feature):
    - Numeric: PSI (quantile bins from reference), KS test p-value, mean/std, missing %
    - Categorical: PSI (category proportions), Chi-squared p-value, top-k coverage, missing %

    Performance drift (overall):
    - Regression metrics (MSE, RMSE, MAE, MAPE, R2) for reference vs current and deltas

    Args:
        data: Optional Polars DataFrame for split-based analysis
        reference: Reference Polars DataFrame (if None, uses data with split)
        current: Current Polars DataFrame (if None, uses data with split)
        target_col: Name of target column
        predicted_col: Optional name of predicted column (if None, performance drift is skipped)
        feature_cols: Optional list of feature columns to evaluate (defaults to all except target/pred)
        n_bins: Number of bins for PSI on numeric features
        psi_threshold: PSI threshold to flag drift (commonly 0.2 moderate, 0.3 major)
        alpha: Significance level for statistical tests
        split_col: Optional column for automatic splitting (time, cohort, etc.)
        split_value: Value to split on (reference: <= split_value, current: > split_value)
        split_percentile: Percentile to split on (e.g., 0.7 = first 70% reference, last 30% current)
        baseline_value: Specific value in split_col to use as baseline/reference dataset
        comparison_values: Either 'auto' (use all unique values except baseline) or list of values
                          to compare against baseline. Each generates a separate drift report.

    Returns:
        If baseline_value is specified:
            dict with keys for each comparison value, each containing:
            - 'feature_drift': Polars DataFrame with per-feature drift metrics
            - 'performance_drift': Polars DataFrame with overall performance metrics and deltas
        Otherwise (single comparison):
            dict with:
            - 'feature_drift': Polars DataFrame with per-feature drift metrics
            - 'performance_drift': Polars DataFrame with overall performance metrics and deltas
    """
    # Handle split-based splitting (time, cohort, etc.)
    if split_col is not None:
        if reference is not None or current is not None:
            raise ValueError("Cannot specify both split_col and reference/current DataFrames")
        if data is None:
            raise ValueError("No data loaded. Use load_data() or provide reference/current DataFrames")
        if split_col not in data.columns:
            raise ValueError(f"Split column '{split_col}' not found in data")
        
        # Multi-dataset comparison mode
        if baseline_value is not None:
            # Extract baseline dataset
            reference = data.filter(pl.col(split_col) == baseline_value)
            if reference.height == 0:
                raise ValueError(f"No data found for baseline_value '{baseline_value}' in column '{split_col}'")
            
            # Determine comparison values
            if comparison_values == 'auto':
                # Use all unique values except baseline
                all_values = data.select(pl.col(split_col).unique()).to_series().to_list()
                comparison_values = [v for v in all_values if v != baseline_value]
            elif comparison_values is None:
                raise ValueError("Must specify comparison_values when using baseline_value (use 'auto' or provide a list)")
            elif not isinstance(comparison_values, list):
                raise ValueError("comparison_values must be 'auto' or a list of values")
            
            # Recursively compute drift for each comparison
            results = {}
            for comp_val in comparison_values:
                current_df = data.filter(pl.col(split_col) == comp_val)
                if current_df.height == 0:
                    continue  # Skip empty datasets
                
                # Recursive call with explicit DataFrames
                comp_result = monitor_regression_drift(
                    reference=reference,
                    current=current_df,
                    target_col=target_col,
                    predicted_col=predicted_col,
                    feature_cols=feature_cols,
                    n_bins=n_bins,
                    psi_threshold=psi_threshold,
                    alpha=alpha,
                )
                
                # Add comparison identifier to results
                comp_result['feature_drift'] = comp_result['feature_drift'].with_columns(
                    pl.lit(str(comp_val)).alias('comparison_group')
                )
                comp_result['performance_drift'] = comp_result['performance_drift'].with_columns(
                    pl.lit(str(comp_val)).alias('comparison_group')
                )
                
                results[str(comp_val)] = comp_result
            
            return results
        
        # Single comparison mode (original behavior)
        else:
            # Determine split point
            if split_value is not None and split_percentile is not None:
                raise ValueError("Cannot specify both split_value and split_percentile")
            elif split_value is not None:
                reference = data.filter(pl.col(split_col) <= split_value)
                current = data.filter(pl.col(split_col) > split_value)
            elif split_percentile is not None:
                if not 0 < split_percentile < 1:
                    raise ValueError("split_percentile must be between 0 and 1")
                split_val = data.select(pl.col(split_col).quantile(split_percentile)).item()
                reference = data.filter(pl.col(split_col) <= split_val)
                current = data.filter(pl.col(split_col) > split_val)
            else:
                # Default: use median split
                split_val = data.select(pl.col(split_col).median()).item()
                reference = data.filter(pl.col(split_col) <= split_val)
                current = data.filter(pl.col(split_col) > split_val)
    else:
        # Validate that reference and current are provided
        if reference is None or current is None:
            raise ValueError("Must provide either (reference and current DataFrames) or (split_col for splitting)")
    
    # Validate columns
    for df_name, df in (('reference', reference), ('current', current)):
        if target_col not in df.columns:
            raise ValueError(f"{df_name} is missing target column '{target_col}'")
        if predicted_col is not None and predicted_col not in df.columns:
            raise ValueError(f"{df_name} is missing predicted column '{predicted_col}'")

    # Determine features set
    if feature_cols is None:
        exclude = {target_col}
        if predicted_col is not None:
            exclude.add(predicted_col)
        feature_cols = [c for c in reference.columns if c not in exclude and c in current.columns]

    # Helper: numeric check
    numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8}
    def is_numeric_dtype(dt: pl.DataType) -> bool:
        return dt in numeric_types

    # Helper: PSI for two distributions (arrays of proportions aligned)
    def psi(ref_prop: np.ndarray, cur_prop: np.ndarray, eps: float = 1e-10) -> float:
        ref_s = np.clip(ref_prop, eps, None)
        cur_s = np.clip(cur_prop, eps, None)
        return float(np.sum((cur_s - ref_s) * np.log(cur_s / ref_s)))
    
    # Helper: Jensen-Shannon Divergence
    def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """Calculate Jensen-Shannon Divergence between two probability distributions."""
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    # Prepare containers
    feat_rows: List[Dict[str, Any]] = []

    # Iterate features
    for feat in feature_cols:
        # Skip if not in both
        if feat not in reference.columns or feat not in current.columns:
            continue

        ref_col = reference.select(pl.col(feat))
        cur_col = current.select(pl.col(feat))

        ref_dtype = ref_col.dtypes[0]
        cur_dtype = cur_col.dtypes[0]

        # Coerce to same logical type if possible (for categoricals, cast to string)
        treat_as_numeric = is_numeric_dtype(ref_dtype) and is_numeric_dtype(cur_dtype)

        # Missing rates
        ref_missing = float(reference.select(pl.col(feat).is_null().sum()).item())
        cur_missing = float(current.select(pl.col(feat).is_null().sum()).item())
        ref_missing_pct = (ref_missing / max(1, reference.height)) * 100.0
        cur_missing_pct = (cur_missing / max(1, current.height)) * 100.0

        if treat_as_numeric:
            # Drop nulls
            ref_vals = reference.select(pl.col(feat)).drop_nulls().to_numpy().flatten()
            cur_vals = current.select(pl.col(feat)).drop_nulls().to_numpy().flatten()

            # Summary stats
            ref_mean = float(np.nanmean(ref_vals)) if ref_vals.size else np.nan
            cur_mean = float(np.nanmean(cur_vals)) if cur_vals.size else np.nan
            ref_std = float(np.nanstd(ref_vals, ddof=1)) if ref_vals.size > 1 else np.nan
            cur_std = float(np.nanstd(cur_vals, ddof=1)) if cur_vals.size > 1 else np.nan

            # Distance metrics and statistical tests
            if ref_vals.size == 0 or cur_vals.size == 0:
                feat_psi = np.nan
                ks_p = np.nan
                wasserstein_dist = np.nan
                js_div = np.nan
            else:
                # Build bin edges from reference quantiles for PSI and JS
                qs = np.linspace(0, 1, n_bins + 1)
                # Use np.unique to ensure monotonic edges
                edges = np.unique(np.quantile(ref_vals, qs))
                # If too few unique edges, fallback to linspace over combined min/max
                if edges.size < 3:
                    min_v = float(np.nanmin(np.concatenate([ref_vals, cur_vals])))
                    max_v = float(np.nanmax(np.concatenate([ref_vals, cur_vals])))
                    edges = np.linspace(min_v, max_v, min(n_bins + 1, 5))

                # Compute proportions per bin for PSI and JS
                ref_counts, _ = np.histogram(ref_vals, bins=edges)
                cur_counts, _ = np.histogram(cur_vals, bins=edges)
                ref_prop = ref_counts / max(1, ref_counts.sum())
                cur_prop = cur_counts / max(1, cur_counts.sum())
                
                # PSI
                feat_psi = psi(ref_prop, cur_prop)
                
                # Jensen-Shannon Divergence
                js_div = jensen_shannon_divergence(ref_prop, cur_prop)
                
                # Wasserstein Distance (Earth Mover's Distance)
                from scipy.stats import wasserstein_distance
                try:
                    wasserstein_dist = float(wasserstein_distance(ref_vals, cur_vals))
                except Exception:
                    wasserstein_dist = np.nan

                # KS test
                from scipy.stats import ks_2samp
                try:
                    _, ks_p = ks_2samp(ref_vals, cur_vals)
                except Exception:
                    ks_p = np.nan

            if (reference.height < 1000) and (current.height < 1000):
                drift_flag = (not np.isnan(feat_psi) and feat_psi >= psi_threshold) or (not np.isnan(ks_p) and ks_p < alpha)
            else:
                drift_flag = (not np.isnan(feat_psi) and feat_psi >= psi_threshold) 

            feat_rows.append({
                'feature': feat,
                'type': 'numeric',
                'psi': float(feat_psi) if feat_psi is not np.nan else np.nan,
                'js_divergence': float(js_div) if js_div is not np.nan else np.nan,
                'wasserstein_distance': float(wasserstein_dist) if wasserstein_dist is not np.nan else np.nan,
                'stat_test': 'ks',
                'p_value': float(ks_p) if ks_p is not np.nan else np.nan,
                'mean_ref': ref_mean,
                'mean_cur': cur_mean,
                'std_ref': ref_std,
                'std_cur': cur_std,
                'missing_pct_ref': ref_missing_pct,
                'missing_pct_cur': cur_missing_pct,
                'drift_detected': bool(drift_flag),
            })
        else:
            # Treat as categorical (cast to string to align categories)
            ref_series = reference.select(pl.col(feat).cast(pl.Utf8)).to_series()
            cur_series = current.select(pl.col(feat).cast(pl.Utf8)).to_series()

            # Align categories as union of levels from both
            ref_counts = (
                pl.DataFrame({feat: ref_series})
                .group_by(feat)
                .agg(pl.len().alias('cnt'))
            )
            cur_counts = (
                pl.DataFrame({feat: cur_series})
                .group_by(feat)
                .agg(pl.len().alias('cnt'))
            )

            # Join on all categories (outer-like union)
            categories = pl.concat([
                ref_counts.select(pl.col(feat)),
                cur_counts.select(pl.col(feat))
            ]).unique()

            ref_counts = categories.join(ref_counts, on=feat, how='left').with_columns(pl.col('cnt').fill_null(0)).rename({'cnt': 'ref_cnt'})
            cur_counts = categories.join(cur_counts, on=feat, how='left').with_columns(pl.col('cnt').fill_null(0)).rename({'cnt': 'cur_cnt'})

            cat_table = ref_counts.join(cur_counts, on=feat, how='inner')

            ref_total = max(1, cat_table.select(pl.col('ref_cnt').sum()).item())
            cur_total = max(1, cat_table.select(pl.col('cur_cnt').sum()).item())

            ref_prop = cat_table.select((pl.col('ref_cnt') / ref_total).alias('p')).to_numpy().flatten()
            cur_prop = cat_table.select((pl.col('cur_cnt') / cur_total).alias('p')).to_numpy().flatten()

            # PSI
            feat_psi = psi(ref_prop, cur_prop)
            
            # Jensen-Shannon Divergence
            js_div = jensen_shannon_divergence(ref_prop, cur_prop)

            # Chi-squared test
            from scipy.stats import chi2_contingency
            try:
                contingency = np.vstack([
                    cat_table.select('ref_cnt').to_numpy().flatten(),
                    cat_table.select('cur_cnt').to_numpy().flatten(),
                ])
                _, chi_p, _, _ = chi2_contingency(contingency)
            except Exception:
                chi_p = np.nan

            if (reference.height < 1000) and (current.height < 1000):
                drift_flag = (not np.isnan(feat_psi) and feat_psi >= psi_threshold) or (not np.isnan(chi_p) and chi_p < alpha)
            else:
                drift_flag = (not np.isnan(feat_psi) and feat_psi >= psi_threshold)

            feat_rows.append({
                'feature': feat,
                'type': 'categorical',
                'psi': float(feat_psi) if feat_psi is not np.nan else np.nan,
                'js_divergence': float(js_div) if js_div is not np.nan else np.nan,
                'wasserstein_distance': np.nan,  # Not applicable for categorical
                'stat_test': 'chi2',
                'p_value': float(chi_p) if chi_p is not np.nan else np.nan,
                'mean_ref': np.nan,
                'mean_cur': np.nan,
                'std_ref': np.nan,
                'std_cur': np.nan,
                'missing_pct_ref': ref_missing_pct,
                'missing_pct_cur': cur_missing_pct,
                'drift_detected': bool(drift_flag),
            })

    feature_drift_df = pl.DataFrame(feat_rows) if feat_rows else pl.DataFrame(
        {
            'feature': [],
            'type': [],
            'psi': [],
            'js_divergence': [],
            'wasserstein_distance': [],
            'stat_test': [],
            'p_value': [],
            'mean_ref': [],
            'mean_cur': [],
            'std_ref': [],
            'std_cur': [],
            'missing_pct_ref': [],
            'missing_pct_cur': [],
            'drift_detected': [],
        }
    )

    # Performance drift
    perf_rows: List[Dict[str, Any]] = []
    if predicted_col is not None:
        def reg_metrics(df: pl.DataFrame) -> Dict[str, float]:
            arr_y = df.select(pl.col(target_col)).drop_nulls().to_numpy().flatten()
            arr_p = df.select(pl.col(predicted_col)).drop_nulls().to_numpy().flatten()
            # Align lengths if drop_nulls removed different rows: reselect rows where both present
            df2 = df.select([
                pl.col(target_col).alias('y'),
                pl.col(predicted_col).alias('p')
            ]).drop_nulls()
            y = df2.select('y').to_numpy().flatten()
            p = df2.select('p').to_numpy().flatten()
            if y.size == 0:
                return {k: np.nan for k in ['mse','rmse','mae','mape','r2','n']}
            mse_v = mean_squared_error(y, p)
            rmse_v = float(np.sqrt(mse_v))
            mae_v = mean_absolute_error(y, p)
            try:
                mape_v = mean_absolute_percentage_error(y, p)
            except Exception:
                mape_v = np.nan
            try:
                r2_v = r2_score(y, p)
            except Exception:
                r2_v = np.nan
            return {'mse': mse_v, 'rmse': rmse_v, 'mae': mae_v, 'mape': mape_v, 'r2': r2_v, 'n': int(y.size)}

        ref_m = reg_metrics(reference)
        cur_m = reg_metrics(current)
        perf_rows.append({
            'metric': 'MSE', 'reference': ref_m['mse'], 'current': cur_m['mse'], 'delta': (cur_m['mse'] - ref_m['mse']) if not np.isnan(ref_m['mse']) and not np.isnan(cur_m['mse']) else np.nan
        })
        perf_rows.append({
            'metric': 'RMSE', 'reference': ref_m['rmse'], 'current': cur_m['rmse'], 'delta': (cur_m['rmse'] - ref_m['rmse']) if not np.isnan(ref_m['rmse']) and not np.isnan(cur_m['rmse']) else np.nan
        })
        perf_rows.append({
            'metric': 'MAE', 'reference': ref_m['mae'], 'current': cur_m['mae'], 'delta': (cur_m['mae'] - ref_m['mae']) if not np.isnan(ref_m['mae']) and not np.isnan(cur_m['mae']) else np.nan
        })
        perf_rows.append({
            'metric': 'MAPE', 'reference': ref_m['mape'], 'current': cur_m['mape'], 'delta': (cur_m['mape'] - ref_m['mape']) if not np.isnan(ref_m['mape']) and not np.isnan(cur_m['mape']) else np.nan
        })
        perf_rows.append({
            'metric': 'R2', 'reference': ref_m['r2'], 'current': cur_m['r2'], 'delta': (cur_m['r2'] - ref_m['r2']) if not np.isnan(ref_m['r2']) and not np.isnan(cur_m['r2']) else np.nan
        })
        perf_rows.append({
            'metric': 'N', 'reference': ref_m['n'], 'current': cur_m['n'], 'delta': (cur_m['n'] - ref_m['n']) if isinstance(ref_m['n'], (int,float)) and isinstance(cur_m['n'], (int,float)) else np.nan
        })

    performance_drift_df = pl.DataFrame(perf_rows) if perf_rows else pl.DataFrame({'metric': [], 'reference': [], 'current': [], 'delta': []})

    return {
        'feature_drift': feature_drift_df,
        'performance_drift': performance_drift_df,
    }
