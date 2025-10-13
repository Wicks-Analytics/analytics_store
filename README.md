# AnalyticsStore

A powerful Python package for data analysis and model evaluation using Polars. Built for high performance and ease of use, with a focus on lift analysis and model performance metrics. Features a functional API design where all functions accept Polars DataFrames as parameters.

## Features

- **Lift Analysis**
  - Single variable lift curves
  - Double lift comparison
  - Bootstrap confidence intervals
  - Point-wise and cumulative lift metrics

- **ROC Analysis**
  - ROC curves with confidence intervals
  - AUC score calculation
  - Optimal threshold selection using Youden's J statistic
  - Comprehensive visualization

- **Regression Metrics**
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - R-squared and Adjusted R-squared
  - Diagnostic plots (Actual vs Predicted, Residuals, Q-Q Plot)

- **Performance Metrics**
  - Gini coefficient
  - Lorenz curve
  - Joint and conditional lift metrics
  - Score correlation analysis

- **Model Monitoring**
  - Data drift detection (PSI, KS test, Chi-squared)
  - Feature drift monitoring
  - Performance drift tracking
  - Population comparison tests

## Installation

```bash
pip install git+https://github.com/Wicks-Analytics/analytics_store
```

## Quick Start

```python
import polars as pl
from analytics_store import model_validation, validation_plots

# Load your data as a Polars DataFrame
df = pl.read_csv("model_results.csv")

# Basic lift curve analysis
lift_result = model_validation.calculate_lift_curve(
    df,
    target_column="actual_values",
    score_column="model_scores",
    n_bins=10
)

# Access results as dataclass attributes
print(f"AUC Lift: {lift_result.auc_score_lift:.3f}")
print(f"Baseline: {lift_result.baseline:.3f}")

# Or convert to Polars DataFrame for further analysis
lift_df = lift_result.to_polars()
lift_df.write_csv("lift_results.csv")

# Create visualizations
validation_plots.plot_lift_curve(
    df,
    target_column="actual_values",
    score_column="model_scores",
    title="Model Lift Analysis"
)

# Compare two models using double lift
validation_plots.plot_double_lift(
    df,
    target_column="actual_values",
    score1_column="model1_scores",
    score2_column="model2_scores",
    title="Model Comparison"
)

# ROC curve analysis with confidence intervals
validation_plots.plot_roc_curve(
    df,
    target_column="actual_values",
    score_column="model_scores",
    with_ci=True,
    n_iterations=1000,
    confidence_level=0.95
)

# Regression metrics and diagnostics
metrics = model_validation.calculate_regression_metrics(
    df,
    actual_column="actual_values",
    predicted_column="predicted_values",
    n_features=10  # Optional, for adjusted R-squared
)
print(f"RMSE: {metrics.rmse:.3f}")
print(f"R-squared: {metrics.r2:.3f}")

# Plot regression diagnostics
validation_plots.plot_regression_diagnostics(
    df,
    actual_column="actual_values",
    predicted_column="predicted_values",
    title="Model Regression Diagnostics"
)
```

## Detailed Documentation

### Lift Analysis

The package provides comprehensive lift analysis capabilities:

```python
from analytics_store import model_validation

# Calculate lift metrics
lift_result = model_validation.calculate_lift_curve(
    df,
    target_column="actual_values",
    score_column="model_scores"
)

print(f"Baseline rate: {lift_result.baseline:.3f}")
print(f"AUC Lift: {lift_result.auc_score_lift:.3f}")

# Access lift values
print("Point-wise lift values:", lift_result.score_lift_values)
print("Cumulative lift:", lift_result.score_cumulative_lift)
```

### Double Lift Analysis

Compare two scoring variables:

```python
results = model_validation.calculate_double_lift(
    df,
    target_column="actual_values",
    score1_column="model1_scores",
    score2_column="model2_scores"
)

print(f"Score correlation: {results.correlation:.3f}")
print(f"Joint lift: {results.joint_lift:.3f}")
print(f"Conditional lift: {results.conditional_lift:.3f}")
```

### ROC Analysis

Evaluate binary classification performance:

```python
roc_result = model_validation.calculate_roc_curve(
    df,
    target_column="actual_values",
    score_column="predicted_probabilities"
)

print(f"AUC: {roc_result.auc_score:.3f}")
print(f"Optimal threshold: {roc_result.optimal_threshold:.3f}")
```

### Regression Analysis

Evaluate regression model performance:

```python
from analytics_store import model_validation, validation_plots

# Get comprehensive regression metrics
metrics = model_validation.calculate_regression_metrics(
    df,
    actual_column="actual_values",
    predicted_column="predicted_values",
    n_features=10  # Optional, for adjusted R-squared
)

print(f"RMSE: {metrics.rmse:.3f}")
print(f"MAE: {metrics.mae:.3f}")
print(f"R-squared: {metrics.r2:.3f}")
print(f"Adjusted R-squared: {metrics.adj_r2:.3f}")
print(f"Number of samples: {metrics.n_samples}")

# Create diagnostic plots
validation_plots.plot_regression_diagnostics(
    df,
    actual_column="actual_values",
    predicted_column="predicted_values",
    title="Regression Model Diagnostics"
)

# Analyze model performance by factor
validation_plots.plot_actual_vs_expected_by_factor(
    df,
    actual_column="actual_values",
    predicted_column="predicted_values",
    factor_column="segment",  # Categorical variable to split by
    exposure_column="exposure",  # Optional exposure/weight column
    title="Model Performance by Segment"
)
```

The `plot_actual_vs_expected_by_factor` function creates a plot comparing actual and predicted values across different factor levels. Each factor level shows:
- Mean actual and predicted values as points
- Connecting lines between actual and predicted points
- Bar chart showing either:
  - Sum of exposure values (if exposure_column specified)
  - Count of observations (if no exposure_column)
- Overall metrics (N, R², RMSE)

For numeric factors with more than 20 unique values, the function automatically bins the data into equal-sized bins for better visualization.

### Converting Results to DataFrames

All result dataclasses have a `.to_polars()` method for easy conversion to DataFrames:

```python
# Calculate metrics
metrics = model_validation.calculate_regression_metrics(df, 'actual', 'predicted')

# Use as dataclass (recommended for most cases)
print(f"RMSE: {metrics.rmse:.3f}")
print(f"R²: {metrics.r2:.3f}")

# Convert to DataFrame when needed
metrics_df = metrics.to_polars()
metrics_df.write_csv("metrics.csv")

# Works with all result types
lift_result = model_validation.calculate_lift_curve(df, 'target', 'score')
lift_df = lift_result.to_polars()  # Multi-row DataFrame with lift curves

roc_result = model_validation.calculate_roc_curve(df, 'target', 'score')
roc_df = roc_result.to_polars()  # Multi-row DataFrame with ROC curve points

# Combine multiple results
results = []
for model in ['model1', 'model2', 'model3']:
    metrics = model_validation.calculate_regression_metrics(df, 'actual', model)
    results.append(metrics.to_polars().with_columns(pl.lit(model).alias('model_name')))

all_metrics = pl.concat(results)
all_metrics.write_parquet("all_model_metrics.parquet")
```

### Model Monitoring

Monitor data drift and model performance over time:

```python
from analytics_store import monitoring

# Compare two populations
result = monitoring.compare_populations(
    df,
    column1="baseline_scores",
    column2="current_scores",
    alpha=0.05,
    test_type='auto'  # Automatically selects t-test or Mann-Whitney U
)

print(f"Test type: {result.test_type}")
print(f"P-value: {result.p_value:.4f}")
print(f"Effect size: {result.effect_size:.3f}")
print(f"Significant: {result.is_significant}")

# Monitor regression model drift
drift_results = monitoring.monitor_regression_drift(
    reference=reference_df,
    current=current_df,
    target_col="actual",
    predicted_col="predicted",
    feature_cols=["feature1", "feature2", "feature3"],
    psi_threshold=0.2
)

# Access feature drift metrics
print(drift_results['feature_drift'])
print(drift_results['performance_drift'])
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About

Developed by Wicks Analytics LTD 2025.
