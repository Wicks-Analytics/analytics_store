# AnalyticsStore

A powerful Python package for data analysis and model evaluation using Polars. Built for high performance and ease of use, with a focus on lift analysis and model performance metrics.

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

## Installation

```bash
pip install https://github.com/Wicks-Analytics/analytics_store
```

## Quick Start

```python
from analytics_store import DataAnalyser

# Initialize analyzer and load data
analyzer = DataAnalyser()
analyzer.load_data("model_results.csv")

# Basic lift curve analysis
lift_result = analyzer.calculate_lift_curve(
    target_column="actual_values",
    score_column="model_scores",
    n_bins=10
)
analyzer.plot_lift_curve(
    target_column="actual_values",
    score_column="model_scores",
    title="Model Lift Analysis"
)

# Compare two models using double lift
analyzer.plot_double_lift(
    target_column="actual_values",
    score1_column="model1_scores",
    score2_column="model2_scores",
    title="Model Comparison"
)

# ROC curve analysis with confidence intervals
analyzer.plot_roc_curve(
    target_column="actual_values",
    score_column="model_scores",
    with_ci=True,
    n_iterations=1000,
    confidence_level=0.95
)

# Regression metrics and diagnostics
metrics = analyzer.calculate_regression_metrics(
    actual_column="actual_values",
    predicted_column="predicted_values",
    n_features=10  # Optional, for adjusted R-squared
)
print(f"RMSE: {metrics.rmse:.3f}")
print(f"R-squared: {metrics.r2:.3f}")

# Plot regression diagnostics
analyzer.plot_regression_diagnostics(
    actual_column="actual_values",
    predicted_column="predicted_values",
    title="Model Regression Diagnostics"
)
```

## Detailed Documentation

### Lift Analysis

The package provides comprehensive lift analysis capabilities:

```python
# Calculate lift metrics
lift_result = analyzer.calculate_lift_curve(
    target_column="actual_values",
    score_column="model_scores"
)

print(f"Baseline rate: {lift_result.baseline:.3f}")
print(f"AUC Lift: {lift_result.auc_lift:.3f}")

# Access lift values
print("Point-wise lift values:", lift_result.lift_values)
print("Cumulative lift:", lift_result.cumulative_lift)
```

### Double Lift Analysis

Compare two scoring variables:

```python
results = analyzer.calculate_double_lift(
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
roc_result = analyzer.calculate_roc_curve(
    target_column="actual_values",
    score_column="predicted_probabilities"
)

print(f"AUC: {roc_result.auc_score:.3f}")
print(f"Optimal threshold: {roc_result.optimal_threshold:.3f}")
```

### Regression Analysis

Evaluate regression model performance:

```python
# Get comprehensive regression metrics
metrics = analyzer.calculate_regression_metrics(
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
analyzer.plot_regression_diagnostics(
    actual_column="actual_values",
    predicted_column="predicted_values",
    title="Regression Model Diagnostics"
)
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About

Developed by Wicks Analytics LTD 2025.
