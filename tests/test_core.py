import pytest
import polars as pl
import numpy as np
from analytics_store import model_validation, monitoring

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    true_values = np.random.binomial(1, 0.3, n_samples)
    scores = true_values + np.random.normal(0, 0.2, n_samples)
    predictions = scores > 0.5
    
    # Create DataFrame
    df = pl.DataFrame({
        'target': true_values,
        'score': scores,
        'prediction': predictions
    })
    return df

def test_sample_data_creation(sample_data):
    """Test that sample data is created correctly."""
    assert sample_data is not None
    assert isinstance(sample_data, pl.DataFrame)
    assert len(sample_data) == 1000
    assert 'target' in sample_data.columns
    assert 'score' in sample_data.columns

def test_calculate_lift_curve(sample_data):
    """Test lift curve calculation."""
    result = model_validation.calculate_lift_curve(sample_data, 'target', 'score')
    
    assert result.baseline > 0
    assert result.auc_score_lift > 0
    assert len(result.percentiles) > 0
    assert len(result.score_lift_values) == len(result.percentiles)
    assert len(result.score_cumulative_lift) == len(result.percentiles)
    assert len(result.target_lift_values) == len(result.percentiles)
    assert len(result.target_cumulative_lift) == len(result.percentiles)

def test_calculate_roc_curve(sample_data):
    """Test ROC curve calculation."""
    result = model_validation.calculate_roc_curve(sample_data, 'target', 'score')
    
    assert 0 <= result.auc_score <= 1
    assert len(result.fpr) == len(result.tpr)
    assert len(result.thresholds) > 0
    assert isinstance(result.optimal_point, tuple)

def test_calculate_regression_metrics(sample_data):
    """Test regression metrics calculation."""
    result = model_validation.calculate_regression_metrics(sample_data, 'target', 'score')
    
    assert result.mse >= 0
    assert result.rmse >= 0
    assert result.mae >= 0
    assert result.mape >= 0
    assert isinstance(result.r2, float)
    assert result.n_samples == len(sample_data)

def test_calculate_double_lift(sample_data):
    """Test double lift calculation."""
    # Add second score for testing
    df_with_second_score = sample_data.with_columns([
        pl.Series('score2', sample_data['score'] + np.random.normal(0, 0.1, len(sample_data)))
    ])
    
    result = model_validation.calculate_double_lift(df_with_second_score, 'target', 'score', 'score2')
    
    assert isinstance(result.correlation, float)
    assert -1 <= result.correlation <= 1
    assert result.joint_lift > 0
    assert result.conditional_lift > 0

def test_calculate_gini(sample_data):
    """Test Gini coefficient calculation."""
    gini = model_validation.calculate_gini(sample_data, 'score')
    
    assert 0 <= gini <= 1
    assert isinstance(gini, float)

def test_calculate_lorenz_curve(sample_data):
    """Test Lorenz curve calculation."""
    x, y, gini = model_validation.calculate_lorenz_curve(sample_data, 'score')
    
    assert len(x) == len(y)
    assert x[0] == 0 and x[-1] == 1
    assert y[0] == 0 and y[-1] == 1
    assert 0 <= gini <= 1

def test_compare_populations(sample_data):
    """Test population comparison."""
    # Create two slightly different distributions
    df_with_two_scores = sample_data.with_columns([
        pl.Series('score2', sample_data['score'] + np.random.normal(0.1, 0.1, len(sample_data)))
    ])
    
    result = monitoring.compare_populations(df_with_two_scores, 'score', 'score2')
    
    assert isinstance(result.p_value, float)
    assert 0 <= result.p_value <= 1
    assert isinstance(result.effect_size, float)
    assert result.test_type in ['t', 'mann_whitney']
    assert isinstance(result.is_significant, bool)

def test_to_polars_methods(sample_data):
    """Test that all result dataclasses have working to_polars() methods."""
    # Test RegressionMetrics
    metrics = model_validation.calculate_regression_metrics(sample_data, 'target', 'score')
    metrics_df = metrics.to_polars()
    assert isinstance(metrics_df, pl.DataFrame)
    assert len(metrics_df) == 1
    assert 'rmse' in metrics_df.columns
    assert 'r2' in metrics_df.columns
    
    # Test LiftResult
    lift = model_validation.calculate_lift_curve(sample_data, 'target', 'score')
    lift_df = lift.to_polars()
    assert isinstance(lift_df, pl.DataFrame)
    assert len(lift_df) > 1  # Multiple rows for percentiles
    assert 'percentile' in lift_df.columns
    assert 'score_lift' in lift_df.columns
    
    # Test ROCResult
    roc = model_validation.calculate_roc_curve(sample_data, 'target', 'score')
    roc_df = roc.to_polars()
    assert isinstance(roc_df, pl.DataFrame)
    assert len(roc_df) > 1  # Multiple rows for thresholds
    assert 'fpr' in roc_df.columns
    assert 'tpr' in roc_df.columns
    
    # Test PopulationTestResult
    df_with_two_scores = sample_data.with_columns([
        pl.Series('score2', sample_data['score'] + np.random.normal(0.1, 0.1, len(sample_data)))
    ])
    pop_result = monitoring.compare_populations(df_with_two_scores, 'score', 'score2')
    pop_df = pop_result.to_polars()
    assert isinstance(pop_df, pl.DataFrame)
    assert len(pop_df) == 1
    assert 'p_value' in pop_df.columns
    assert 'effect_size' in pop_df.columns
