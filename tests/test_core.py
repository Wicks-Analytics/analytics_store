import pytest
import polars as pl
import numpy as np
from analytics_store.core import DataAnalyser

@pytest.fixture
def analyzer():
    """Create a DataAnalyser instance for testing."""
    return DataAnalyser()

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

def test_load_data(analyzer, sample_data):
    """Test loading data into analyzer."""
    analyzer.load_data(sample_data)
    assert analyzer.data is not None
    assert isinstance(analyzer.data, pl.DataFrame)
    assert len(analyzer.data) == len(sample_data)

def test_calculate_lift_curve(analyzer, sample_data):
    """Test lift curve calculation."""
    analyzer.load_data(sample_data)
    result = analyzer.calculate_lift_curve('target', 'score')
    
    assert result.baseline > 0
    assert result.auc_lift > 0
    assert len(result.percentiles) > 0
    assert len(result.lift_values) == len(result.percentiles)
    assert len(result.cumulative_lift) == len(result.percentiles)

def test_calculate_roc_curve(analyzer, sample_data):
    """Test ROC curve calculation."""
    analyzer.load_data(sample_data)
    result = analyzer.calculate_roc_curve('target', 'score')
    
    assert 0 <= result.auc_score <= 1
    assert len(result.fpr) == len(result.tpr)
    assert len(result.thresholds) > 0
    assert isinstance(result.optimal_point, tuple)

def test_calculate_regression_metrics(analyzer, sample_data):
    """Test regression metrics calculation."""
    analyzer.load_data(sample_data)
    result = analyzer.calculate_regression_metrics('target', 'score')
    
    assert result.rmse >= 0
    assert result.mae >= 0
    assert isinstance(result.r2, float)
    assert result.n_samples == len(sample_data)

def test_calculate_double_lift(analyzer, sample_data):
    """Test double lift calculation."""
    # Add second score for testing
    df_with_second_score = sample_data.with_columns([
        pl.Series('score2', sample_data['score'] + np.random.normal(0, 0.1, len(sample_data)))
    ])
    
    analyzer.load_data(df_with_second_score)
    result = analyzer.calculate_double_lift('target', 'score', 'score2')
    
    assert isinstance(result.correlation, float)
    assert -1 <= result.correlation <= 1
    assert result.joint_lift > 0
    assert result.conditional_lift > 0
