"""
analytics_store - A Python package for data analysis and analytics using Polars.

This package provides tools for:
- Lift curve analysis
- ROC curve analysis with confidence intervals
- Regression metrics and diagnostics
- Model performance evaluation
- Data drift monitoring
- Population comparison tests

All functions use a functional API design where Polars DataFrames are passed as parameters.
"""

# Import modules for direct access
from . import model_validation
from . import validation_plots
from . import monitoring

# Import commonly used dataclasses
from .model_validation import (
    LiftResult,
    DoubleLiftResult,
    ROCResult,
    RegressionMetrics,
    BootstrapResult,
    BinaryClassificationMetrics
)

from .monitoring import PopulationTestResult

__version__ = "0.1.0"
__author__ = "Wicks Analytics LTD"

__all__ = [
    'model_validation',
    'validation_plots',
    'monitoring',
    'LiftResult',
    'DoubleLiftResult',
    'ROCResult',
    'RegressionMetrics',
    'BootstrapResult',
    'BinaryClassificationMetrics',
    'PopulationTestResult'
]
