"""
AnalyticsStore - A Python package for data analysis and analytics using Polars.

This package provides tools for:
- Lift curve analysis
- ROC curve analysis with confidence intervals
- Regression metrics and diagnostics
- Model performance evaluation
"""

from .core import (
    DataAnalyzer,
    LiftResult,
    DoubleLiftResult,
    ROCResult,
    RegressionMetrics,
    BootstrapResult
)

__version__ = "0.1.0"
__author__ = "Wicks Analytics LTD"

__all__ = [
    'DataAnalyzer',
    'LiftResult',
    'DoubleLiftResult',
    'ROCResult',
    'RegressionMetrics',
    'BootstrapResult'
]
