from cProfile import label
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Union, List, Tuple, Dict, Any, Literal
from dataclasses import dataclass
from sklearn.metrics import (auc, roc_curve, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, 
                         precision_score, recall_score, f1_score, accuracy_score, confusion_matrix,
                         precision_recall_curve)

# Import validation functions
from .model_validation import (
    calculate_lorenz_curve, bootstrap_gini, calculate_lift_curve,
    calculate_roc_curve, bootstrap_roc_curve, calculate_double_lift,
    calculate_regression_metrics
)


def _get_backend_from_config(backend: Optional[str], config: Optional[Dict[str, Any]]) -> str:
    """Helper function to determine plotting backend.
    
    Priority: explicit backend parameter > config > default (matplotlib)
    
    Args:
        backend: Explicit backend parameter (if provided)
        config: Config dictionary that may contain "plotting_backend"
        
    Returns:
        Backend string ("matplotlib" or "plotly")
    """
    if backend is not None:
        return backend
    if config is not None and "plotting_backend" in config:
        return config["plotting_backend"]
    return "matplotlib"


# ============================================================================
# Lorenz Curve and Gini Coefficient Functions
# ============================================================================
def plot_lorenz_curve(df: pl.DataFrame, column: str, exposure_column: Optional[str] = None, 
                      title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6),
                      backend: Optional[Literal["matplotlib", "plotly"]] = None,
                      config: Optional[Dict[str, Any]] = None) -> Union[plt.Figure, go.Figure]:
    """Plot the Lorenz curve for a given column.
    
    Args:
        df: Polars DataFrame
        column: Name of the numeric column to plot Lorenz curve for
        exposure_column: Optional name of the column containing exposure/weight values
        title: Optional title for the plot
        figsize: Tuple of (width, height) for the plot
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from
        
    Raises:
        ValueError: If columns don't exist or contain non-numeric data
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)
    
    # Calculate Lorenz curve coordinates
    x, y, gini = calculate_lorenz_curve(df, column, exposure_column)
    
    if backend == "plotly":
        # Create plotly figure
        fig = go.Figure()
        
        # Add Lorenz curve
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Lorenz Curve',
            line=dict(color='blue', width=2)
        ))
        
        # Add line of perfect equality
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Line of Perfect Equality',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        # Update layout
        plot_title = title if title else f'Lorenz Curve for {column}<br>Gini Coefficient: {gini:.3f}'
        fig.update_layout(
            title=plot_title,
            xaxis_title='Cumulative Proportion of Population',
            yaxis_title=f'Cumulative Proportion of {column}',
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        # Add annotation with Gini coefficient
        fig.add_annotation(
            text=f'Gini Coefficient: {gini:.3f}',
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor="wheat",
            opacity=0.8
        )
        
        return fig
    else:
        # Matplotlib backend
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

def plot_lorenz_curve_with_ci(df: pl.DataFrame, column: str, exposure_column: Optional[str] = None,
                              n_iterations: int = 1000, confidence_level: float = 0.95,
                              title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6),
                              random_seed: Optional[int] = None,
                              backend: Optional[Literal["matplotlib", "plotly"]] = None,
                              config: Optional[Dict[str, Any]] = None) -> Union[plt.Figure, go.Figure]:
    """Plot Lorenz curve with bootstrapped confidence intervals.
    
    Args:
        df: Polars DataFrame
        column: Name of the numeric column to plot Lorenz curve for
        exposure_column: Optional name of the column containing exposure/weight values
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level for the intervals (e.g., 0.95 for 95% CI)
        title: Optional title for the plot
        figsize: Tuple of (width, height) for the plot
        random_seed: Optional random seed for reproducibility
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)
    
    # Calculate Lorenz curve with confidence intervals
    x, y, gini, lower_bound, upper_bound, gini_ci = bootstrap_gini(
        df, column, exposure_column, n_iterations, confidence_level, random_seed
    )
    
    if backend == "plotly":
        # Create plotly figure
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level*100:.0f}% Confidence Interval',
            showlegend=True
        ))
        
        # Add Lorenz curve
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Lorenz Curve',
            line=dict(color='blue', width=2)
        ))
        
        # Add line of perfect equality
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Line of Perfect Equality',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        # Update layout
        plot_title = title if title else f'Lorenz Curve for {column}<br>Gini Coefficient: {gini:.3f} [{gini_ci.lower_bound:.3f}, {gini_ci.upper_bound:.3f}]'
        fig.update_layout(
            title=plot_title,
            xaxis_title='Cumulative Proportion of Population',
            yaxis_title=f'Cumulative Proportion of {column}',
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        # Add annotation with Gini coefficient and CI
        fig.add_annotation(
            text=f'Gini Coefficient: {gini:.3f}<br>95% CI: [{gini_ci.lower_bound:.3f}, {gini_ci.upper_bound:.3f}]',
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor="wheat",
            opacity=0.8
        )
        
        return fig
    else:
        # Matplotlib backend
        plt.style.use('seaborn-v0_8-dark')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the main Lorenz curve
        ax.plot(x, y, 'b-', label='Lorenz Curve', linewidth=2)
        ax.plot([0, 1], [0, 1], 'r--', label='Line of Perfect Equality', linewidth=1)
        
        # Plot confidence intervals
        ax.fill_between(x, lower_bound, upper_bound, alpha=0.2, color='blue',
                       label=f'{confidence_level*100:.0f}% Confidence Interval')
        
        # Add labels and title
        ax.set_xlabel('Cumulative Proportion of Population')
        ax.set_ylabel(f'Cumulative Proportion of {column}')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Lorenz Curve for {column}\nGini Coefficient: {gini:.3f} [{gini_ci.lower_bound:.3f}, {gini_ci.upper_bound:.3f}]')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set aspect ratio to make the plot square
        ax.set_aspect('equal')
        
        # Add text box with Gini coefficient and CI
        textstr = (f'Gini Coefficient: {gini:.3f}\n'
                  f'95% CI: [{gini_ci.lower_bound:.3f}, {gini_ci.upper_bound:.3f}]')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig

def plot_lift_curve(df: pl.DataFrame, target_column: str, score_column: str, n_bins: int = 10,
                    title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6),
                    backend: Optional[Literal["matplotlib", "plotly"]] = None,
                    config: Optional[Dict[str, Any]] = None) -> Union[plt.Figure, go.Figure]:
    """Plot lift curve showing both point-wise and cumulative lift.
    
    Args:
        df: Polars DataFrame
        target_column: Name of the binary target column (0/1)
        score_column: Name of the score/probability column
        n_bins: Number of bins to divide the data into
        title: Optional title for the plot
        figsize: Tuple of (width, height) for the plot
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)
    
    # Calculate lift curve
    lift_result = calculate_lift_curve(df, target_column, score_column, n_bins)
    
    if backend == "plotly":
        # Create plotly figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Point-wise Lift by Percentile', 'Cumulative Lift by Percentile')
        )
        
        # Add point-wise lift traces
        fig.add_trace(
            go.Scatter(x=lift_result.percentiles, y=lift_result.score_lift_values,
                      mode='lines+markers', name='Predicted', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=lift_result.percentiles, y=lift_result.target_lift_values,
                      mode='lines+markers', name='Actual', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=1)
        
        # Add cumulative lift traces
        fig.add_trace(
            go.Scatter(x=lift_result.percentiles, y=lift_result.score_cumulative_lift,
                      mode='lines+markers', name='Predicted', line=dict(color='red'),
                      showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=lift_result.percentiles, y=lift_result.target_cumulative_lift,
                      mode='lines+markers', name='Actual', line=dict(color='blue'),
                      showlegend=False),
            row=1, col=2
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=2)
        
        # Update axes
        fig.update_xaxes(title_text="Percentile", row=1, col=1)
        fig.update_xaxes(title_text="Percentile", row=1, col=2)
        fig.update_yaxes(title_text="Lift", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Lift", row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title_text=title if title else "Lift Curve Analysis",
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            showlegend=True
        )
        
        # Add annotation with metrics
        fig.add_annotation(
            text=f'Baseline: {lift_result.baseline:.3f}<br>AUC Lift: {lift_result.auc_score_lift:.3f}',
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="wheat",
            opacity=0.8,
            xanchor='left',
            yanchor='top'
        )
        
        return fig
    else:
        # Matplotlib backend
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

def plot_roc_curve(df: pl.DataFrame, target_column: str, score_column: str,
                   with_ci: bool = True, n_iterations: int = 1000,
                   confidence_level: float = 0.95, title: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 8),
                   random_seed: Optional[int] = None,
                   backend: Optional[Literal["matplotlib", "plotly"]] = None,
                   config: Optional[Dict[str, Any]] = None) -> Union[plt.Figure, go.Figure]:
    """Plot ROC curve optionally with bootstrap confidence intervals.
    
    Args:
        df: Polars DataFrame
        target_column: Name of the column containing actual binary values
        score_column: Name of the column containing model scores
        with_ci: Whether to include bootstrap confidence intervals
        n_iterations: Number of bootstrap iterations if with_ci is True
        confidence_level: Confidence level for intervals if with_ci is True
        title: Optional title for the plot
        figsize: Tuple of (width, height) for the plot
        random_seed: Optional random seed for reproducibility
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from
        
    Raises:
        ValueError: If parameters are invalid or data is missing
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)
    
    if with_ci:
        # Get ROC curve with confidence intervals
        roc_result, ci_lower, ci_upper, fpr_points = bootstrap_roc_curve(
            df, target_column, score_column, n_iterations, confidence_level, random_seed
        )
    else:
        # Just calculate basic ROC curve
        roc_result = calculate_roc_curve(df, target_column, score_column)
        ci_lower = ci_upper = fpr_points = None
    
    if backend == "plotly":
        # Create plotly figure
        fig = go.Figure()
        
        # Add confidence interval if requested
        if with_ci:
            fig.add_trace(go.Scatter(
                x=np.concatenate([fpr_points, fpr_points[::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level*100:.0f}% Confidence Interval',
                showlegend=True
            ))
        
        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=roc_result.fpr, y=roc_result.tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_result.auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        # Add optimal point
        fig.add_trace(go.Scatter(
            x=[roc_result.optimal_point[0]], y=[roc_result.optimal_point[1]],
            mode='markers',
            name=f'Optimal Point (threshold = {roc_result.optimal_threshold:.3f})',
            marker=dict(color='green', size=10)
        ))
        
        # Update layout
        plot_title = title if title else 'ROC Curve Analysis'
        fig.update_layout(
            title=plot_title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        # Add annotation with metrics
        fig.add_annotation(
            text=f'AUC: {roc_result.auc_score:.3f}<br>Optimal Threshold: {roc_result.optimal_threshold:.3f}<br>Optimal Point FPR: {roc_result.optimal_point[0]:.3f}<br>Optimal Point TPR: {roc_result.optimal_point[1]:.3f}',
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor="wheat",
            opacity=0.8,
            xanchor='left',
            yanchor='top'
        )
        
        return fig
    else:
        # Matplotlib backend
        plt.style.use('seaborn-v0_8-dark')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot confidence intervals if requested
        if with_ci:
            ax.fill_between(fpr_points, ci_lower, ci_upper, color='blue', alpha=0.2,
                          label=f'{confidence_level*100:.0f}% Confidence Interval')
        
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

def plot_double_lift(df: pl.DataFrame, target_column: str, score1_column: str, score2_column: str,
                     n_bins: int = 10, title: Optional[str] = None, 
                     figsize: Tuple[int, int] = (15, 5),
                     backend: Optional[Literal["matplotlib", "plotly"]] = None,
                     config: Optional[Dict[str, Any]] = None) -> Union[plt.Figure, go.Figure]:
    """Plot double lift curves comparing two models.
    
    Args:
        df: Polars DataFrame
        target_column: Name of the binary target column (0/1)
        score1_column: Name of the first score/probability column
        score2_column: Name of the second score/probability column
        n_bins: Number of bins to divide the data into
        title: Optional title for the plot
        figsize: Tuple of (width, height) for the plot
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)
    
    # Calculate double lift
    result = calculate_double_lift(df, target_column, score1_column, score2_column, n_bins)
    
    if backend == "plotly":
        # Create plotly figure with subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Point-wise Lift Comparison', 'Cumulative Lift Comparison', 'Score Correlation')
        )
        
        # Plot 1: Point-wise lift
        fig.add_trace(
            go.Scatter(x=result.lift1.percentiles, y=result.lift1.score_lift_values,
                      mode='lines+markers', name=score1_column, line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=result.lift2.percentiles, y=result.lift2.score_lift_values,
                      mode='lines+markers', name=score2_column, line=dict(color='green'),
                      marker=dict(symbol='square')),
            row=1, col=1
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=1)
        
        # Plot 2: Cumulative lift
        fig.add_trace(
            go.Scatter(x=result.lift1.percentiles, y=result.lift1.score_cumulative_lift,
                      mode='lines+markers', name=score1_column, line=dict(color='blue'),
                      showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=result.lift2.percentiles, y=result.lift2.score_cumulative_lift,
                      mode='lines+markers', name=score2_column, line=dict(color='green'),
                      marker=dict(symbol='square'), showlegend=False),
            row=1, col=2
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=2)
        
        # Plot 3: Scatter plot
        fig.add_trace(
            go.Scatter(x=df[score1_column], y=df[score2_column], mode='markers',
                      marker=dict(size=5, opacity=0.5), showlegend=False),
            row=1, col=3
        )
        
        # Update axes
        fig.update_xaxes(title_text="Percentile", row=1, col=1)
        fig.update_xaxes(title_text="Percentile", row=1, col=2)
        fig.update_xaxes(title_text=score1_column, row=1, col=3)
        fig.update_yaxes(title_text="Lift", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Lift", row=1, col=2)
        fig.update_yaxes(title_text=score2_column, row=1, col=3)
        
        # Update layout
        fig.update_layout(
            title_text=title if title else "Double Lift Analysis",
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            showlegend=True
        )
        
        # Add annotation with metrics
        fig.add_annotation(
            text=f'Correlation: {result.correlation:.3f}<br>Joint Lift: {result.joint_lift:.3f}<br>Conditional Lift: {result.conditional_lift:.3f}<br><br>AUC Lift 1: {result.lift1.auc_score_lift:.3f}<br>AUC Lift 2: {result.lift2.auc_score_lift:.3f}',
            xref="paper", yref="paper",
            x=0.68, y=0.95,
            showarrow=False,
            bgcolor="wheat",
            opacity=0.8,
            xanchor='left',
            yanchor='top'
        )
        
        return fig
    else:
        # Matplotlib backend
        plt.style.use('seaborn-v0_8-dark')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Compare point-wise lift curves
        ax1.plot(result.lift1.percentiles, result.lift1.score_lift_values, 'b-', 
                marker='o', label=score1_column)
        ax1.plot(result.lift2.percentiles, result.lift2.score_lift_values, 'g-',
                marker='s', label=score2_column)
        ax1.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax1.set_xlabel('Percentile')
        ax1.set_ylabel('Lift')
        ax1.set_title('Point-wise Lift Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Compare cumulative lift curves
        ax2.plot(result.lift1.percentiles, result.lift1.score_cumulative_lift, 'b-',
                marker='o', label=score1_column)
        ax2.plot(result.lift2.percentiles, result.lift2.score_cumulative_lift, 'g-',
                marker='s', label=score2_column)
        ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('Cumulative Lift')
        ax2.set_title('Cumulative Lift Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Score correlation scatter plot
        ax3.scatter(df[score1_column], df[score2_column], alpha=0.5, s=20)
        ax3.set_xlabel(score1_column)
        ax3.set_ylabel(score2_column)
        ax3.set_title('Score Correlation')
        ax3.grid(True, alpha=0.3)
        
        # Add text box with metrics
        textstr = (f'Correlation: {result.correlation:.3f}\n'
                  f'Joint Lift: {result.joint_lift:.3f}\n'
                  f'Conditional Lift: {result.conditional_lift:.3f}\n\n'
                  f'AUC Lift 1: {result.lift1.auc_score_lift:.3f}\n'
                  f'AUC Lift 2: {result.lift2.auc_score_lift:.3f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, y=1.05)
        
        plt.tight_layout()
        return fig

def plot_regression_diagnostics(df: pl.DataFrame, actual_column: str, predicted_column: str,
                               title: Optional[str] = None, figsize: Tuple[int, int] = (12, 8),
                               backend: Optional[Literal["matplotlib", "plotly"]] = None,
                               config: Optional[Dict[str, Any]] = None) -> Union[plt.Figure, go.Figure]:
    """Create a comprehensive set of regression diagnostic plots.
    
    Creates a 2x2 panel of plots:
    1. Actual vs Predicted scatter plot
    2. Residuals vs Predicted values
    3. Residual histogram
    4. Q-Q plot of residuals
    
    Args:
        df: Polars DataFrame
        actual_column: Name of the column containing actual values
        predicted_column: Name of the column containing predicted values
        title: Optional title for the plot
        figsize: Tuple of (width, height) for the plot
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)
    
    if actual_column not in df.columns:
        raise ValueError(f"Column '{actual_column}' not found in data")
        
    if predicted_column not in df.columns:
        raise ValueError(f"Column '{predicted_column}' not found in data")
        
    # Get data as numpy arrays
    data_df = df.select([
        pl.col(actual_column).alias('actual'),
        pl.col(predicted_column).alias('predicted')
    ]).drop_nulls()
    
    actual = data_df.select('actual').to_numpy().flatten()
    predicted = data_df.select('predicted').to_numpy().flatten()
    
    # Calculate residuals
    residuals = actual - predicted
    
    # Calculate metrics
    metrics = calculate_regression_metrics(df, actual_column, predicted_column)
    
    if backend == "plotly":
        from scipy import stats as scipy_stats
        
        # Create Q-Q plot data
        qq_result = scipy_stats.probplot(residuals, dist="norm")
        theoretical_quantiles = qq_result[0][0]
        ordered_values = qq_result[0][1]
        slope = qq_result[1][0]
        intercept = qq_result[1][1]
        fit_line = slope * theoretical_quantiles + intercept
        
        # Create plotly figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted', 'Residuals vs Predicted', 
                          'Residual Distribution', 'Normal Q-Q Plot')
        )
        
        # 1. Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=predicted, y=actual, mode='markers',
                      marker=dict(size=5, opacity=0.5), showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[actual.min(), actual.max()], y=[actual.min(), actual.max()],
                      mode='lines', line=dict(color='red', dash='dash'),
                      showlegend=False),
            row=1, col=1
        )
        
        # 2. Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=predicted, y=residuals, mode='markers',
                      marker=dict(size=5, opacity=0.5), showlegend=False),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Residual histogram
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, showlegend=False),
            row=2, col=1
        )
        
        # 4. Q-Q plot
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=ordered_values, mode='markers',
                      marker=dict(size=5), showlegend=False, name='Data'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=fit_line, mode='lines',
                      line=dict(color='red'), showlegend=False, name='Fit'),
            row=2, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
        fig.update_xaxes(title_text="Residual Value", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Ordered Values", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text=title if title else "Regression Diagnostics",
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            showlegend=False
        )
        
        # Add annotation with metrics
        fig.add_annotation(
            text=f'RMSE: {metrics.rmse:.2f}<br>MAE: {metrics.mae:.2f}<br>R²: {metrics.r2:.3f}',
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="wheat",
            opacity=0.8,
            xanchor='left',
            yanchor='top'
        )
        
        return fig
    else:
        # Matplotlib backend
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
        textstr = (f'RMSE: {metrics.rmse:.2f}\n'
                  f'MAE: {metrics.mae:.2f}\n'
                  f'R²: {metrics.r2:.3f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, textstr, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig

def plot_actual_vs_expected_by_factor(df: pl.DataFrame, actual_column: str, predicted_column: str, factor_column: str,
                                     exposure_column: Optional[str] = None, title: Optional[str] = None, 
                                     figsize: Optional[Tuple[int, int]] = None,
                                     n_bins: int = 20,
                                     backend: Literal["matplotlib", "plotly"] = "matplotlib") -> Union[plt.Figure, go.Figure]:
    """Create an actual vs expected plot grouped by a factor on the x-axis.
    
    Args:
        df: Polars DataFrame
        actual_column: Name of the column containing actual values
        predicted_column: Name of the column containing predicted values
        factor_column: Name of the column containing the categorical factor to group by (x-axis)
        exposure_column: Optional column to show as bar chart on secondary y-axis
        title: Optional title for the plot
        figsize: Optional tuple of (width, height) for the plot
        n_bins: Optional number of bins to split numeric factor columns into
        backend: Plotting backend to use ("matplotlib" or "plotly")
        
    Raises:
        ValueError: If columns don't exist or contain invalid data
    """
    if backend == "plotly":
        raise NotImplementedError("Plotly backend is not yet implemented for plot_actual_vs_expected_by_factor. Please use backend='matplotlib'.")
    
    required_cols = [actual_column, predicted_column, factor_column]
    if exposure_column is not None:
        required_cols.append(exposure_column)
        
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")
    
    # Get data and drop nulls
    select_cols = [
        pl.col(actual_column).alias('actual'),
        pl.col(predicted_column).alias('predicted'),
        pl.col(factor_column).alias('factor')
    ]
    if exposure_column is not None:
        select_cols.append(pl.col(exposure_column).alias('exposure'))
        
    data_df = df.select(select_cols).drop_nulls()
    
    # Check if factor is numeric and needs binning
    is_numeric = data_df.select(pl.col('factor')).dtypes[0] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    unique_factors = data_df.select('factor').unique().height
    needs_binning = is_numeric and unique_factors > n_bins
    
    if needs_binning:
        # Create bins for numeric factors
        quantiles = np.linspace(0, 1, n_bins + 1)
        
        # Get quantile values
        factor_quantiles = data_df.select(pl.col('factor').quantile(quantiles)).row(0)
        
        # Add bin column using qcut for equal-sized bins
        data_df = data_df.with_columns([
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
            
        stats = data_df.group_by('factor_bin').agg(agg_exprs).sort('factor_bin')
        x_labels = stats.select('factor_bin').to_numpy().flatten()
    else:
        # Calculate statistics by original factor
        agg_exprs = [
            pl.col('actual').mean().alias('actual_mean'),
            pl.col('predicted').mean().alias('predicted_mean'),
            pl.col('actual').count().alias('count')
        ]
        if exposure_column is not None:
            agg_exprs.append(pl.col('exposure').sum().alias('exposure_sum'))
            
        stats = data_df.group_by('factor').agg(agg_exprs).sort('factor')
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
    metrics = calculate_regression_metrics(df, actual_column, predicted_column)
    
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

def plot_residual_ratios(df: pl.DataFrame, actual_col: str, predicted_col: str, factor_col: str,
                        group_col: Optional[str] = None, rebase_means: bool = False,
                        n_bins: int = 20,
                        title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6),
                        backend: Literal["matplotlib", "plotly"] = "matplotlib") -> Union[plt.Figure, go.Figure]:
    """Plot the residual ratio (actual/predicted) by a factor with separate lines for each group.
    
    Args:
        df: Polars DataFrame
        actual_col: Name of the column containing actual values
        predicted_col: Name of the column containing predicted values
        factor_col: Name of the column to plot on x-axis
        group_col: Optional name of the column to use for grouping (different lines)
        rebase_means: If True, divide each group's ratios by their mean to center them at 1.0
        n_bins: Number of bins for numeric factors
        title: Optional title for the plot
        figsize: Tuple of (width, height) for the plot
        backend: Plotting backend to use ("matplotlib" or "plotly")
        
    Raises:
        ValueError: If required columns don't exist or contain invalid data
    """
    if backend == "plotly":
        raise NotImplementedError("Plotly backend is not yet implemented for plot_residual_ratios. Please use backend='matplotlib'.")
    
    required_cols = [actual_col, predicted_col, factor_col]
    if group_col is not None:
        required_cols.append(group_col)
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing one or more required columns: {required_cols}")
        
    # Select required columns
    select_cols = [
        pl.col(actual_col).alias('actual'),
        pl.col(predicted_col).alias('predicted'),
        pl.col(factor_col).alias('factor')
    ]
    if group_col is not None:
        select_cols.append(pl.col(group_col).alias('group'))
    data_df = df.select(select_cols).drop_nulls()
    
    # Check if factor is numeric and needs binning
    is_numeric = data_df.select(pl.col('factor')).dtypes[0] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    unique_factors = data_df.select('factor').unique().height
    needs_binning = is_numeric and unique_factors > n_bins
    
    if needs_binning:
        # Create bins for numeric factors
        quantiles = np.linspace(0, 1, n_bins + 1)
                                
        # Add bin column using qcut for equal-sized bins
        data_df = data_df.with_columns([
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
    data_df = data_df.with_columns([
        (pl.col('actual') / pl.col('predicted')).alias('ratio')
    ])
    
    if rebase_means and group_col is not None:
        # Calculate mean ratio for each group and divide through
        group_means = data_df.group_by('group').agg(
            pl.col('ratio').mean().alias('group_mean')
        )
        data_df = data_df.join(group_means, on='group')
        data_df = data_df.with_columns([
            (pl.col('ratio') / pl.col('group_mean')).alias('ratio')
        ])
    
    # Calculate mean ratios by factor and optionally group
    group_cols = [plot_factor]
    if group_col is not None:
        group_cols.append('group')
    stats = data_df.group_by(group_cols).agg([
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
