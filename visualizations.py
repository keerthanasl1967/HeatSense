"""
Interactive visualization module for HeatSense Bengaluru using Plotly.

Adheres to modern environmental analytics design:
- Crisp teal, emerald, sky-blue, amber, and crimson color scheme.
- Clean typography and light background cards.
- High-risk highlighting with danger warnings.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Theme color palette
COLORS = {
    "teal": "#0f766e",
    "teal_light": "#14b8a6",
    "blue": "#0284c7",
    "emerald": "#059669",
    "amber": "#d97706",
    "crimson": "#dc2626",
    "gray_text": "#334155",
    "bg_white": "#ffffff",
    "grid_color": "#e2e8f0",
}

RISK_COLOR_MAP = {
    "Low": "#10b981",       # Emerald
    "Moderate": "#f59e0b",  # Amber
    "High": "#ef4444",      # Crimson / Red
    "Extreme": "#991b1b",   # Deep Maroon
}


def plot_risk_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Renders a stylized donut chart of urban heat-risk categories."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ["Risk_Category", "Count"]

    color_seq = [RISK_COLOR_MAP.get(c, COLORS["teal"]) for c in counts["Risk_Category"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["Risk_Category"],
                values=counts["Count"],
                hole=0.55,
                marker=dict(colors=color_seq, line=dict(color="#ffffff", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        title="Heat-Risk Category Distribution",
        title_font=dict(size=16, color=COLORS["gray_text"]),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def plot_weather_timeseries(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    temp_col: str = "temperature_2m",
    humidity_col: str = "relative_humidity_2m",
) -> go.Figure:
    """Renders dual-axis interactive time series of temperature and humidity."""
    fig = go.Figure()

    # Temperature trace
    fig.add_trace(
        go.Scatter(
            x=df[timestamp_col],
            y=df[temp_col],
            name="Air Temperature (°C)",
            line=dict(color="#dc2626", width=2),
            yaxis="y1",
        )
    )

    # Relative humidity trace
    fig.add_trace(
        go.Scatter(
            x=df[timestamp_col],
            y=df[humidity_col],
            name="Relative Humidity (%)",
            line=dict(color="#0284c7", width=1.8, dash="dot"),
            yaxis="y2",
            opacity=0.75,
        )
    )

    # Add IMD heat stress threshold line (35°C)
    fig.add_hline(
        y=35.0,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="35°C Heat Stress Caution",
        annotation_position="top left",
        yref="y1",
    )

    fig.update_layout(
        title="Chronological Meteorological Observations (Bengaluru)",
        title_font=dict(size=16, color=COLORS["gray_text"]),
        template="plotly_white",
        xaxis=dict(title="Timestamp", showgrid=True, gridcolor=COLORS["grid_color"]),
        yaxis=dict(
            title=dict(text="Air Temperature (°C)", font=dict(color="#dc2626")),
            tickfont=dict(color="#dc2626"),
            showgrid=True,
            gridcolor=COLORS["grid_color"],
        ),
        yaxis2=dict(
            title=dict(text="Relative Humidity (%)", font=dict(color="#0284c7")),
            tickfont=dict(color="#0284c7"),
            overlaying="y",
            side="right",
            range=[0, 105],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def plot_feature_distributions(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Renders distribution histograms for meteorological variables."""
    from plotly.subplots import make_subplots

    n = len(features)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f.replace("_", " ").title() for f in features])

    for i, f in enumerate(features):
        r = (i // cols) + 1
        c = (i % cols) + 1
        fig.add_trace(
            go.Histogram(
                x=df[f].dropna(),
                name=f,
                marker=dict(color=COLORS["teal"], line=dict(color="#ffffff", width=1)),
                opacity=0.8,
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        title="Meteorological Feature Distributions",
        title_font=dict(size=16, color=COLORS["gray_text"]),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_metrics_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Renders grouped bar chart comparing Accuracy, Macro F1, High-Risk Recall, and FNR.
    """
    metrics = ["accuracy", "f1_macro", "high_risk_recall", "high_risk_fnr"]
    metric_labels = {
        "accuracy": "Accuracy",
        "f1_macro": "Macro F1",
        "high_risk_recall": "High-Risk Recall (Critical)",
        "high_risk_fnr": "High-Risk FNR (Lower is better)",
    }

    fig = go.Figure()

    colors = [COLORS["teal"], COLORS["blue"], COLORS["emerald"], COLORS["crimson"]]

    for i, m in enumerate(metrics):
        if m in comparison_df.columns:
            # Handle possible None / N/A
            vals = [v if v is not None and not np.isnan(v) else 0.0 for v in comparison_df[m]]
            fig.add_trace(
                go.Bar(
                    x=comparison_df["Model"],
                    y=vals,
                    name=metric_labels.get(m, m),
                    marker_color=colors[i % len(colors)],
                    text=[f"{v:.3f}" if v > 0 else "N/A" for v in vals],
                    textposition="auto",
                )
            )

    fig.update_layout(
        title="Comparative Predictive & Risk Performance: Classical vs Quantum ML",
        title_font=dict(size=16, color=COLORS["gray_text"]),
        template="plotly_white",
        barmode="group",
        yaxis=dict(title="Score (0.0 to 1.0)", range=[0, 1.08], gridcolor=COLORS["grid_color"]),
        xaxis=dict(title="Model"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=50, b=60),
    )
    return fig


def plot_timing_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Renders side-by-side training duration and prediction latency comparison."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Training Time (Seconds)", "Inference Latency (ms / sample)"],
    )

    # Training time
    fig.add_trace(
        go.Bar(
            x=comparison_df["Model"],
            y=comparison_df["training_time_seconds"],
            marker_color=COLORS["blue"],
            text=[f"{t:.2f}s" for t in comparison_df["training_time_seconds"]],
            textposition="auto",
            name="Training Time",
        ),
        row=1,
        col=1,
    )

    # Latency
    fig.add_trace(
        go.Bar(
            x=comparison_df["Model"],
            y=comparison_df["latency_ms_per_sample"],
            marker_color=COLORS["amber"],
            text=[f"{l:.2f}ms" for l in comparison_df["latency_ms_per_sample"]],
            textposition="auto",
            name="Inference Latency",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Computational Resource Benchmark (Simulation vs Classical CPU)",
        title_font=dict(size=16, color=COLORS["gray_text"]),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], title: str = "Confusion Matrix") -> go.Figure:
    """Renders an interactive heatmap for the confusion matrix."""
    cm_arr = np.array(cm)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=class_names,
            y=class_names,
            colorscale=[[0, "#f8fafc"], [0.5, "#99f6e4"], [1, "#0f766e"]],
            text=cm_arr,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        title=title,
        title_font=dict(size=15, color=COLORS["gray_text"]),
        template="plotly_white",
        xaxis=dict(title="Predicted Label", tickmode="array", tickvals=list(range(len(class_names)))),
        yaxis=dict(title="True Reference Label", tickmode="array", tickvals=list(range(len(class_names))), autorange="reversed"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
