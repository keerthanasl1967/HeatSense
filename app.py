"""
HeatSense Bengaluru
Classical and Quantum Machine Learning for Urban Heat-Risk Classification in Bengaluru

Main Streamlit Application.
"""

import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.classical_models import ClassicalModelTrainer
from src.data_loader import fetch_open_meteo_historical, load_dataset
from src.evaluation import compute_metrics, load_results, save_results
from src.experiment_manager import ExperimentManager
from src.labeling import apply_heat_risk_labeling
from src.prediction import PredictionService, RESEARCH_DISCLAIMER
from src.preprocessing import PreprocessingPipeline
from src.quantum_models import QISKIT_AVAILABLE, QuantumClassifierTrainer
from src.visualizations import (
    COLORS,
    plot_confusion_matrix,
    plot_feature_distributions,
    plot_metrics_comparison,
    plot_risk_distribution,
    plot_timing_comparison,
    plot_weather_timeseries,
)

# Page configuration
st.set_page_config(
    page_title="HeatSense Bengaluru | Classical & Quantum ML",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Teal, Green, Sky-Blue palette, white cards, high readability)
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .research-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .disclaimer-banner {
        background-color: #f0fdfa;
        border-left: 4px solid #0d9488;
        padding: 0.85rem 1.15rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        color: #0f766e;
        font-size: 0.88rem;
    }
    .badge-classical {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-quantum {
        background-color: #f3e8ff;
        color: #7e22ce;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_experiment_manager() -> ExperimentManager:
    em = ExperimentManager()
    em.load_config()
    # Try loading previous experiment results if present
    saved = load_results("results/latest_experiment.json")
    if saved:
        em.results = saved
        em.is_completed = True
    return em


@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    path = "data/bengaluru_heat_records.csv"
    if os.path.exists(path):
        df, _ = load_dataset(path)
        return df
    return pd.DataFrame()


def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            <span style="font-size:26px;">🌡️</span>
            <div>
                <h3 style="margin:0; padding:0; color:#0f766e; line-height:1.2;">HeatSense</h3>
                <small style="color:#64748b;">Bengaluru Urban Heat Analytics</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home Dashboard",
            "Heat-Risk Prediction",
            "Classical vs Quantum ML Comparison",
            "Historical Weather Analysis",
            "Dataset and Methodology",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Study Context")
    st.sidebar.caption(
        "**Location:** Bengaluru, India\n\n"
        "**Coordinates:** 12.9716°N, 77.5946°E\n\n"
        "**Elevation:** ~920 m (Deccan Plateau)\n\n"
        "**Climate:** Tropical Savanna / Urban Island"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("HeatSense Research Applet v1.0 • Qiskit 2.6 • Scikit-Learn 1.9")

    return page


# ---------------------------------------------------------
# PAGE A: HOME DASHBOARD
# ---------------------------------------------------------
def render_home_page(em: ExperimentManager, df_data: pd.DataFrame):
    st.title("HeatSense Bengaluru")
    st.subheader("Classical and Quantum Machine Learning for Urban Heat-Risk Classification")

    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>Study Objective:</strong> Demonstrating and benchmarking classical machine learning 
            (Logistic Regression, Random Forest) and genuine quantum machine learning (QSVC / Quantum Kernel) 
            for classifying urban heat stress in Bengaluru using meteorological observations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    total_records = len(df_data) if not df_data.empty else 0
    date_min = df_data["timestamp"].min() if ("timestamp" in df_data and not df_data.empty) else "N/A"
    date_max = df_data["timestamp"].max() if ("timestamp" in df_data and not df_data.empty) else "N/A"

    col1.metric("Weather Records", f"{total_records:,}")
    col2.metric("Study Area", "Bengaluru, India", "12.97°N, 77.59°E")
    col3.metric("Models Active", "3 Tested", "2 Classical + 1 Qiskit")
    col4.metric("Temporal Range", "2023 Summer & Monsoon", "Hourly ERA5")

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 🌆 Urban Heat Island (UHI) in Bengaluru")
        st.write(
            """
            Bengaluru has experienced exponential urban expansion over the past three decades, leading to 
            substantial loss of green tree canopy, depletion of historic lake systems, and dense concrete construction. 
            These changes create localized **Urban Heat Islands (UHI)**, where daytime microclimate temperatures 
            spike significantly higher than surrounding rural outskirts.
            
            When combined with high relative humidity and intense solar radiation, heat dissipation slows down, 
            amplifying human heat-stress risks for vulnerable outdoor workers, commuters, and high-density settlements.
            """
        )

        st.markdown("### 🚀 Quick Navigation")
        c_a, c_b, c_c = st.columns(3)
        if c_a.button("🔮 Predict Heat Risk", use_container_width=True):
            st.session_state["nav_page"] = "Heat-Risk Prediction"
            st.rerun()
        if c_b.button("⚖️ Compare ML vs QML", use_container_width=True):
            st.session_state["nav_page"] = "Classical vs Quantum ML Comparison"
            st.rerun()
        if c_c.button("📈 Historical Analysis", use_container_width=True):
            st.session_state["nav_page"] = "Historical Weather Analysis"
            st.rerun()

    with col_right:
        if not df_data.empty and "temperature_2m" in df_data.columns:
            # Apply labeling for preview chart
            df_preview, _ = apply_heat_risk_labeling(
                df_data,
                method="heat_index_noaa",
                moderate_threshold=30.0,
                high_threshold=34.0,
            )
            fig_pie = plot_risk_distribution(df_preview, "heat_risk_category")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Historical dataset not loaded yet.")

    st.markdown("---")
    st.markdown("### 📋 Latest Meteorological Observation")
    if not df_data.empty:
        latest = df_data.iloc[-1]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Timestamp", str(latest.get("timestamp", "N/A"))[:16])
        m2.metric("Air Temp", f"{latest.get('temperature_2m', 0.0):.1f} °C")
        m3.metric("Rel Humidity", f"{latest.get('relative_humidity_2m', 0.0):.0f} %")
        m4.metric("Wind Speed", f"{latest.get('wind_speed_10m', 0.0):.1f} km/h")
        m5.metric("Solar Radiation", f"{latest.get('surface_solar_radiation', 0.0):.0f} W/m²")
        st.caption("Data source: Open-Meteo ERA5 Reanalysis for Bengaluru Coordinates (12.9716°N, 77.5946°E).")


# ---------------------------------------------------------
# PAGE B: HEAT-RISK PREDICTION
# ---------------------------------------------------------
def render_prediction_page(em: ExperimentManager, df_data: pd.DataFrame):
    st.title("Heat-Risk Prediction")
    st.write(
        "Enter ambient meteorological parameters or select historical observations "
        "to classify urban heat-risk levels using trained classical and quantum classifiers."
    )

    st.markdown(
        f"""
        <div class="disclaimer-banner">
            <strong>Research Notice:</strong> {RESEARCH_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if models are trained and saved
    prep_path = "models/preprocessor.joblib"
    rf_path = "models/random_forest.joblib"
    lr_path = "models/logistic_regression.joblib"
    q_path = "models/quantum_model.joblib"

    models_ready = os.path.exists(prep_path) and (os.path.exists(rf_path) or os.path.exists(lr_path))

    if not models_ready:
        st.warning("⚠️ Machine learning models are not yet trained. Please train the models first.")
        if st.button("Train Models Now in Comparison Hub"):
            st.session_state["nav_page"] = "Classical vs Quantum ML Comparison"
            st.rerun()
        return

    # Load fitted preprocessor and trainers
    preprocessor = PreprocessingPipeline.load(prep_path)
    classical_models = {}
    if os.path.exists(lr_path):
        classical_models["Logistic Regression"] = ClassicalModelTrainer.load(lr_path)
    if os.path.exists(rf_path):
        classical_models["Random Forest"] = ClassicalModelTrainer.load(rf_path)

    quantum_trainer = None
    if os.path.exists(q_path):
        try:
            quantum_trainer = QuantumClassifierTrainer.load(q_path)
        except Exception:
            quantum_trainer = None

    pred_service = PredictionService(
        preprocessor=preprocessor,
        classical_models=classical_models,
        quantum_model=quantum_trainer,
    )

    # Input method selection
    input_method = st.radio(
        "Select Input Mode:",
        ["Manual Measurement Input", "Sample from Historical Bengaluru Records"],
        horizontal=True,
    )

    temp_val = 33.5
    humidity_val = 55.0
    wind_val = 12.0
    solar_val = 650.0

    if input_method == "Sample from Historical Bengaluru Records" and not df_data.empty:
        st.markdown("#### Pick a Historical Record")
        sample_indices = [
            f"Record #{i} ({df_data.iloc[i].get('timestamp', 'N/A')}) - {df_data.iloc[i].get('temperature_2m', 0.0):.1f}°C, {df_data.iloc[i].get('relative_humidity_2m', 0.0):.0f}% RH"
            for i in range(min(50, len(df_data)))
        ]
        chosen = st.selectbox("Historical Timestamp Record:", sample_indices)
        idx = int(chosen.split("#")[1].split(" ")[0])
        rec = df_data.iloc[idx]
        temp_val = float(rec.get("temperature_2m", 32.0))
        humidity_val = float(rec.get("relative_humidity_2m", 50.0))
        wind_val = float(rec.get("wind_speed_10m", 10.0))
        solar_val = float(rec.get("surface_solar_radiation", 500.0))

    # Form inputs
    with st.form("heat_risk_form"):
        st.markdown("#### Ambient Weather Measurements")
        col1, col2 = st.columns(2)
        with col1:
            in_temp = st.number_input(
                "Air Temperature (°C)",
                min_value=5.0,
                max_value=50.0,
                value=float(temp_val),
                step=0.5,
                help="Dry-bulb air temperature at 2 meters.",
            )
            in_humidity = st.number_input(
                "Relative Humidity (%)",
                min_value=5.0,
                max_value=100.0,
                value=float(humidity_val),
                step=1.0,
                help="Ambient relative humidity at 2 meters.",
            )
        with col2:
            in_wind = st.number_input(
                "Wind Speed (km/h)",
                min_value=0.0,
                max_value=100.0,
                value=float(wind_val),
                step=0.5,
                help="Surface wind velocity at 10 meters.",
            )
            in_solar = st.number_input(
                "Surface Solar Radiation (W/m²)",
                min_value=0.0,
                max_value=1400.0,
                value=float(solar_val),
                step=10.0,
                help="Direct and diffuse solar irradiance reaching urban ground level.",
            )

        submit = st.form_submit_button("⚡ Run Multi-Model Risk Inference", use_container_width=True)

    if submit:
        inputs = {
            "temperature_2m": in_temp,
            "relative_humidity_2m": in_humidity,
            "wind_speed_10m": in_wind,
            "surface_solar_radiation": in_solar,
        }

        with st.spinner("Executing classical and quantum classifiers..."):
            res = pred_service.predict(inputs)

        if not res["success"]:
            st.error(f"Prediction input error: {', '.join(res.get('errors', []))}")
            return

        st.markdown("### 📊 Model Predictions & Risk Diagnostics")
        preds = res["predictions"]

        cols = st.columns(len(preds))
        for idx, (model_name, p_data) in enumerate(preds.items()):
            with cols[idx]:
                is_quantum = "Quantum" in p_data.get("model_family", "")
                badge_class = "badge-quantum" if is_quantum else "badge-classical"

                st.markdown(
                    f"""
                    <div class="research-card">
                        <span class="{badge_class}">{p_data.get('model_family', 'Model')}</span>
                        <h4 style="margin:8px 0 4px 0;">{model_name}</h4>
                    """,
                    unsafe_allow_html=True,
                )

                if p_data.get("status") == "Success":
                    cat = p_data.get("predicted_category", "Unknown")
                    cat_color = COLORS["crimson"] if cat == "High" else (COLORS["amber"] if cat == "Moderate" else COLORS["emerald"])

                    st.markdown(
                        f"""
                        <div style="font-size:24px; font-weight:700; color:{cat_color}; margin:8px 0;">
                            {cat} Risk
                        </div>
                        <p style="margin:0; font-size:13px; color:#64748b;">
                            Latency: <strong>{p_data.get('latency_ms', 0)} ms</strong>
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Show probabilities if present
                    dist = p_data.get("confidence_distribution")
                    if dist:
                        st.markdown("<small style='color:#475569;'>Class Probabilities:</small>", unsafe_allow_html=True)
                        for c_name, prob in dist.items():
                            st.progress(float(prob), text=f"{c_name}: {prob*100:.1f}%")
                    elif is_quantum:
                        st.caption(f"Kernel mapping on {p_data.get('circuit_qubits', 4)} simulated qubits.")
                else:
                    st.info(p_data.get("status", "Not available"))

                st.markdown("</div>", unsafe_allow_html=True)

        st.caption(res["disclaimer"])


# ---------------------------------------------------------
# PAGE C: CLASSICAL VS QUANTUM ML COMPARISON
# ---------------------------------------------------------
def render_comparison_page(em: ExperimentManager, df_data: pd.DataFrame):
    st.title("Classical vs Quantum Machine Learning Comparison")
    st.markdown(
        "**Core Research Experiment:** Comparing predictive accuracy, high-risk recall, "
        "false-negative rates, and computational overhead between classical baselines "
        "and genuine Qiskit quantum classifiers."
    )

    with st.expander("⚙️ Experiment Controls & Hyperparameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            split_choice = st.selectbox(
                "Train/Test Split Strategy:",
                ["chronological", "random"],
                index=0,
                help="Chronological avoids future data leaking into the past. Random allows high-risk cases in test split if limited to one season.",
            )
            label_choice = st.selectbox(
                "Heat-Risk Standard:",
                ["heat_index_noaa", "imd_criteria", "apparent_temp_wbgt"],
                index=0,
            )
        with col2:
            mod_thresh = st.slider("Moderate Risk Threshold (°C)", 26.0, 34.0, 30.0, 0.5)
            high_thresh = st.slider("High Risk Threshold (°C)", 32.0, 42.0, 34.0, 0.5)
        with col3:
            enable_quantum = st.checkbox("Train Quantum Model (QSVC)", value=True)
            qubits = st.selectbox("Quantum Qubits (Feature Map dim):", [2, 3, 4], index=2)
            q_train_samples = st.slider("Max Quantum Training Samples", 30, 200, 80, 10, help="Simulation time scales polynomially with sample size.")

        start_training = st.button("🚀 Run Comparative Benchmark Experiment", type="primary", use_container_width=True)

    if start_training:
        with st.status("Executing HeatSense Research Pipeline...", expanded=True) as status:
            progress_placeholder = st.empty()

            def progress_cb(msg: str):
                progress_placeholder.write(f"🔄 {msg}")

            try:
                # Update config
                em.config["split"]["strategy"] = split_choice
                results = em.run_pipeline(
                    dataset_path="data/bengaluru_heat_records.csv",
                    labeling_method=label_choice,
                    moderate_threshold=mod_thresh,
                    high_threshold=high_thresh,
                    train_quantum=enable_quantum,
                    quantum_qubits=qubits,
                    quantum_train_samples=q_train_samples,
                    progress_callback=progress_cb,
                )
                status.update(label="✅ Comparative Experiment Complete!", state="complete", expanded=False)
                st.success("All models trained and evaluated on held-out test records!")
            except Exception as e:
                status.update(label="❌ Experiment Error", state="error")
                st.error(f"Experiment execution encountered an issue: {str(e)}")

    # Load and display comparison results
    comp_df = em.get_comparison_table()

    if not comp_df.empty:
        st.markdown("### 📋 Unified Evaluation Benchmark Table")
        st.caption(
            "All models evaluated under consistent test definitions. "
            "High-risk false negative rate (FNR = FN / (FN + TP)) measures critical under-prediction of heat emergencies."
        )

        display_cols = [
            "Model",
            "Accuracy",
            "Precision (Macro)",
            "Recall (Macro)",
            "Macro F1",
            "Weighted F1",
            "High-Risk Recall",
            "High-Risk FNR",
            "Training Time (s)",
            "Latency (ms/sample)",
            "Test Samples",
        ]
        st.dataframe(comp_df[display_cols], use_container_width=True, hide_index=True)

        # Download CSV button
        csv_buffer = io.StringIO()
        comp_df[display_cols].to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Experiment Benchmark CSV",
            data=csv_buffer.getvalue(),
            file_name="heatsense_bengaluru_model_comparison.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("### 📊 Interactive Comparative Visualizations")

        c1, c2 = st.columns(2)
        with c1:
            fig_perf = plot_metrics_comparison(comp_df)
            st.plotly_chart(fig_perf, use_container_width=True)

        with c2:
            fig_timing = plot_timing_comparison(comp_df)
            st.plotly_chart(fig_timing, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔲 Confusion Matrices (Held-Out Test Set)")

        models_data = em.results.get("models", {})
        cm_cols = st.columns(len([m for m in models_data.values() if "confusion_matrix" in m]))
        col_idx = 0

        for model_name, m_info in models_data.items():
            if "confusion_matrix" in m_info:
                cm = m_info["confusion_matrix"]
                class_names = m_info.get("class_names", ["Low", "Moderate", "High"])
                with cm_cols[col_idx]:
                    fig_cm = plot_confusion_matrix(cm, class_names, title=f"{model_name}")
                    st.plotly_chart(fig_cm, use_container_width=True)
                col_idx += 1

        # Quantum Circuit Deep-Dive
        if "QSVC (Quantum Kernel)" in models_data and "circuit_info" in models_data["QSVC (Quantum Kernel)"]:
            q_info = models_data["QSVC (Quantum Kernel)"]["circuit_info"]
            st.markdown("---")
            st.markdown("### ⚛️ Quantum Circuit Architecture (Qiskit ZZFeatureMap)")

            qc1, qc2, qc3, qc4 = st.columns(4)
            qc1.metric("Qubits Simulated", q_info.get("num_qubits", qubits))
            qc2.metric("Circuit Depth", q_info.get("circuit_depth", "N/A"))
            qc3.metric("Entangling Gates (CX)", q_info.get("gate_count", {}).get("cx", 0))
            qc4.metric("Phase Rotations (P)", q_info.get("gate_count", {}).get("p", 0))

            with st.expander("View Qiskit Quantum Feature Map ASCII Circuit Diagram"):
                st.code(q_info.get("ascii_diagram", "Circuit diagram not available."), language="text")

            sim_note = models_data["QSVC (Quantum Kernel)"].get("simulation_note")
            if sim_note:
                st.info(f"ℹ️ **Simulation Context:** {sim_note}")

    else:
        st.info("No experiment has been run yet. Click 'Run Comparative Benchmark Experiment' above to initiate training.")


# ---------------------------------------------------------
# PAGE D: HISTORICAL WEATHER ANALYSIS
# ---------------------------------------------------------
def render_historical_page(df_data: pd.DataFrame):
    st.title("Historical Weather Analysis")
    st.write(
        "Inspect meteorological distributions, validate time-series integrity, "
        "and explore urban heat patterns in Bengaluru."
    )

    # Option to upload custom CSV or use default
    st.markdown("#### 📂 Dataset Selection")
    col_up1, col_up2 = st.columns([1.5, 1])

    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Custom Meteorological CSV (Optional):",
            type=["csv"],
            help="Upload a CSV with timestamp, temperature, humidity, wind, and solar radiation columns.",
        )

    with col_up2:
        st.write("Or Fetch Fresh Historical Records from Open-Meteo:")
        if st.button("🌐 Refresh 2023 Bengaluru Records via API"):
            with st.spinner("Contacting Open-Meteo Historical Weather API..."):
                new_df, err = fetch_open_meteo_historical()
                if err:
                    st.error(err)
                elif new_df is not None:
                    new_df.to_csv("data/bengaluru_heat_records.csv", index=False)
                    st.success(f"Successfully downloaded {len(new_df)} hourly records!")
                    st.rerun()

    active_df = df_data
    if uploaded_file is not None:
        active_df, val_report = load_dataset(uploaded_file)
        if not val_report["success"]:
            st.error(f"Uploaded CSV validation failed: {', '.join(val_report['errors'])}")
            return
        st.success(f"Loaded custom dataset: {len(active_df)} valid records.")
        if val_report.get("warnings"):
            for w in val_report["warnings"]:
                st.warning(w)

    if active_df.empty:
        st.info("No dataset currently loaded.")
        return

    st.markdown("---")
    st.markdown("### 📋 Dataset Preview & Column Integrity")
    st.dataframe(active_df.head(10), use_container_width=True)

    # Date range filtering if timestamp parsed
    if "timestamp" in active_df.columns:
        active_df["timestamp"] = pd.to_datetime(active_df["timestamp"], errors="coerce")
        active_df = active_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        t_min = active_df["timestamp"].min()
        t_max = active_df["timestamp"].max()

        st.markdown("### 📅 Temporal Filtering")
        date_range = st.date_input(
            "Filter Observation Date Range:",
            value=(t_min.date(), t_max.date()),
            min_value=t_min.date(),
            max_value=t_max.date(),
        )

        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
            filtered_df = active_df[(active_df["timestamp"] >= start_dt) & (active_df["timestamp"] <= end_dt)]
        else:
            filtered_df = active_df
    else:
        filtered_df = active_df

    st.markdown("---")
    st.markdown("### 📈 Time-Series Meteorological Dynamics")
    fig_ts = plot_weather_timeseries(filtered_df)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Variable Distributions")
    feat_cols = [c for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_solar_radiation"] if c in filtered_df.columns]
    if feat_cols:
        fig_dist = plot_feature_distributions(filtered_df, feat_cols)
        st.plotly_chart(fig_dist, use_container_width=True)


# ---------------------------------------------------------
# PAGE E: DATASET AND METHODOLOGY
# ---------------------------------------------------------
def render_methodology_page():
    st.title("Dataset & Research Methodology")
    st.markdown(
        """
        ### 1. Research Overview & Study Location
        - **Study Location:** Bengaluru, Karnataka, India (Coordinates: 12.9716° N, 77.5946° E; Altitude: ~920 m a.s.l.).
        - **Objective:** Benchmarking classical machine learning against quantum kernel classifiers (QSVC) 
          for meteorological urban heat-risk classification.
        - **Data Source:** Open-Meteo Historical Weather API powered by ECMWF ERA5 reanalysis and validated 
          against Bengaluru surface meteorological observations.
        
        ---
        ### 2. Meteorological Variables & Target Leakage Prevention
        The model inputs consist strictly of raw ambient variables:
        - `temperature_2m` (°C): 2-meter air temperature.
        - `relative_humidity_2m` (%): Ambient relative humidity.
        - `wind_speed_10m` (km/h): 10-meter horizontal wind speed.
        - `surface_solar_radiation` (W/m²): Direct normal and diffuse solar irradiance.
        
        **Target Leakage Safeguard:** If heat risk is categorized based on Steadman Heat Index or Apparent Temperature, 
        those derived intermediate formulas are **strictly excluded** from model features. 
        Models must learn the multi-variable non-linear interactions directly from raw weather signals.

        ---
        ### 3. Chronological Train/Test Partitioning
        Weather records have strong temporal autocorrelation and seasonal shifts. 
        Random shuffling leaks future temperature trends into past predictions, producing unrealistically inflated 
        metrics. A strict **chronological split** (past 75% for training, future 25% for held-out evaluation) 
        faithfully reflects real-world deployment.

        ---
        ### 4. Quantum Machine Learning Architecture (Qiskit 2.x)
        - **Model:** Quantum Support Vector Classifier (QSVC) using `FidelityQuantumKernel`.
        - **Feature Map:** Second-order Pauli-Z expansion (`ZZFeatureMap`), encoding classical features 
          into qubit rotation angles $\\theta \\in [0, 2\\pi]$ and entangling gates ($CX$):
          $$\\mathcal{U}_{\\Phi(x)} = \\exp\\left(i \\sum_{j,k} (\\pi - x_j)(\\pi - x_k) Z_j Z_k\\right) \\prod_j H_j R_z(2x_j) H_j$$
        - **Dimensionality Reduction:** Principal Component Analysis (PCA) maps the 4-dimensional 
          meteorological feature space into $N=2$ to $4$ orthogonal quantum channels.
        - **Simulation Execution:** Executed on Qiskit's Statevector backend. Due to $O(N^2)$ kernel evaluation 
          complexity on classical CPUs, quantum models are trained on representative subsamples (50–120 records).

        ---
        ### 5. Research Metrics & False-Negative Rate
        In urban environmental risk governance, failing to detect an extreme heat event (False Negative) 
        has severe public health consequences. We therefore emphasize:
        - **High-Risk Recall:** $\\frac{TP}{TP + FN}$ for dangerous heat conditions.
        - **High-Risk False-Negative Rate:** $\\frac{FN}{FN + TP} = 1 - \\text{Recall}_{\\text{high}}$.
        - **Zero-Division Handling:** When no high-risk events occur in the held-out evaluation window, 
          the metric is explicitly denoted as **N/A** rather than fabricated.
        """
    )


# ---------------------------------------------------------
# MAIN APP ROUTER
# ---------------------------------------------------------
def main():
    em = get_experiment_manager()
    df_data = get_cached_dataset()

    # Handle programmatic navigation requests
    if "nav_page" in st.session_state:
        page = st.session_state.pop("nav_page")
    else:
        page = render_sidebar()

    if page == "Home Dashboard":
        render_home_page(em, df_data)
    elif page == "Heat-Risk Prediction":
        render_prediction_page(em, df_data)
    elif page == "Classical vs Quantum ML Comparison":
        render_comparison_page(em, df_data)
    elif page == "Historical Weather Analysis":
        render_historical_page(df_data)
    elif page == "Dataset and Methodology":
        render_methodology_page()


if __name__ == "__main__":
    main()
