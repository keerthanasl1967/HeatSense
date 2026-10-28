# HeatSense Bengaluru 🌡️
### Classical and Quantum Machine Learning for Urban Heat-Risk Classification in Bengaluru

[![Python 3.11](https://img.shields.io/badge/Python-3.11-teal.svg)](https://www.python.org/)
[![Qiskit 2.6](https://img.shields.io/badge/Qiskit-2.6-purple.svg)](https://qiskit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-red.svg)](https://streamlit.io/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

HeatSense Bengaluru is an environmental analytics and machine learning research platform designed to investigate and benchmark classical machine learning algorithms against quantum machine learning (QML) models for classifying urban heat-risk conditions in the tropical highland metropolitan area of Bengaluru, Karnataka, India.

---

## 🎯 Research Objectives

1. **Comparative Predictive Performance:** Benchmark classical baselines (Logistic Regression, Random Forest) against a genuine Quantum Support Vector Classifier (QSVC via Qiskit's `FidelityQuantumKernel` and `ZZFeatureMap`).
2. **False-Negative Rate (FNR) Minimization:** In urban heat emergencies, failing to predict high-risk heat events ($FN$) can cause preventable heat illnesses and mortality. HeatSense evaluates models on both overall Accuracy/F1 and high-risk False-Negative Rate ($FNR = FN / (FN + TP)$).
3. **Computational Resource Assessment:** Measure wall-clock training duration, single-sample inference latency, and simulation overhead between classical CPU execution and quantum circuit simulation.
4. **Target Leakage Prevention & Temporal Integrity:** Implement strict chronological train/test partitioning to prevent temporal look-ahead bias and eliminate intermediate derived indices from model training features.

---

## 🔬 Technology Stack

- **Interface:** Streamlit (v1.35+)
- **Quantum Machine Learning:** Qiskit (v2.6+), Qiskit Machine Learning (v0.9+), Qiskit Algorithms
- **Classical Machine Learning:** Scikit-learn (v1.4+)
- **Data Processing:** Pandas (v2.2+), NumPy (v1.26+)
- **Visualizations:** Plotly (v5.20+)
- **Model Serialization:** Joblib (v1.4+)
- **Automated Testing:** Pytest (v8.0+)
- **Configuration:** PyYAML

---

## 📁 Project Structure

```
HeatSense/
├── app.py                      # Main Streamlit 5-page application entry point
├── config.yaml                 # Research experiment hyperparameters & settings
├── requirements.txt            # Locked, compatible Python dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # Complete research documentation and setup guide
├── .gitignore                  # Git ignore rules for venv, models, and cache
│
├── data/
│   ├── .gitkeep
│   └── bengaluru_heat_records.csv   # Historical hourly meteorological dataset for Bengaluru (ERA5)
│
├── models/
│   ├── .gitkeep
│   ├── preprocessor.joblib     # Fitted feature scalers, encoders, and PCA transformers
│   ├── logistic_regression.joblib  # Trained classical Logistic Regression baseline
│   ├── random_forest.joblib    # Trained classical Random Forest baseline
│   └── quantum_model.joblib    # Trained Qiskit QSVC model and circuit architecture
│
├── results/
│   ├── .gitkeep
│   └── latest_experiment.json  # Exported metrics, confusion matrices, and latency stats
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV ingestion, data validation, Open-Meteo API fetcher
│   ├── labeling.py             # NOAA Steadman, IMD, and BOM heat-risk labeling formulas
│   ├── preprocessing.py        # Leakage-free scaling, chronological splitting, quantum PCA
│   ├── classical_models.py     # Scikit-learn Logistic Regression and Random Forest trainers
│   ├── quantum_models.py       # Genuine Qiskit ZZFeatureMap circuit and QSVC classifier
│   ├── evaluation.py           # Multi-class and high-risk binary metrics (FNR, Recall, F1)
│   ├── prediction.py           # Unified inference service with physical bounds validation
│   ├── visualizations.py       # Interactive Plotly time-series, bar charts, confusion matrices
│   └── experiment_manager.py   # End-to-end pipeline orchestrator and comparison aggregator
│
└── tests/
    ├── test_data_loader.py     # Tests for CSV loading, bounds checks, missing columns
    ├── test_preprocessing.py   # Tests for scaling, chronological vs random splits
    ├── test_classical_models.py# Tests for Logistic Regression and Random Forest
    ├── test_evaluation.py      # Tests for FNR calculation and zero-division handling
    ├── test_prediction.py      # Tests for input validation and prediction service
    └── test_quantum_models.py  # Tests for Qiskit circuit initialization and QSVC simulation
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.11 (or Python 3.10+)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/HeatSense.git
cd HeatSense
```

### 3. Create a Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt / PowerShell):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
Or with custom port and host:
```bash
streamlit run app.py --server.port 3000 --server.address 0.0.0.0
```
Open your browser at `http://localhost:3000` (or `http://localhost:8501`).

---

## 🧪 Running Automated Tests

Run the full Pytest test suite (covering unit tests for all components):
```bash
pytest tests/
```
Output:
```
tests/test_classical_models.py ...                                       [ 14%]
tests/test_data_loader.py ......                                         [ 42%]
tests/test_evaluation.py ....                                            [ 61%]
tests/test_prediction.py ..                                              [ 71%]
tests/test_preprocessing.py ....                                         [ 90%]
tests/test_quantum_models.py ..                                          [100%]
======================= 21 passed in 3.85s ========================
```

---

## 📊 Dataset Format

HeatSense accepts standard CSV files with the following recommended meteorological columns:

| Column Name | Type | Physical Bounds | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | Datetime | 2000–Present | Hourly observation timestamp (ISO format: `YYYY-MM-DD HH:MM`) |
| `temperature_2m` | Float | -5.0 to 55.0 °C | Ambient 2-meter air temperature |
| `relative_humidity_2m` | Float | 0.0 to 100.0 % | Ambient relative humidity at 2 meters |
| `wind_speed_10m` | Float | 0.0 to 150.0 km/h | Horizontal wind speed at 10 meters |
| `surface_solar_radiation` | Float | 0.0 to 1400.0 W/m² | Direct and diffuse surface solar irradiance |

### Built-in Dataset
The repository includes real hourly meteorological reanalysis data for Bengaluru (`data/bengaluru_heat_records.csv`) retrieved from the Open-Meteo Historical Weather API (ERA5 Reanalysis).

---

## 🏷️ Heat-Risk Labeling Methodologies

1. **NOAA / NWS Steadman Heat Index (Default):**
   Calculates the Steadman apparent temperature via the multi-term Rothfusz regression equation:
   - **Low Risk:** Heat Index $< 30.0^\circ\text{C}$ (Normal / Caution)
   - **Moderate Risk:** $30.0^\circ\text{C} \le \text{Heat Index} < 34.0^\circ\text{C}$ (Extreme Caution)
   - **High Risk:** $\text{Heat Index} \ge 34.0^\circ\text{C}$ (Heat Stroke / Danger)
   *(Thresholds are configurable in `config.yaml` or directly within the Streamlit UI to match local Bengaluru Urban Action Plan criteria).*

2. **IMD Tropical Heat Advisory Criteria:**
   - **High Risk:** Air Temperature $\ge 35.0^\circ\text{C}$ OR ($\text{Temp} \ge 32.0^\circ\text{C}$ and $\text{RH} \ge 65\%$).
   - **Moderate Risk:** Air Temperature $\ge 32.0^\circ\text{C}$ OR ($\text{Temp} \ge 29.0^\circ\text{C}$ and $\text{RH} \ge 60\%$).
   - **Low Risk:** Below moderate criteria.

3. **Pre-existing Labels:**
   Researchers can select an existing label column in uploaded CSVs.

---

## ⚛️ Quantum Machine Learning Implementation

The quantum classifier is implemented through **Qiskit Machine Learning**:
- **Algorithm:** Quantum Support Vector Classifier (`QSVC`) utilizing `FidelityQuantumKernel`.
- **Quantum Circuit:** Second-order Pauli-Z feature map (`ZZFeatureMap`) with $N = 2$ to $4$ qubits, circuit depth reps $= 2$, and linear entangling topology.
- **Dimensionality Reduction:** Principal Component Analysis (`PCA`) projects the 4 classical meteorological dimensions onto $N$ orthogonal axes, which are scaled to $[0, 2\pi]$ for parameterized $R_z$ rotations.
- **Simulator:** Runs locally on Qiskit's `Statevector` quantum simulator without requiring cloud quantum hardware credentials.
- **Simulation Note:** Quantum kernel simulation complexity scales quadratically with dataset size ($O(N^2)$). Subsampling (50–120 training records) is applied by default to ensure manageable execution on standard laptops.

---

## 📈 Evaluation Metrics

All models are evaluated on the identical held-out test partition:
- **Accuracy:** Total correct classifications over total samples.
- **Macro F1-Score:** Unweighted mean of per-class F1-scores, sensitive to minority high-risk classes.
- **Weighted F1-Score:** Class-frequency weighted mean F1-score.
- **High-Risk Recall:** $\frac{TP_{\text{high}}}{TP_{\text{high}} + FN_{\text{high}}}$.
- **High-Risk False-Negative Rate (FNR):** $\frac{FN_{\text{high}}}{TP_{\text{high}} + FN_{\text{high}}}$.
- **Zero-Division Transparency:** When a test split contains zero high-risk events (e.g. during Bengaluru's monsoon months in a chronological split), metrics are reported as **N/A** rather than fabricated or producing a division-by-zero error.

---

## ⚠️ Limitations & Ethical Disclaimers

1. **Experimental Classification Only:** Outputs generated by HeatSense are experimental meteorological research models. They do not constitute emergency medical diagnoses, clinical assessments, or official municipal heat-wave warnings.
2. **Quantum Simulator Scope:** The current quantum implementation uses classical CPU simulation of ideal statevectors. It does not account for physical quantum hardware noise (decoherence, gate infidelities, or readout errors) present in Noisy Intermediate-Scale Quantum (NISQ) devices.
3. **Microclimate Resolution:** Regional reanalysis models (ERA5) provide grid-level averages (~9–25 km resolution). Urban localized hot-spots (e.g. unshaded asphalt corridors) may experience higher surface temperatures than station observations.

---

## 📄 License
This project is licensed under the Apache License 2.0.
