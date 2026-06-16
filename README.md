# ⛈️ Atmospheric Decision Support System: Chi-Square & MLP Framework

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chi-square-multi-layer-perceptron.streamlit.app/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Davey117/Chi-Square-Multi-layer-Perceptron)

## 📌 Project Overview
The landscape of modern meteorology requires high-frequency digital sensing and computational intelligence. Traditional Numerical Weather Prediction (NWP) models often lack the agility required to capture localized, chaotic transitions, particularly within the Sub-Saharan climatic context. 

This repository houses a **Hybrid Meteorological Forecasting System** engineered to predict daily precipitation events. By synergizing statistical dimensionality reduction with deep connectionist learning, this framework neutralizes "learning fatigue" and maximizes predictive precision against high-dimensional atmospheric noise.

**Live Cloud Deployment:** [Access the Dashboard Here](https://chi-square-multi-layer-perceptron.streamlit.app/)

---

## ⚙️ System Architecture

The pipeline is strictly partitioned into two primary functional cores:

1. **The Statistical Architect (Chi-Square Optimization Core):**
   - Ingests raw thermodynamic, hygrometric, and aerodynamic logs.
   - Computes a mathematical association matrix ($\chi^2$) to evaluate feature independence.
   - Surgically prunes redundant sensor noise (e.g., secondary pressure indices) to isolate the top 8 high-fidelity environmental predictors, bypassing the "curse of dimensionality."

2. **The Neural Inference Engine (Multilayer Perceptron):**
   - A deep feedforward neural topology configured with hidden layers `(32, 16)`.
   - Utilizes `ReLU` non-linear activation and the `Adam` gradient optimizer.
   - Leverages backpropagation to iteratively adjust synaptic weights across the optimized subspace, outputting a highly calibrated binary prediction for `RainTomorrow`.

---

## 📊 Empirical Performance Metrics

The architecture was rigorously validated using a stratified split to mathematically protect against class imbalance bias. Standard accuracy was rejected as the primary benchmark in favor of the **Matthews Correlation Coefficient (MCC)** to ensure symmetrical predictive resilience.

| Evaluation Metric | Score | Technical Implication |
| :--- | :--- | :--- |
| **Accuracy Baseline** | `84.00%` | Robust generalization across chaotic atmospheric records. |
| **Precision (Minority Class)** | `0.7200` | High confidence; when rain is forecasted, the system is highly accurate, mitigating false alarms. |
| **Root Mean Square Error (RMSE)**| `0.3386` | Minimal predictive deviation from the observed physical environment. |
| **Matthews Correlation (MCC)** | `0.4977` | Strong structural proof of high-fidelity, balanced predictions despite severe (78/22) dataset imbalance. |

---

## 💻 Tech Stack & Dependencies

- **Algorithm Design:** `scikit-learn`, `numpy`, `pandas`
- **Model Serialization:** `joblib`
- **Frontend / Cloud Serving:** `streamlit`, `streamlit-option-menu`
- **Language:** `Python 3.10+`

---

## 🚀 Local Installation & Execution

To replicate this environment locally and verify the architectural states:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Davey117/Chi-Square-Multi-layer-Perceptron.git](https://github.com/Davey117/Chi-Square-Multi-layer-Perceptron.git)
   cd Chi-Square-Multi-layer-Perceptron
