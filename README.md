# 🧬 AI-Driven AMR Biosafety & Novelty Detection System (Research Grade)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-FF4B4B)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.50.0-orange)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A **Research-Grade** analytical pipeline for **Antimicrobial Resistance (AMR)** surveillance, utilizing advanced Machine Learning (**XGBoost**), Explainable AI (**SHAP**), Statistical Trend Analysis (**Mann-Kendall**), and Generative AI (**CTGAN**) for synthetic data augmentation.

Designed for **clinical researchers**, **public health teams**, and **LMIC biosafety environments** to bridge the gap between raw surveillance data and actionable policy insights.

---

## 🚀 Key Features

### 1. 🤖 Advanced ML Predictions (XGBoost)
- **State-of-the-art Accuracy**: Uses **XGBoost Regressor** (Gradient Boosting) instead of traditional Random Forest.
- **Optimized Pipeline**: Automated One-Hot Encoding and feature scaling via `scikit-learn` pipelines.
- **Metrics**: Real-time calculation of **R² Score** and **RMSE** (Root Mean Squared Error).

### 2. 🧠 Explainable AI (SHAP)
- **Black Box no more**: Integrates **SHAP (SHapley Additive exPlanations)** to interpret model decisions.
- **Feature Importance**: Visualizes *why* the model predicts high resistance for specific pathogen-drug combinations (e.g., "Is it the location or the year driving the risk?").

### 3. 📉 Statistical Rigor (Mann-Kendall Test)
- **Trend Validation**: Goes beyond visual lines. Uses the **Mann-Kendall Trend Test** to statistically validate increasing/decreasing resistance trends.
- **P-Values**: Filters and highlights only statistically significant trends ($p < 0.05$).

### 4. 🧪 Synthetic Lab (Generative AI)
- **Data Augmentation**: Solves the "Data Scarcity" problem in LMICs using **CTGAN (Conditional Tabular GAN)**.
- **Privacy-Preserving**: Generates realistic, synthetic surveillance data that mimics the statistical properties of real data without compromising patient privacy.
- **Downloadable**: Users can generate and download 1,000+ synthetic samples for research.

### 5. ⚠️ Multidimensional Risk Scoring
- **Composite Index**: Calculates a dynamic risk score based on:
  $$ Risk = (Predicted Rate \times 0.6) + (Trend Slope \times 0.4) + Penalty $$
- **Prioritization**: Automatically ranks the top 10 highest-risk pathogen-antibiotic combinations for immediate intervention.

---

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/GillStorm/AI-Driven-AMR-Biosafety-Dashboard.git
    cd AI-Driven-AMR-Biosafety-Dashboard
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Requires `xgboost<2.0.0` for SHAP compatibility.*

3.  **Run the Application**
    ```bash
    streamlit run biosafety.py
    ```

---

## 📊 Usage Guide

### 1. Upload Data
- Use the sidebar to upload a CSV file.
- **Required Columns**: `location`, `year`, `pathogen`, `antibiotic`, `n_tested`, `n_resistant`.
- *Sample data (`who_glass_2022_processed.csv`) is included in the repo.*

### 2. Dashboard Tabs
- **📊 Dashboard & Trends**: View global resistance rates, time-series plots, and heatmaps.
- **🤖 AI Analysis**: Train the XGBoost model and view SHAP explanation plots.
- **⚠️ Risk & Novelty**: See the calculated Risk Scores and detect statistical/genomic anomalies.
- **🧪 Synthetic Lab**: Train a GAN on your data and generate new synthetic samples.

---

## 📐 Methodology

### Machine Learning Pipeline
The system employs a rigorous ML pipeline:
1.  **Preprocessing**: `ColumnTransformer` handles categorical variables (Location, Pathogen, Antibiotic) via One-Hot Encoding.
2.  **Modeling**: `XGBRegressor` with `n_estimators=200`, `learning_rate=0.05`, `max_depth=6`.
3.  **Validation**: 80/20 Train-Test split with `random_state=42` for reproducibility.

### Statistical Tests
- **Mann-Kendall**: Non-parametric test for monotonic trends.
  - $H_0$: No trend exists.
  - $H_a$: Monotonic trend exists.
  - Significance level: $\alpha = 0.05$.

### Novelty Detection
- **Statistical**: Z-Score analysis ($Z > 2.5$) to detect resistance spikes.
- **Isolation Forest**: Unsupervised anomaly detection for multidimensional outliers.

---

## 📂 Project Structure

```
AI-Driven-AMR-Biosafety-Dashboard/
├── biosafety.py                # Main Streamlit Application
├── preprocess_glass.py         # Script to clean WHO GLASS data
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── who_glass_2022.csv          # Raw WHO GLASS dataset
├── who_glass_2022_processed.csv# Cleaned dataset for analysis
└── amr_data.csv                # Legacy sample data
```

---

## 📚 Research Contribution

This project contributes to the field of **Digital Epidemiology** by:
1.  Providing a **deployable, open-source tool** for LMIC surveillance.
2.  Demonstrating the use of **GenAI (GANs)** to address data scarcity in public health.
3.  Integrating **Explainable AI** into biosafety decision-making.

---

*Built with ❤️ for Global Health Security.*
