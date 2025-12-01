# AI-Driven-AMR-Biosafety-Dashboard
A research-grade analytical pipeline for Antimicrobial Resistance (AMR) surveillance, trend analysis, and machine-learning prediction, designed for clinical researchers, public health teams, and LMIC biosafety environments.

This project automatically processes AMR surveillance CSV files to generate:

📊 Resistance trends

🦠 Pathogen–drug resistance profiles

🤖 Machine learning predictions

🔥 AMR risk scoring

🗺️ Hotspot identification

📈 Interactive dashboards

🚨 Problem Statement

Antimicrobial resistance is rising globally, especially in low-resource settings where antimicrobial misuse, poor diagnostics, and limited surveillance accelerate resistant infections. Existing AMR datasets are fragmented, inconsistent, imbalanced, and difficult to analyze.

Although machine learning can support AMR prediction, adoption is limited because:

AMR datasets suffer from class imbalance, missing values, and low quality

Resistance patterns vary across regions — models lack generalizability

There is no easy-to-use tool that integrates ML and AMR surveillance

This project solves these challenges by building a complete, automated AMR analytics system that combines data cleaning, feature engineering, ML modeling, and biosafety visualization in a single research-ready dashboard.

📐 Methodology

The system follows an 8-stage research pipeline:

1. Data Acquisition

Users upload CSV files with columns:

location, year, pathogen, antibiotic, n_tested, n_resistant

2. Data Preprocessing

Remove invalid/duplicate rows

Handle missing values

Encode categorical variables

Fix class imbalance via SMOTE

Standardize formats (pathogen/antibiotic normalization)

3. Feature Engineering

Compute resistance percentage (n_resistant/n_tested × 100)

Yearly trends

Pathogen–antibiotic interaction features

Hotspot identification metrics

4. Exploratory Data Analysis (EDA)

Automatic generation of:

Resistance trend lines

Heatmaps

Country/pathogen comparisons

Resistant vs susceptible distributions

5. Machine Learning Model Training

Models supported:

Logistic Regression

Random Forest

SVM

Decision Trees

K-Means for clustering

Training framework:

80/20 stratified split

Cross-validation

Hyperparameter tuning

6. Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-score

Balanced Accuracy

AUC-ROC

7. Risk Scoring Engine

AMR severity labeled as:

Low (<20%)

Moderate (20–50%)

High (50–70%)

Critical (>70%)

8. Dashboard Deployment

Interactive Streamlit dashboard:

Upload CSV

View trends & heatmaps

See model predictions

Export analysis output

📚 Research Contribution

This project contributes:

A unified AMR surveillance and ML prediction framework

Support for LMIC surveillance datasets

A pipeline addressing class imbalance, data quality issues, and low generalizability

A deployable biosafety dashboard

Explainable ML for antimicrobial stewardship
