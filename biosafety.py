import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap
import pymannkendall as mk
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ---------------------------------------------------------
# BASIC PAGE CONFIG (no external animations -> more stable)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AMR Biosafety Dashboard",
    page_icon="🦠",
    layout="wide",
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🧬 AI-Driven AMR Biosafety & Novelty Detection System (Research Grade)")
st.write(
    "Upload AMR surveillance data to get: "
    "1) resistance trends, 2) ML predictions, 3) novelty / anomaly detection, "
    "and 4) biosafety risk scores."
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("📂 Upload Your AMR Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

show_plots = st.sidebar.checkbox("Show Visualizations", True, help="Trends & heatmaps")
show_risk = st.sidebar.checkbox("Show Multidimensional Risk Scores", True, help="Composite risk scoring")
show_novelty = st.sidebar.checkbox("Show Novelty Detection", True, help="Spike/novelty alerts")
show_shap = st.sidebar.checkbox("Show SHAP Explanations", True, help="Explainable AI for model predictions")
show_stats = st.sidebar.checkbox("Show Statistical Trends", True, help="Mann-Kendall Trend Tests")
show_synthetic = st.sidebar.checkbox("🧪 Show Synthetic Lab (GANs)", False, help="Generate synthetic data using CTGAN")

st.sidebar.markdown("---")
st.sidebar.info("Expected columns: location, year, pathogen, antibiotic, n_tested, n_resistant (+ optional gene).")

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
if uploaded_file is not None:

    # ---------- LOAD DATA ----------
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ---------- CHECK REQUIRED COLUMNS ----------
    required_cols = ["location", "year", "pathogen", "antibiotic", "n_tested", "n_resistant"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        st.stop()

    # ---------- BASIC CLEANING ----------
    # Enforce numeric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["n_tested"] = pd.to_numeric(df["n_tested"], errors="coerce")
    df["n_resistant"] = pd.to_numeric(df["n_resistant"], errors="coerce")

    # Drop invalid numeric rows
    df = df.dropna(subset=["year", "n_tested", "n_resistant"])
    df = df[df["n_tested"] > 0]

    # Compute resistance rate
    df["resistance_rate"] = df["n_resistant"] / df["n_tested"]
    df = df[(df["resistance_rate"] >= 0) & (df["resistance_rate"] <= 1)]

    if df.empty:
        st.error("All rows were invalid after cleaning (check n_tested/n_resistant).")
        st.stop()

    st.subheader("⚙️ Processed Data")
    st.dataframe(df.head())

    # ---------------------------------------------------------
    # MACHINE LEARNING MODEL (XGBoost)
    # ---------------------------------------------------------
    st.subheader("🤖 AMR Prediction Model (XGBoost)")

    X = df[["location", "year", "pathogen", "antibiotic"]].copy()
    y = df["resistance_rate"].copy()

    # Preprocessing: one-hot encode categorical, passthrough year
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["location", "pathogen", "antibiotic"]),
            ("num", "passthrough", ["year"]),
        ]
    )

    # XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42,
        n_jobs=-1
    )
    
    # Pipeline
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    # Guard: need at least 5 rows to split for ML
    if len(df) < 5:
        st.warning("Not enough rows (<5) for train/test split. Showing basic stats only.")
        pipeline.fit(X, y)
        y_pred_full = pipeline.predict(X)
        st.write("Mean predicted resistance rate:", float(y_pred_full.mean()))
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("📈 R² Score", f"{r2:.3f}")
        with c2:
            st.metric("📉 RMSE", f"{rmse:.3f}")

    # ---------------------------------------------------------
    # SHAP EXPLANATIONS
    # ---------------------------------------------------------
    if show_shap:
        st.subheader("🧠 Explainable AI (SHAP)")
        st.write("Explaining why the model predicts high/low resistance.")
        
        # We need the transformed feature names for SHAP
        # Fit preprocessor first to get feature names
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Get feature names
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["location", "pathogen", "antibiotic"])
        feature_names = list(cat_features) + ["year"]
        
        # Create a dataframe for SHAP
        X_shap_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Explain the model
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        shap_values = explainer(X_shap_df)
        
        # Summary Plot
        st.write("#### Feature Importance (SHAP Summary)")
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values, X_shap_df, show=False)
        st.pyplot(fig_shap)

    # ---------------------------------------------------------
    # STATISTICAL TRENDS (Mann-Kendall)
    # ---------------------------------------------------------
    if show_stats:
        st.subheader("📉 Statistical Trend Analysis (Mann-Kendall)")
        
        trend_results = []
        
        # Group by location, pathogen, antibiotic
        grouped = df.groupby(["location", "pathogen", "antibiotic"])
        
        for name, group in grouped:
            if len(group) >= 4: # Need at least 4 points for a meaningful trend
                group = group.sort_values("year")
                try:
                    mk_result = mk.original_test(group["resistance_rate"])
                    trend_results.append({
                        "location": name[0],
                        "pathogen": name[1],
                        "antibiotic": name[2],
                        "trend": mk_result.trend,
                        "p_value": mk_result.p,
                        "slope": mk_result.slope
                    })
                except:
                    pass
        
        if trend_results:
            trend_df = pd.DataFrame(trend_results)
            st.write("#### Significant Trends (p < 0.05)")
            sig_trends = trend_df[trend_df["p_value"] < 0.05]
            st.dataframe(sig_trends)
            
            # Merge slope back to df for risk score
            df = df.merge(trend_df[["location", "pathogen", "antibiotic", "slope"]], 
                          on=["location", "pathogen", "antibiotic"], 
                          how="left")
            df["slope"] = df["slope"].fillna(0)
        else:
            st.info("Not enough data points per group for Mann-Kendall test.")
            df["slope"] = 0

    # ---------------------------------------------------------
    # MULTIDIMENSIONAL RISK SCORE
    # ---------------------------------------------------------
    if show_risk:
        st.subheader("⚠️ Multidimensional Biosafety Risk Score")
        st.write("Risk = (Predicted Rate * 0.6) + (Trend Slope * 0.4) + Penalty")

        # Predict for all rows (full dataset)
        try:
            df["predicted_rate"] = pipeline.predict(X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
            
        if "slope" not in df.columns:
             df["slope"] = 0

        # Normalize slope (simple min-max or sigmoid could be better, but linear for now)
        # Slope is usually small (e.g. 0.05 per year). We scale it up.
        
        # Formula: 
        # Base Risk: Predicted Rate (0-1) * 100
        # Trend Bonus: Slope * 500 (e.g., 0.1 slope -> +50 risk)
        # Clip to 0-100
        
        df["base_risk"] = df["predicted_rate"] * 100
        df["trend_risk"] = df["slope"] * 500
        
        df["risk_score"] = (df["base_risk"] + df["trend_risk"]).clip(0, 100)

        st.write("Sample of risk-scored rows:")
        st.dataframe(
            df[["location", "year", "pathogen", "antibiotic", "risk_score", "slope"]].head()
        )

        high_risk = df.sort_values("risk_score", ascending=False).head(10)
        st.write("### 🔥 Top 10 Highest-Risk Combinations")
        st.table(high_risk[["location", "year", "pathogen", "antibiotic", "risk_score", "slope"]])

    # ---------------------------------------------------------
    # NOVELTY / ANOMALY DETECTION
    # ---------------------------------------------------------
    if show_novelty:

        st.subheader("🧪 Novelty & Anomaly Detection")

        # ---------- 1) STATISTICAL NOVELTY (Z-SCORE) ----------
        st.markdown("#### 1️⃣ Statistical Novelty (Z-score spikes)")
        if df["resistance_rate"].std() == 0 or len(df) < 3:
            st.info("Not enough variability for statistical anomaly detection.")
        else:
            df["z_score"] = (
                (df["resistance_rate"] - df["resistance_rate"].mean())
                / df["resistance_rate"].std()
            )
            df["statistical_novelty"] = df["z_score"].abs() > 2.5
            novel_stats = df[df["statistical_novelty"]]

            if not novel_stats.empty:
                st.error("⚠ Statistical anomalies detected (unusual resistance spikes):")
                st.dataframe(
                    novel_stats[
                        ["location", "year", "pathogen", "antibiotic", "resistance_rate", "z_score"]
                    ]
                )
            else:
                st.success("No strong statistical resistance spikes detected.")

        # ---------- 2) ML-BASED NOVELTY (Isolation Forest) ----------
        st.markdown("#### 2️⃣ ML Novelty (Isolation Forest)")
        if len(df) < 10:
            st.info("Need at least 10 rows for Isolation Forest. Skipping ML-based novelty.")
        else:
            iso_features = df[["year", "resistance_rate"]].copy()

            try:
                iso = IsolationForest(
                    contamination=0.1, random_state=42
                )  # 10% assumed anomaly
                df["ml_novelty_score"] = iso.fit_predict(iso_features)
                ml_novelty = df[df["ml_novelty_score"] == -1]

                if not ml_novelty.empty:
                    st.error("⚠ ML-based novel patterns detected (multidimensional anomalies):")
                    st.dataframe(
                        ml_novelty[
                            [
                                "location",
                                "year",
                                "pathogen",
                                "antibiotic",
                                "resistance_rate",
                            ]
                        ]
                    )
                else:
                    st.success("No ML-detected anomalies in the current dataset.")
            except Exception as e:
                st.warning(f"Isolation Forest could not run: {e}")

        # ---------- 3) GENOMIC NOVELTY (NEW AMR GENES) ----------
        st.markdown("#### 3️⃣ Genomic Novelty (new / unusual AMR genes)")
        if "gene" not in df.columns:
            st.info("No 'gene' column found. Add a 'gene' column to enable genomic novelty detection.")
        else:
            # Simple reference set of frequently reported AMR genes
            reference_gene_list = {
                "mcr-1", "mcr-2", "blaNDM", "blaCTX-M", "blaKPC",
                "tetA", "vanA", "OXA-48"
            }

            df["genomic_novelty"] = ~df["gene"].isin(reference_gene_list)
            novel_genes = df[df["genomic_novelty"]]

            if not novel_genes.empty:
                st.error("⚠ Potential novel or uncommon AMR genes detected:")
                st.dataframe(
                    novel_genes[
                        ["location", "year", "pathogen", "antibiotic", "gene"]
                    ].drop_duplicates()
                )
            else:
                st.success("No novel AMR genes detected compared to the reference list.")

    # ---------------------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------------------
    if show_plots:
        st.subheader("📊 Visual Analytics")

        # ---------- LINE PLOT ----------
        st.markdown("#### Resistance over time by pathogen")
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            sns.lineplot(
                data=df.sort_values("year"),
                x="year",
                y="resistance_rate",
                hue="pathogen",
                marker="o",
                ax=ax,
            )
            ax.set_ylabel("Resistance rate")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not create line plot: {e}")

        # ---------- HEATMAP ----------
        st.markdown("#### Pathogen × Antibiotic mean resistance heatmap")
        try:
            pivot = df.pivot_table(
                values="resistance_rate",
                index="pathogen",
                columns="antibiotic",
                aggfunc="mean",
            )
            if pivot.empty:
                st.info("Not enough data to build a heatmap.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(pivot, annot=False, cmap="viridis")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not create heatmap: {e}")

    # ---------------------------------------------------------
    # SYNTHETIC LAB (GANs)
    # ---------------------------------------------------------
    if show_synthetic:
        st.subheader("🧪 Synthetic Lab (Generative AI)")
        st.markdown(
            """
            **Novelty**: Use **CTGAN (Conditional Tabular GAN)** to learn the distribution of your AMR data 
            and generate **realistic synthetic samples**. This solves the "Data Scarcity" problem in LMICs.
            """
        )
        
        if st.button("🚀 Train GAN & Generate Synthetic Data"):
            with st.spinner("Training CTGAN... This may take a minute..."):
                try:
                    # 1. Prepare Metadata
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(df)
                    
                    # 2. Initialize CTGAN
                    # Use fewer epochs for demo speed, but enough to learn something
                    synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=True)
                    
                    # 3. Train
                    synthesizer.fit(df)
                    
                    # 4. Generate
                    synthetic_data = synthesizer.sample(num_rows=1000)
                    
                    st.success("✅ Synthetic Data Generated Successfully!")
                    
                    st.write("### 🧬 Synthetic Data Preview")
                    st.dataframe(synthetic_data.head())
                    
                    # 5. Download
                    csv = synthetic_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Synthetic Dataset",
                        csv,
                        "synthetic_amr_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # 6. Quality Check (Simple Visual)
                    st.write("### 📊 Real vs Synthetic Distribution (Resistance Rate)")
                    fig_comp, ax_comp = plt.subplots(1, 2, figsize=(12, 4))
                    
                    sns.histplot(df["resistance_rate"], ax=ax_comp[0], color="blue", kde=True)
                    ax_comp[0].set_title("Real Data")
                    
                    sns.histplot(synthetic_data["resistance_rate"], ax=ax_comp[1], color="green", kde=True)
                    ax_comp[1].set_title("Synthetic Data (GAN)")
                    
                    st.pyplot(fig_comp)
                    
                except Exception as e:
                    st.error(f"GAN Training Failed: {e}")

else:
    st.info("📥 Upload a CSV file to begin analysis. Use the sample datasets I provided.")
