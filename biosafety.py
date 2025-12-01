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
st.title("🧬 AI-Driven AMR Biosafety & Novelty Detection System")
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
show_risk = st.sidebar.checkbox("Show AMR Risk Scores", True, help="Risk scoring by combo")
show_novelty = st.sidebar.checkbox("Show Novelty Detection", True, help="Spike/novelty alerts")

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
    # MACHINE LEARNING MODEL (Random Forest Regression)
    # ---------------------------------------------------------
    st.subheader("🤖 AMR Prediction Model (Random Forest)")

    X = df[["location", "year", "pathogen", "antibiotic"]].copy()
    y = df["resistance_rate"].copy()

    # Preprocessing: one-hot encode categorical, passthrough year
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             ["location", "pathogen", "antibiotic"]),
            ("num", "passthrough", ["year"]),
        ]
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
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
    # RISK SCORE
    # ---------------------------------------------------------
    if show_risk:
        st.subheader("⚠️ AMR Biosafety Risk Score")

        # Predict for all rows (full dataset)
        try:
            df["predicted_rate"] = pipeline.predict(X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        df["risk_score"] = (df["predicted_rate"] * 100).clip(0, 100)

        st.write("Sample of risk-scored rows:")
        st.dataframe(
            df[["location", "year", "pathogen", "antibiotic", "risk_score"]].head()
        )

        high_risk = df.sort_values("risk_score", ascending=False).head(10)
        st.write("### 🔥 Top 10 Highest-Risk Combinations")
        st.table(high_risk[["location", "year", "pathogen", "antibiotic", "risk_score"]])

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

else:
    st.info("📥 Upload a CSV file to begin analysis. Use the sample datasets I provided.")
