import streamlit as st
import pandas as pd
import json
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------
# LOTTIE LOADER
# ----------------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AMR Biosafety Dashboard",
    page_icon="🦠",
    layout="wide",
)

header_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
success_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")
alert_anim = load_lottie_url("https://assets3.lottiefiles.com/private_files/lf30_editor_zuz8ic.json")

# ----------------------------
# HEADER
# ----------------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🧬 AI-Driven AMR & Biosafety System")
    st.write("A decision-support dashboard using machine learning + global antibiotic usage data.")
with col2:
    st_lottie(header_anim, height=180)


# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("📂 Upload Your AMR Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("📊 Options")
show_plots = st.sidebar.checkbox("Show Visualizations", True)
show_risk = st.sidebar.checkbox("Show AMR Risk Scores", True)

st.sidebar.markdown("---")
st.sidebar.info("This dashboard merges AMR trends + OWID antibiotic usage + ML predictions.")


# ----------------------------
# MAIN LOGIC
# ----------------------------
if uploaded_file is not None:

    st.success("Dataset uploaded successfully!")
    st_lottie(success_anim, height=120)

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # --------------------------------------------
    # MERGE ANTIBIOTIC USAGE (children + livestock)
    # --------------------------------------------
    st.subheader("🌍 Adding OWID Antibiotic Usage Data")

    owid_children = pd.read_csv("external_data/antibiotic-usage-in-children_raw.csv")
    owid_livestock = pd.read_csv("external_data/antibiotic-use-livestock-tonnes_raw.csv")

    owid_children.rename(columns={"Year": "year", "Entity": "location"}, inplace=True)
    owid_livestock.rename(columns={"Year": "year", "Entity": "location"}, inplace=True)

    merged = df.merge(
        owid_children[["location", "year", "antibiotic_usage__pct"]],
        on=["location", "year"],
        how="left"
    ).merge(
        owid_livestock[["location", "year", "livestock_antimicrobial_usage_tonnes"]],
        on=["location", "year"],
        how="left"
    )

    st.write("Merged dataset shape:", merged.shape)
    st.dataframe(merged.head())


    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------
    merged["n_tested"] = merged["n_tested"].astype(float)
    merged["n_resistant"] = merged["n_resistant"].astype(float)
    merged["resistance_rate"] = merged["n_resistant"] / merged["n_tested"]

    # Drop invalid
    merged = merged[(merged["resistance_rate"] >= 0) & (merged["resistance_rate"] <= 1)]

    # ----------------------------
    # MODEL TRAINING
    # ----------------------------
    st.subheader("🤖 Machine Learning Model (Random Forest)")

    X = merged[["location", "year", "pathogen", "antibiotic"]]
    y = merged["resistance_rate"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["location", "pathogen", "antibiotic"]),
        ("num", "passthrough", ["year"])
    ])

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    pipe = Pipeline([("preprocess", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.metric("📈 R² Score", f"{r2:.3f}")
    st.metric("📉 RMSE", f"{rmse:.3f}")


    # ----------------------------
    # AMR RISK SCORE
    # ----------------------------
    if show_risk:
        st.subheader("⚠️ AMR Biosafety Risk Score")

        merged["predicted_rate"] = pipe.predict(X)
        merged["risk_score"] = merged["predicted_rate"] * 100

        st.dataframe(merged[["location", "pathogen", "antibiotic", "risk_score"]].head())

        high_risk = merged.sort_values("risk_score", ascending=False).head(10)

        st.write("### 🔥 Highest-Risk Combinations")
        st.table(high_risk[["location", "pathogen", "antibiotic", "risk_score"]])

        if high_risk["risk_score"].max() > 70:
            st_lottie(alert_anim, height=150)
            st.error("High AMR risk detected — review biosafety & stewardship protocols.")


    # VISUALIZATIONS
    if show_plots:

        st.subheader("📊 Resistance Over Time")

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=merged, x="year", y="resistance_rate", hue="pathogen", ax=ax)
        st.pyplot(fig)

        st.subheader("🔬 Pathogen × Antibiotic Heatmap")

        pivot = (
            merged.groupby(["pathogen", "antibiotic"])["resistance_rate"]
            .mean()
            .reset_index()
            .pivot("pathogen", "antibiotic", "resistance_rate")
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=False, cmap="viridis")
        st.pyplot(fig)

else:
    st.info("Upload a CSV file to begin analysis.")