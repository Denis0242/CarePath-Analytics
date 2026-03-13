import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostic Dashboard",
    page_icon="🎗️",
    layout="wide",
)

# ---------------------------------------------------
# THEME COLORS
# ---------------------------------------------------
PRIMARY_PINK = "#E91E63"
LIGHT_PINK = "#FCE4EC"
SOFT_PINK = "#F8BBD0"
DEEP_ROSE = "#AD1457"
BG_COLOR = "#FFF7FA"
CARD_BG = "#FFFFFF"
TEXT_DARK = "#4A4A4A"
BENIGN_COLOR = "#F48FB1"
MALIGNANT_COLOR = "#C2185B"
ACCENT_PURPLE = "#CE93D8"
GB_COLOR = "#8E24AA"

# ---------------------------------------------------
# STYLING
# ---------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #fff7fa 0%, #fffdfd 100%);
        color: {TEXT_DARK};
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #fff0f5 0%, #fde7ef 100%);
        border-right: 2px solid #f8bbd0;
    }}

    .main-title {{
        font-size: 2.7rem;
        font-weight: 800;
        color: {DEEP_ROSE};
        margin-bottom: 0.15rem;
    }}

    .sub-text {{
        color: #6e5a62;
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }}

    .dashboard-card {{
        background: {CARD_BG};
        border-radius: 18px;
        padding: 1rem 1.1rem;
        border: 1px solid #f3c2d2;
        box-shadow: 0 4px 18px rgba(233, 30, 99, 0.08);
        margin-bottom: 1rem;
    }}

    .section-title {{
        color: {DEEP_ROSE};
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }}

    .small-note {{
        color: #7a6870;
        font-size: 0.92rem;
    }}

    .sidebar-box {{
        background: rgba(255,255,255,0.75);
        padding: 0.8rem;
        border-radius: 14px;
        border: 1px solid #f2bfd0;
        margin-bottom: 1rem;
    }}

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #f2bfd0;
        padding: 12px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(233, 30, 99, 0.07);
    }}

    .ribbon-badge {{
        display: inline-block;
        background: {LIGHT_PINK};
        color: {DEEP_ROSE};
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        border: 1px solid #f3bfd1;
        margin-bottom: 0.6rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# IMAGE HELPERS
# ---------------------------------------------------
def show_local_image(path_str: str, width=None, use_column_width=True):
    path = Path(path_str)
    if path.exists():
        st.image(str(path), width=width, use_container_width=use_column_width)


def image_exists(path_str: str) -> bool:
    return Path(path_str).exists()


# ---------------------------------------------------
# RISK HELPERS
# ---------------------------------------------------
def get_risk_band(probability: float):
    if probability < 0.30:
        return "Low Risk", BENIGN_COLOR, "Routine review / monitor"
    elif probability < 0.70:
        return "Moderate Risk", PRIMARY_PINK, "Recommend closer clinical assessment"
    else:
        return "High Risk", MALIGNANT_COLOR, "Escalate for urgent diagnostic follow-up"


def build_full_input(input_values: dict, feature_columns: list, df: pd.DataFrame):
    full_input = {}
    for col in feature_columns:
        if col in input_values:
            full_input[col] = input_values[col]
        else:
            full_input[col] = float(df[col].median())
    return pd.DataFrame([full_input])


def get_tree_shap_array(shap_values):
    """
    Convert SHAP outputs into a clean 2D numpy array of shape:
    (n_samples, n_features)
    """
    if isinstance(shap_values, list):
        arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        arr = np.array(arr)
    elif hasattr(shap_values, "values"):
        arr = np.array(shap_values.values)
    else:
        arr = np.array(shap_values)

    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2:
            arr = arr[1]
        else:
            arr = arr.mean(axis=-1)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape after processing: {arr.shape}")

    return arr


# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------
@st.cache_data
def load_data():
    csv_path = Path("data/breast_cancer_data.csv")

    if not csv_path.exists():
        st.error(
            "File not found: data/breast_cancer_data.csv\n\n"
            "Please place your dataset in the data folder and name it exactly "
            "'breast_cancer_data.csv'."
        )
        st.stop()

    df = pd.read_csv(csv_path)

    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    df.columns = [col.strip() for col in df.columns]

    if "diagnosis" not in df.columns:
        st.error("The dataset must contain a 'diagnosis' column.")
        st.stop()

    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
    df = df[df["diagnosis"].isin(["B", "M"])].copy()

    return df, "breast_cancer_data.csv"


# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
@st.cache_resource
def train_models(df: pd.DataFrame):
    model_df = df.copy()

    y = model_df["diagnosis"].map({"B": 0, "M": 1})

    X = model_df.drop(columns=["diagnosis"], errors="ignore")

    for col in ["id"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    X = X.select_dtypes(include=[np.number]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    lr = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    lr.fit(X_train_scaled, y_train)

    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    gb_pred = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]

    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    results_df = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Accuracy": accuracy_score(y_test, rf_pred),
                "Precision": precision_score(y_test, rf_pred),
                "Recall": recall_score(y_test, rf_pred),
                "F1": f1_score(y_test, rf_pred),
                "ROC_AUC": roc_auc_score(y_test, rf_proba),
            },
            {
                "Model": "Gradient Boosting",
                "Accuracy": accuracy_score(y_test, gb_pred),
                "Precision": precision_score(y_test, gb_pred),
                "Recall": recall_score(y_test, gb_pred),
                "F1": f1_score(y_test, gb_pred),
                "ROC_AUC": roc_auc_score(y_test, gb_proba),
            },
            {
                "Model": "Logistic Regression",
                "Accuracy": accuracy_score(y_test, lr_pred),
                "Precision": precision_score(y_test, lr_pred),
                "Recall": recall_score(y_test, lr_pred),
                "F1": f1_score(y_test, lr_pred),
                "ROC_AUC": roc_auc_score(y_test, lr_proba),
            },
        ]
    ).sort_values("ROC_AUC", ascending=False)

    rf_feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    gb_feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": gb.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    cm_rf = confusion_matrix(y_test, rf_pred)
    cm_gb = confusion_matrix(y_test, gb_pred)
    cm_lr = confusion_matrix(y_test, lr_pred)

    # SHAP using RF on sample for speed
    shap_sample = X_train.sample(min(120, len(X_train)), random_state=42)
    shap_explainer_rf = shap.TreeExplainer(rf)
    raw_shap_values_rf = shap_explainer_rf.shap_values(shap_sample)
    shap_values_rf = get_tree_shap_array(raw_shap_values_rf)

    best_model_name = results_df.iloc[0]["Model"]
    if best_model_name == "Random Forest":
        best_model = rf
        best_model_uses_scaling = False
    elif best_model_name == "Gradient Boosting":
        best_model = gb
        best_model_uses_scaling = False
    else:
        best_model = lr
        best_model_uses_scaling = True

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "rf": rf,
        "gb": gb,
        "lr": lr,
        "scaler": scaler,
        "results_df": results_df,
        "rf_feature_importance": rf_feature_importance,
        "gb_feature_importance": gb_feature_importance,
        "feature_importance": rf_feature_importance,
        "cm_rf": cm_rf,
        "cm_gb": cm_gb,
        "cm_lr": cm_lr,
        "shap_sample": shap_sample,
        "shap_explainer_rf": shap_explainer_rf,
        "shap_values_rf": shap_values_rf,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_model_uses_scaling": best_model_uses_scaling,
    }


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
df, source_used = load_data()
artifacts = train_models(df)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    if image_exists("assets/ribbon.png"):
        st.image("assets/ribbon.png", width=90)
    else:
        st.markdown("## 🎗️")

    st.markdown("## Navigation")

    st.markdown(
        """
        <div class="sidebar-box">
        Breast Cancer Awareness themed analytics dashboard for diagnosis exploration,
        model evaluation, explainability, and clinical risk scoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Go to",
        [
            "Overview",
            "Data Explorer",
            "Model Performance",
            "Prediction Tool",
            "SHAP Explainability",
            "Visualizations",
            "Project Report",
        ],
    )

    st.markdown("---")
    st.write("**Data source:**", source_used)
    st.success("Using the real breast_cancer_data.csv dataset")
    st.info(f"Best model by ROC-AUC: {artifacts['best_model_name']}")

# ---------------------------------------------------
# HEADER / HERO
# ---------------------------------------------------
if image_exists("assets/breast_cancer_banner.jpg"):
    st.image("assets/breast_cancer_banner.jpg", use_container_width=True)

st.markdown('<div class="ribbon-badge">🎗️ Breast Cancer Awareness Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Breast Cancer Diagnostic Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Interactive ML dashboard using your real breast_cancer_data.csv dataset, extended with Gradient Boosting, SHAP explainability, and clinical risk scoring.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------
# OVERVIEW
# ---------------------------------------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Best ROC-AUC", f"{artifacts['results_df']['ROC_AUC'].max():.3f}")
    c4.metric("Best Model", artifacts["best_model_name"])

    st.markdown('<div class="section-title">Project Summary</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dashboard-card">
        This dashboard uses the breast cancer dataset to explore diagnosis patterns, compare
        machine learning model performance, explain predictions with SHAP, and support
        triage-style clinical risk scoring in a healthcare analytics interface.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-title">Diagnosis Breakdown</div>', unsafe_allow_html=True)
        diagnosis_counts = df["diagnosis"].value_counts().reset_index()
        diagnosis_counts.columns = ["diagnosis", "count"]

        fig_diag = px.pie(
            diagnosis_counts,
            names="diagnosis",
            values="count",
            title="Benign vs Malignant Distribution",
            color="diagnosis",
            color_discrete_map={
                "B": BENIGN_COLOR,
                "M": MALIGNANT_COLOR,
            },
        )
        fig_diag.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            title_font_color=DEEP_ROSE,
        )
        st.plotly_chart(fig_diag, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Top 10 Important Features</div>', unsafe_allow_html=True)
        top_features = artifacts["rf_feature_importance"].head(10).sort_values("importance")

        fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title="Random Forest Feature Importance",
            color="importance",
            color_continuous_scale=["#fde0ea", "#f48fb1", "#e91e63", "#ad1457"],
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            title_font_color=DEEP_ROSE,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# DATA EXPLORER
# ---------------------------------------------------
elif page == "Data Explorer":
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown('<div class="section-title">Dataset Info</div>', unsafe_allow_html=True)
    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Missing Values": df.isnull().sum().values,
        }
    )
    st.dataframe(info_df, use_container_width=True)

    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=[np.number])
    st.dataframe(numeric_df.describe().T, use_container_width=True)

    st.markdown('<div class="section-title">Diagnosis Distribution</div>', unsafe_allow_html=True)
    counts = df["diagnosis"].value_counts().reset_index()
    counts.columns = ["diagnosis", "count"]

    fig = px.bar(
        counts,
        x="diagnosis",
        y="count",
        color="diagnosis",
        color_discrete_map={
            "B": BENIGN_COLOR,
            "M": MALIGNANT_COLOR,
        },
        title="Class Distribution",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font_color=DEEP_ROSE,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------
elif page == "Model Performance":
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    results_df = artifacts["results_df"].copy()
    st.dataframe(results_df, use_container_width=True)

    melted = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Performance Metrics by Model",
        color_discrete_map={
            "Random Forest": PRIMARY_PINK,
            "Gradient Boosting": GB_COLOR,
            "Logistic Regression": ACCENT_PURPLE,
        },
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font_color=DEEP_ROSE,
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">RF Top 15 Features</div>', unsafe_allow_html=True)
        rf_feature_importance = artifacts["rf_feature_importance"].head(15).sort_values("importance")

        fig2 = px.bar(
            rf_feature_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Random Forest Feature Importance",
            color="importance",
            color_continuous_scale=["#fde0ea", "#f8bbd0", "#e91e63", "#ad1457"],
        )
        fig2.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            title_font_color=DEEP_ROSE,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">GB Top 15 Features</div>', unsafe_allow_html=True)
        gb_feature_importance = artifacts["gb_feature_importance"].head(15).sort_values("importance")

        fig3 = px.bar(
            gb_feature_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Gradient Boosting Feature Importance",
            color="importance",
            color_continuous_scale=["#f3e5f5", "#ce93d8", "#ab47bc", "#6a1b9a"],
        )
        fig3.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            title_font_color=DEEP_ROSE,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Random Forest**")
        cm_rf_df = pd.DataFrame(
            artifacts["cm_rf"],
            index=["Actual Benign", "Actual Malignant"],
            columns=["Pred Benign", "Pred Malignant"],
        )
        st.dataframe(cm_rf_df, use_container_width=True)

    with col2:
        st.write("**Gradient Boosting**")
        cm_gb_df = pd.DataFrame(
            artifacts["cm_gb"],
            index=["Actual Benign", "Actual Malignant"],
            columns=["Pred Benign", "Pred Malignant"],
        )
        st.dataframe(cm_gb_df, use_container_width=True)

    with col3:
        st.write("**Logistic Regression**")
        cm_lr_df = pd.DataFrame(
            artifacts["cm_lr"],
            index=["Actual Benign", "Actual Malignant"],
            columns=["Pred Benign", "Pred Malignant"],
        )
        st.dataframe(cm_lr_df, use_container_width=True)

# ---------------------------------------------------
# PREDICTION TOOL
# ---------------------------------------------------
elif page == "Prediction Tool":
    st.markdown('<div class="section-title">Clinical Risk Scoring</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="dashboard-card">
        Enter values for the top predictive tumor features. The dashboard estimates malignancy
        probability and translates it into a clinical risk tier for triage-style interpretation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_option = st.selectbox(
        "Choose prediction model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        index=0 if artifacts["best_model_name"] == "Random Forest"
        else 1 if artifacts["best_model_name"] == "Gradient Boosting"
        else 2,
    )

    if model_option == "Random Forest":
        model = artifacts["rf"]
        use_scaled = False
        model_feature_rank = artifacts["rf_feature_importance"]
    elif model_option == "Gradient Boosting":
        model = artifacts["gb"]
        use_scaled = False
        model_feature_rank = artifacts["gb_feature_importance"]
    else:
        model = artifacts["lr"]
        use_scaled = True
        model_feature_rank = artifacts["rf_feature_importance"]

    top_cols = model_feature_rank["feature"].head(10).tolist()

    input_values = {}
    cols = st.columns(2)

    for i, feature in enumerate(top_cols):
        median_val = float(df[feature].median())
        with cols[i % 2]:
            input_values[feature] = st.number_input(
                feature,
                value=median_val,
                step=0.01,
                format="%.4f",
                key=f"pred_{feature}",
            )

    if st.button("Run Clinical Risk Score"):
        input_df = build_full_input(input_values, artifacts["X"].columns.tolist(), df)

        if use_scaled:
            score_input = artifacts["scaler"].transform(input_df)
            probability = model.predict_proba(score_input)[0, 1]
            prediction = model.predict(score_input)[0]
        else:
            probability = model.predict_proba(input_df)[0, 1]
            prediction = model.predict(input_df)[0]

        risk_label, risk_color, recommendation = get_risk_band(probability)

        left, middle, right = st.columns([1, 1, 1])

        with left:
            if prediction == 1:
                st.error("Prediction: Malignant")
            else:
                st.success("Prediction: Benign")

            st.metric("Malignancy Probability", f"{probability:.2%}")
            st.metric("Model Used", model_option)

        with middle:
            st.markdown(
                f"""
                <div class="dashboard-card" style="border-left:8px solid {risk_color};">
                    <h4 style="color:{DEEP_ROSE}; margin-bottom:8px;">Clinical Risk Tier</h4>
                    <h2 style="margin:0; color:{risk_color};">{risk_label}</h2>
                    <p style="margin-top:10px;"><b>Suggested action:</b> {recommendation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={"text": "Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color},
                        "steps": [
                            {"range": [0, 30], "color": "#fde0ea"},
                            {"range": [30, 70], "color": "#f8bbd0"},
                            {"range": [70, 100], "color": "#f06292"},
                        ],
                    },
                )
            )
            fig.update_layout(
                height=300,
                paper_bgcolor="white",
                font={"color": DEEP_ROSE},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "This score is for analytics and educational decision support only. "
            "It is not a substitute for clinical diagnosis."
        )

        st.markdown("#### Input Feature Snapshot")
        st.dataframe(
            input_df.T.rename(columns={0: "Entered Value"}),
            use_container_width=True,
        )

# ---------------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------------
elif page == "SHAP Explainability":
    st.markdown('<div class="section-title">SHAP Explainability</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="dashboard-card">
        SHAP explains how each feature contributes to model output. This helps convert a strong
        predictive model into a more interpretable healthcare analytics tool by showing which
        tumor measurements most increase or decrease malignancy risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Global SHAP Summary (Random Forest)")

    shap_array = artifacts["shap_values_rf"]
    shap_sample = artifacts["shap_sample"]

    fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_array,
        shap_sample,
        show=False,
    )
    st.pyplot(fig_summary, clear_figure=True)

    st.markdown("#### Top 15 Features by Mean Absolute SHAP Impact")

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    mean_abs_shap = np.asarray(mean_abs_shap).reshape(-1)

    shap_df = pd.DataFrame({
        "feature": shap_sample.columns.tolist(),
        "mean_abs_shap": mean_abs_shap.tolist(),
    }).sort_values("mean_abs_shap", ascending=False).head(15)

    fig_bar = px.bar(
        shap_df.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Top 15 Features by Mean |SHAP|",
        color="mean_abs_shap",
        color_continuous_scale=["#fde0ea", "#f8bbd0", "#e91e63", "#ad1457"],
    )
    fig_bar.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font_color=DEEP_ROSE,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------
elif page == "Visualizations":
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)

    numeric_df = artifacts["X"].copy()
    corr = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[
                [0.0, "#fff0f5"],
                [0.25, "#f8bbd0"],
                [0.5, "#e1bee7"],
                [0.75, "#ec407a"],
                [1.0, "#880e4f"],
            ],
        )
    )
    fig.update_layout(
        height=800,
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font_color=DEEP_ROSE,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Feature Scatter Plot</div>', unsafe_allow_html=True)
    numeric_cols = artifacts["X"].columns.tolist()

    x_feature = st.selectbox("X-axis", numeric_cols, index=0)
    y_feature = st.selectbox("Y-axis", numeric_cols, index=1)

    fig2 = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="diagnosis",
        title=f"{x_feature} vs {y_feature}",
        hover_data=["id"] if "id" in df.columns else None,
        color_discrete_map={
            "B": BENIGN_COLOR,
            "M": MALIGNANT_COLOR,
        },
    )
    fig2.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font_color=DEEP_ROSE,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# PROJECT REPORT
# ---------------------------------------------------
elif page == "Project Report":
    st.markdown('<div class="section-title">Healthcare Analytics Report</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="dashboard-card">

        <h4 style="color:#AD1457;">Executive Summary</h4>
        <p>
        This healthcare analytics dashboard applies machine learning to breast tumor diagnostic
        measurements to support the classification of <b>benign</b> versus <b>malignant</b> cases.
        The product combines predictive modeling, model comparison, SHAP-based interpretability,
        and clinical risk scoring to translate complex feature patterns into decision-support
        insights that are easier for healthcare stakeholders to understand and act on.
        </p>

        <h4 style="color:#AD1457;">Clinical Problem</h4>
        <p>
        Breast cancer diagnosis depends on interpreting multiple tumor characteristics at the same time,
        including radius, texture, perimeter, smoothness, and concavity-related signals. In practice,
        this creates a high-dimensional decision environment where subtle differences in measurements
        can materially affect diagnostic interpretation.
        </p>
        <p>
        This project evaluates whether machine learning can improve analytical support by identifying
        feature patterns associated with malignancy, comparing model performance across algorithms,
        and surfacing interpretable risk signals that can strengthen early detection workflows,
        research analytics, and triage-style review.
        </p>

        <h4 style="color:#AD1457;">Key Analytical Insights</h4>
        <ul>
            <li>High-impact tumor measurements related to <b>radius</b>, <b>perimeter</b>, <b>concavity</b>, and <b>area</b> consistently emerge as strong indicators of malignancy risk.</li>
            <li>Model-based feature rankings and SHAP outputs show that structural tumor characteristics contribute more strongly to prediction than lower-signal variables.</li>
            <li>Clinical risk scoring makes model output more usable by translating raw malignancy probabilities into low, moderate, and high risk tiers for faster interpretation.</li>
        </ul>

        <h4 style="color:#AD1457;">Machine Learning Models Evaluated</h4>
        <ul>
            <li><b>Random Forest</b> – A strong nonlinear ensemble model that captures complex feature interactions and supports feature-importance analysis.</li>
            <li><b>Gradient Boosting</b> – A boosting-based model designed to improve predictive performance by sequentially learning residual error patterns.</li>
            <li><b>Logistic Regression</b> – An interpretable baseline model that provides a simpler benchmark for binary clinical classification.</li>
        </ul>

        <h4 style="color:#AD1457;">Explainability & Decision Support</h4>
        <ul>
            <li>SHAP explainability is used to quantify how individual tumor features influence malignancy predictions at the model level.</li>
            <li>Feature importance views and SHAP impact rankings improve transparency by showing which variables drive prediction strength most strongly.</li>
            <li>Clinical risk tiers convert model probabilities into more interpretable categories that better support healthcare analytics storytelling and review workflows.</li>
        </ul>

        <h4 style="color:#AD1457;">Data Preparation Pipeline</h4>
        <ul>
            <li>Uses <b>diagnosis</b> as the core target variable to model clinically relevant classification outcomes between benign and malignant tumors.</li>
            <li>Removes the <b>id</b> column because it is an administrative identifier rather than a meaningful clinical predictor.</li>
            <li>Drops <b>Unnamed: 32</b> as an empty/non-contributory field to improve dataset quality and reduce analytical noise.</li>
            <li>Retains numeric tumor measurement variables as the primary feature set for machine learning–based risk modeling.</li>
            <li>Splits the dataset into training and test samples to validate model performance and assess generalization on unseen patient records.</li>
        </ul>

        <h4 style="color:#AD1457;">Business & Healthcare Value</h4>
        <ul>
            <li>Supports earlier pattern detection by surfacing features associated with higher malignancy risk.</li>
            <li>Improves stakeholder understanding by combining predictive accuracy with model interpretability.</li>
            <li>Provides a reusable analytics framework for healthcare dashboards, research prototypes, and clinical decision-support exploration.</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Project Structure")
    st.code(
        """Causality-Standalone/
├── app.py
├── requirements.txt
├── data/
│   └── breast_cancer_data.csv
└── assets/
    ├── breast_cancer_banner.jpg
    └── ribbon.png""",
        language="text",
    )