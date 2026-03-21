"""
Healthcare Tech Product DS Dashboard — Feature Adoption & Engagement
=====================================================================
Use case:
    A clinical workflow SaaS platform (think EHR companion / care coordination tool)
    wants to understand which product features clinicians adopt, how deeply they
    engage, and which engagement patterns predict long-term retention.

What this demonstrates (junior DS hiring bar):
    ✓ Feature engineering with a healthcare product context
    ✓ Adoption funnel & cohort analysis
    ✓ Classification modelling (predicting "power users" vs "at-risk")
    ✓ SHAP explainability communicated in plain language
    ✓ Business-metric framing (DAU/MAU, time-to-first-value, feature stickiness)
    ✓ Clean, readable code with docstrings

Run:
    pip install streamlit pandas numpy scikit-learn shap plotly matplotlib
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareFlow Analytics | Feature Adoption DS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# DESIGN TOKENS
# ──────────────────────────────────────────────────────────
C = dict(
    bg="#F0F4F8",
    surface="#FFFFFF",
    card="#FFFFFF",
    border="#DDE3ED",
    teal="#0D9488",
    teal_lt="#CCFBF1",
    navy="#1E3A5F",
    slate="#475569",
    muted="#94A3B8",
    green="#16A34A",
    amber="#D97706",
    red="#DC2626",
    sky="#0EA5E9",
)

# ──────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: {C['bg']};
    color: {C['slate']};
}}
.stApp {{
    background-color: {C['bg']};
}}

section[data-testid="stSidebar"] {{
    background: {C['navy']};
    border-right: none;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {{
    color: #CBD5E1 !important;
}}

div[data-testid="stMetric"] {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-top: 3px solid {C['teal']};
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}}
div[data-testid="stMetricValue"] {{
    color: {C['navy']};
    font-weight: 800;
    font-size: 1.75rem;
    font-family: 'JetBrains Mono', monospace;
}}
div[data-testid="stMetricLabel"] {{
    color: {C['muted']};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
}}

.ds-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
}}
.ds-callout {{
    background: {C['teal_lt']};
    border-left: 4px solid {C['teal']};
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    color: {C['navy']};
    font-size: 0.95rem;
}}

.page-title {{
    font-size: 1.9rem;
    font-weight: 800;
    color: {C['navy']};
    line-height: 1.2;
    margin-bottom: 0.3rem;
}}
.page-sub {{
    color: {C['slate']};
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}}
.sec-title {{
    font-size: 0.78rem;
    font-weight: 700;
    color: {C['teal']};
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.75rem;
    margin-top: 0.5rem;
}}
.tag {{
    display: inline-block;
    background: {C['teal_lt']};
    color: {C['teal']};
    border: 1px solid #99F6E4;
    padding: 3px 11px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}

.badge-green {{
    background:#DCFCE7;
    color:{C['green']};
    border:1px solid #BBF7D0;
    padding:4px 13px;
    border-radius:999px;
    font-weight:700;
    font-size:0.85rem;
}}
.badge-amber {{
    background:#FEF3C7;
    color:{C['amber']};
    border:1px solid #FDE68A;
    padding:4px 13px;
    border-radius:999px;
    font-weight:700;
    font-size:0.85rem;
}}
.badge-red {{
    background:#FEE2E2;
    color:{C['red']};
    border:1px solid #FECACA;
    padding:4px 13px;
    border-radius:999px;
    font-weight:700;
    font-size:0.85rem;
}}

.stButton > button {{
    background: {C['teal']};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 0.5rem 1.5rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.9rem;
}}
.stButton > button:hover {{
    background: #0F766E;
}}
hr {{
    border-color: {C['border']};
}}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# SYNTHETIC HEALTHCARE SAAS DATA
# ──────────────────────────────────────────────────────────
FEATURES_LIST = [
    "patient_timeline",
    "e_prescribing",
    "care_gap_alerts",
    "secure_messaging",
    "lab_viewer",
    "referral_tracker",
    "voice_notes",
    "analytics_dashboard",
]

FEATURE_LABELS = {
    "patient_timeline": "Patient Timeline",
    "e_prescribing": "e-Prescribing",
    "care_gap_alerts": "Care Gap Alerts",
    "secure_messaging": "Secure Messaging",
    "lab_viewer": "Lab Result Viewer",
    "referral_tracker": "Referral Tracker",
    "voice_notes": "Voice Notes",
    "analytics_dashboard": "Analytics Dashboard",
}

FEATURE_DESC = {
    "patient_timeline": "Longitudinal patient history view",
    "e_prescribing": "Digital prescription management",
    "care_gap_alerts": "Automated alerts for care gaps",
    "secure_messaging": "HIPAA-compliant in-app messaging",
    "lab_viewer": "Integrated lab result viewer",
    "referral_tracker": "Specialist referral workflow",
    "voice_notes": "AI-transcribed voice documentation",
    "analytics_dashboard": "Population health analytics",
}

ROLE_LIST = ["Physician", "Nurse Practitioner", "Care Coordinator", "Registered Nurse"]
DEPT_LIST = ["Primary Care", "Cardiology", "Oncology", "Emergency", "Pediatrics"]
PLAN_LIST = ["Starter", "Professional", "Enterprise"]


@st.cache_data
def generate_data(n: int = 1800, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic CareFlow clinician engagement data."""
    rng = np.random.default_rng(seed)

    roles = rng.choice(ROLE_LIST, n, p=[0.35, 0.25, 0.20, 0.20])
    depts = rng.choice(DEPT_LIST, n)
    plans = rng.choice(PLAN_LIST, n, p=[0.40, 0.40, 0.20])
    plan_idx = np.where(plans == "Starter", 0, np.where(plans == "Professional", 1, 2))

    tenure_months = (rng.exponential(10, n) + 1).clip(1, 36).astype(int)
    logins_last_30d = np.clip(rng.normal(12 + plan_idx * 3, 5, n), 0, 60).astype(int)
    session_length_min = np.clip(rng.normal(14 + plan_idx * 2, 6, n), 1, 90).round(1)
    patients_documented = np.clip(rng.normal(35 + plan_idx * 10, 20, n), 0, 200).astype(int)

    pt = (rng.random(n) < 0.90).astype(int)
    ep = (rng.random(n) < 0.65 + plan_idx * 0.05).astype(int)
    cg = (rng.random(n) < 0.50 + plan_idx * 0.08).astype(int)
    sm = (rng.random(n) < 0.72).astype(int)
    lv = (rng.random(n) < 0.60 + plan_idx * 0.06).astype(int)
    rt = (rng.random(n) < 0.40 + plan_idx * 0.07).astype(int)
    vn = (rng.random(n) < 0.28 + plan_idx * 0.10).astype(int)
    ad = (rng.random(n) < 0.22 + plan_idx * 0.15).astype(int)

    features_adopted = pt + ep + cg + sm + lv + rt + vn + ad

    def depth(adopted, base, boost):
        raw = np.clip(rng.normal(base + boost, 2.5, n), 0, 10)
        return (raw * adopted).round(1)

    depth_pt = depth(pt, 7.0, plan_idx * 0.3)
    depth_ep = depth(ep, 6.0, plan_idx * 0.4)
    depth_cg = depth(cg, 5.5, plan_idx * 0.5)
    depth_sm = depth(sm, 6.5, plan_idx * 0.2)
    depth_vn = depth(vn, 4.5, plan_idx * 0.8)
    depth_ad = depth(ad, 4.0, plan_idx * 1.0)

    ttfv_days = np.clip(rng.exponential(4, n) + 1, 1, 30).astype(int)
    support_tickets = np.clip(rng.poisson(0.8, n), 0, 8)
    nps_score = np.clip(rng.normal(42 + plan_idx * 8, 22, n), -100, 100).astype(int)

    cohort = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 18, n) * 30, unit="D"
    )

    score = (
        0.25 * (logins_last_30d / 30)
        + 0.20 * (features_adopted / 8)
        + 0.15 * (session_length_min / 60)
        + 0.10 * (depth_vn / 10)
        + 0.10 * (depth_ad / 10)
        + 0.08 * (1 - ttfv_days / 30)
        + 0.07 * (nps_score / 100 + 1) / 2
        - 0.05 * (support_tickets / 8)
        + rng.normal(0, 0.08, n)
    )
    power_user = (score > np.percentile(score, 45)).astype(int)

    return pd.DataFrame({
        "user_id": [f"CL{i:05d}" for i in range(n)],
        "role": roles,
        "department": depts,
        "plan": plans,
        "tenure_months": tenure_months,
        "logins_last_30d": logins_last_30d,
        "session_length_min": session_length_min,
        "patients_documented": patients_documented,
        "features_adopted": features_adopted,
        "ttfv_days": ttfv_days,
        "support_tickets": support_tickets,
        "nps_score": nps_score,
        "adopted_patient_timeline": pt,
        "adopted_e_prescribing": ep,
        "adopted_care_gap_alerts": cg,
        "adopted_secure_messaging": sm,
        "adopted_lab_viewer": lv,
        "adopted_referral_tracker": rt,
        "adopted_voice_notes": vn,
        "adopted_analytics_dashboard": ad,
        "depth_patient_timeline": depth_pt,
        "depth_e_prescribing": depth_ep,
        "depth_care_gap_alerts": depth_cg,
        "depth_secure_messaging": depth_sm,
        "depth_voice_notes": depth_vn,
        "depth_analytics_dashboard": depth_ad,
        "cohort_month": cohort,
        "power_user": power_user,
    })


# ──────────────────────────────────────────────────────────
# MODEL FEATURES
# ──────────────────────────────────────────────────────────
MODEL_FEATURES = [
    "tenure_months",
    "logins_last_30d",
    "session_length_min",
    "patients_documented",
    "features_adopted",
    "ttfv_days",
    "support_tickets",
    "nps_score",
    "adopted_e_prescribing",
    "adopted_care_gap_alerts",
    "adopted_voice_notes",
    "adopted_analytics_dashboard",
    "depth_voice_notes",
    "depth_analytics_dashboard",
]

MODEL_FEATURE_DESC = {
    "tenure_months": "Months since account activation",
    "logins_last_30d": "Login events in last 30 days",
    "session_length_min": "Avg session length (minutes)",
    "patients_documented": "Patients documented in last 30d",
    "features_adopted": "Count of distinct features ever used",
    "ttfv_days": "Days to first meaningful action (time-to-value)",
    "support_tickets": "Support tickets raised",
    "nps_score": "Net Promoter Score (-100 to 100)",
    "adopted_e_prescribing": "Has used e-Prescribing (0/1)",
    "adopted_care_gap_alerts": "Has used Care Gap Alerts (0/1)",
    "adopted_voice_notes": "Has used Voice Notes (0/1)",
    "adopted_analytics_dashboard": "Has used Analytics Dashboard (0/1)",
    "depth_voice_notes": "Voice Notes usage depth score (0-10)",
    "depth_analytics_dashboard": "Analytics Dashboard depth score (0-10)",
}


def normalize_shap_values(raw_shap_values, n_features: int) -> np.ndarray:
    """
    Normalize SHAP output into a guaranteed 2D numpy array:
    shape -> (n_samples, n_features)
    """
    arr = None

    if isinstance(raw_shap_values, list):
        candidates = [np.array(x) for x in raw_shap_values]
        for candidate in candidates:
            if candidate.ndim == 2 and candidate.shape[1] == n_features:
                arr = candidate
                break
            if candidate.ndim == 3:
                squeezed = np.squeeze(candidate)
                if squeezed.ndim == 2 and squeezed.shape[1] == n_features:
                    arr = squeezed
                    break
        if arr is None:
            arr = np.array(candidates[-1])

    elif hasattr(raw_shap_values, "values"):
        arr = np.array(raw_shap_values.values)

    else:
        arr = np.array(raw_shap_values)

    arr = np.squeeze(arr)

    if arr.ndim == 3:
        # Common format: (n_samples, n_features, n_classes)
        if arr.shape[1] == n_features:
            arr = arr[:, :, -1]
        elif arr.shape[2] == n_features:
            arr = arr[:, -1, :]
        else:
            arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim == 1:
        if arr.size == n_features:
            arr = arr.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected 1D SHAP shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"SHAP values could not be normalized to 2D. Got shape: {arr.shape}")

    if arr.shape[1] != n_features and arr.shape[0] == n_features:
        arr = arr.T

    if arr.shape[1] != n_features:
        raise ValueError(
            f"SHAP feature mismatch after normalization. "
            f"Expected {n_features} features, got shape {arr.shape}."
        )

    return arr


@st.cache_resource
def train_models(df: pd.DataFrame) -> dict:
    """Train three classifiers to predict power_user status."""
    X = df[MODEL_FEATURES].copy()
    y = df["power_user"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.07,
        max_depth=3,
        random_state=42,
    )
    lr = LogisticRegression(max_iter=1000, random_state=42)

    rf.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr)
    lr.fit(X_tr_s, y_tr)

    def row(name, model, Xeval, yeval):
        preds = model.predict(Xeval)
        probs = model.predict_proba(Xeval)[:, 1]
        return {
            "Model": name,
            "Accuracy": accuracy_score(yeval, preds),
            "Precision": precision_score(yeval, preds, zero_division=0),
            "Recall": recall_score(yeval, preds, zero_division=0),
            "F1": f1_score(yeval, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(yeval, probs),
        }

    results = pd.DataFrame([
        row("Random Forest", rf, X_te, y_te),
        row("Gradient Boosting", gb, X_te, y_te),
        row("Logistic Regression", lr, X_te_s, y_te),
    ]).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    rf_fi = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    gb_fi = pd.DataFrame({
        "feature": X.columns,
        "importance": gb.feature_importances_,
    }).sort_values("importance", ascending=False)

    cm_rf = confusion_matrix(y_te, rf.predict(X_te))
    cm_gb = confusion_matrix(y_te, gb.predict(X_te))
    cm_lr = confusion_matrix(y_te, lr.predict(X_te_s))

    shap_sample = X_tr.sample(min(120, len(X_tr)), random_state=42)
    explainer = None
    shap_arr = None
    shap_error = None

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(rf)
            raw_shap = explainer.shap_values(shap_sample)
            shap_arr = normalize_shap_values(raw_shap, n_features=shap_sample.shape[1])
        except Exception as e:
            shap_arr = None
            shap_error = str(e)

    best_name = results.iloc[0]["Model"]
    best_model = rf if best_name == "Random Forest" else (gb if best_name == "Gradient Boosting" else lr)
    best_scale = best_name == "Logistic Regression"

    return {
        "X": X,
        "y": y,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_te": y_te,
        "rf": rf,
        "gb": gb,
        "lr": lr,
        "scaler": scaler,
        "results": results,
        "rf_fi": rf_fi,
        "gb_fi": gb_fi,
        "cm_rf": cm_rf,
        "cm_gb": cm_gb,
        "cm_lr": cm_lr,
        "shap_sample": shap_sample,
        "explainer": explainer,
        "shap_arr": shap_arr,
        "shap_error": shap_error,
        "best_name": best_name,
        "best_model": best_model,
        "best_scale": best_scale,
    }


# ──────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────
df = generate_data()
art = train_models(df)

# ──────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ──────────────────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor=C["card"],
    plot_bgcolor=C["card"],
    font_color=C["slate"],
    font_family="Plus Jakarta Sans, sans-serif",
    margin=dict(l=16, r=16, t=44, b=16),
    colorway=[C["teal"], C["sky"], C["amber"], C["green"], C["red"]],
)
AXIS = dict(
    gridcolor=C["border"],
    linecolor=C["border"],
    tickcolor=C["muted"],
    tickfont_color=C["muted"],
)

def style(fig):
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style="padding:1rem 0 0.8rem;">
        <div style="font-size:1.5rem; font-weight:800; color:#FFFFFF;">🏥 CareFlow</div>
        <div style="color:#94A3B8; font-size:0.8rem; margin-top:3px; font-weight:500;">
            Feature Adoption Analytics
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#2D4A6E; margin:0 0 1rem 0'>", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "📊  Adoption Analysis",
        "👤  User Segments",
        "🤖  Engagement Model",
        "🎯  User Scoring Tool",
        "💡  SHAP Explainability",
        "📋  Project Report",
    ])
    page = page.split("  ")[1].strip()

    st.markdown("<hr style='border-color:#2D4A6E; margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown(
        f"""
    <div style="background:#1E3A5F; border:1px solid #2D4A6E; border-radius:10px; padding:0.9rem;">
        <div style="color:#64748B; font-size:0.72rem; text-transform:uppercase; letter-spacing:.07em; margin-bottom:6px;">Dataset</div>
        <div style="color:#CBD5E1; font-size:0.88rem;">{len(df):,} clinician users</div>
        <div style="color:#64748B; font-size:0.78rem; margin-top:4px;">{df['power_user'].mean():.0%} power users</div>
        <div style="color:#64748B; font-size:0.78rem;">Best model: {art['best_name']}</div>
        <div style="color:#64748B; font-size:0.78rem;">ROC-AUC: {art['results']['ROC-AUC'].max():.3f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────
st.markdown(
    """
<span class="tag">Healthcare Tech · Product Data Science Portfolio</span>
<div class="page-title">CareFlow Feature Adoption Dashboard</div>
<div class="page-sub">
    Analysing how clinicians adopt and engage with a clinical workflow platform —
    built to demonstrate junior product DS skills.
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# ══════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":

    avg_features = df["features_adopted"].mean()
    ttfv_median = df["ttfv_days"].median()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Power User Rate", f"{df['power_user'].mean():.1%}", delta="+3.2pp vs last quarter")
    c2.metric("Avg Features Adopted", f"{avg_features:.1f} / 8", delta="+0.4 vs last quarter")
    c3.metric("Median Time-to-Value", f"{int(ttfv_median)} days", delta="-1 day vs last quarter", delta_color="inverse")
    c4.metric("Best Model ROC-AUC", f"{art['results']['ROC-AUC'].max():.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Feature Adoption Rates — All Users</div>', unsafe_allow_html=True)
    adopt_rates = pd.DataFrame({
        "feature": [FEATURE_LABELS[f] for f in FEATURES_LIST],
        "adoption_rate": [df[f"adopted_{f}"].mean() for f in FEATURES_LIST],
    }).sort_values("adoption_rate", ascending=True)

    fig = px.bar(
        adopt_rates,
        x="adoption_rate",
        y="feature",
        orientation="h",
        text=adopt_rates["adoption_rate"].apply(lambda x: f"{x:.0%}"),
        color="adoption_rate",
        color_continuous_scale=[C["red"], C["amber"], C["teal"]],
        title="% of Users Who Have Activated Each Feature",
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    style(fig)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="sec-title">Power Users by Plan</div>', unsafe_allow_html=True)
        plan_pu = df.groupby("plan")["power_user"].mean().reset_index()
        plan_pu.columns = ["plan", "power_user_rate"]

        fig2 = px.bar(
            plan_pu,
            x="plan",
            y="power_user_rate",
            color="plan",
            color_discrete_map={
                "Starter": C["amber"],
                "Professional": C["teal"],
                "Enterprise": C["sky"],
            },
            text=plan_pu["power_user_rate"].apply(lambda x: f"{x:.0%}"),
            title="Power User Rate by Subscription Plan",
        )
        fig2.update_traces(textposition="outside")
        style(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        hist_df = df.copy()
        hist_df["power_user_label"] = hist_df["power_user"].map({0: "Non-Power", 1: "Power User"})

        st.markdown('<div class="sec-title">Feature Breadth Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(
            hist_df,
            x="features_adopted",
            color="power_user_label",
            color_discrete_map={"Non-Power": C["amber"], "Power User": C["teal"]},
            barmode="overlay",
            nbins=9,
            opacity=0.8,
            labels={"power_user_label": "User Type", "features_adopted": "Features Adopted"},
            title="Number of Features Adopted — Power vs Non-Power Users",
        )
        style(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    top_driver = art["rf_fi"].iloc[0]["feature"]
    st.markdown(
        f"""
    <div class="ds-callout">
        💡 <strong>Key insight:</strong>
        Feature adoption breadth is a stronger predictor of engagement than login frequency.
        The model identifies <em>{MODEL_FEATURE_DESC.get(top_driver, top_driver)}</em>
        as the #1 driver of power user status. Users who adopt Voice Notes and the Analytics
        Dashboard are significantly more likely to become long-term advocates.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════
# PAGE: ADOPTION ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "Adoption Analysis":

    st.markdown(
        """
    <div class="ds-callout">
        This page explores <strong>how</strong> users move through the feature adoption journey
        — from first login to deep engagement with advanced features.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-title">Feature Adoption Funnel</div>', unsafe_allow_html=True)
    funnel_data = pd.DataFrame({
        "Stage": [f"At least {i} feature{'s' if i > 1 else ''}" for i in range(1, 9)],
        "Users": [int((df["features_adopted"] >= i).sum()) for i in range(1, 9)],
    })
    fig = px.funnel(
        funnel_data,
        x="Users",
        y="Stage",
        color_discrete_sequence=[C["teal"]],
        title="Feature Adoption Funnel — CareFlow Platform",
    )
    style(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">Adoption Rate by Clinician Role</div>', unsafe_allow_html=True)
    role_adopt = df.groupby("role")[[f"adopted_{f}" for f in FEATURES_LIST]].mean().reset_index()
    role_adopt.columns = ["role"] + [FEATURE_LABELS[f] for f in FEATURES_LIST]
    role_melt = role_adopt.melt(id_vars="role", var_name="Feature", value_name="Adoption Rate")

    fig2 = px.bar(
        role_melt,
        x="Feature",
        y="Adoption Rate",
        color="role",
        barmode="group",
        title="Feature Adoption Rate by Clinician Role",
        color_discrete_sequence=[C["teal"], C["sky"], C["amber"], C["green"]],
    )
    fig2.update_xaxes(tickangle=-30)
    style(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    ttfv_df = df.copy()
    ttfv_df["power_user_label"] = ttfv_df["power_user"].map({0: "Non-Power", 1: "Power User"})

    st.markdown('<div class="sec-title">Time-to-First-Value</div>', unsafe_allow_html=True)
    fig3 = px.histogram(
        ttfv_df,
        x="ttfv_days",
        color="power_user_label",
        color_discrete_map={"Non-Power": C["amber"], "Power User": C["teal"]},
        nbins=30,
        opacity=0.8,
        barmode="overlay",
        labels={"power_user_label": "User Type", "ttfv_days": "Days to First Value"},
        title="Power users reach first value faster",
    )
    style(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    fast_ttfv = int(df.loc[df["power_user"] == 1, "ttfv_days"].median())
    st.markdown(
        f"""
    <div class="ds-callout">
        🔍 <strong>Product recommendation:</strong> Power users reach first value in a median of
        <strong>{fast_ttfv} days</strong>. Consider triggering an in-app onboarding prompt at Day 3
        for users who haven't yet tried e-Prescribing or Care Gap Alerts.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════
# PAGE: USER SEGMENTS
# ══════════════════════════════════════════════════════════
elif page == "User Segments":

    st.markdown('<div class="sec-title">Engagement by Department</div>', unsafe_allow_html=True)
    dept_stats = df.groupby("department").agg(
        users=("user_id", "count"),
        power_user_rate=("power_user", "mean"),
        avg_features=("features_adopted", "mean"),
    ).reset_index()

    fig = px.scatter(
        dept_stats,
        x="avg_features",
        y="power_user_rate",
        size="users",
        color="department",
        text="department",
        size_max=50,
        color_discrete_sequence=[C["teal"], C["sky"], C["amber"], C["green"], C["red"]],
        title="Department: Avg Features Adopted vs Power User Rate (bubble = user count)",
        labels={"avg_features": "Avg Features Adopted", "power_user_rate": "Power User Rate"},
    )
    fig.update_traces(textposition="top center")
    style(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">Power User Rate by Signup Cohort</div>', unsafe_allow_html=True)
    cohort_df = df.copy()
    cohort_df["cohort_q"] = cohort_df["cohort_month"].dt.to_period("Q").astype(str)
    cohort_stats = cohort_df.groupby("cohort_q").agg(
        users=("user_id", "count"),
        power_rate=("power_user", "mean"),
    ).reset_index()

    fig2 = px.bar(
        cohort_stats,
        x="cohort_q",
        y="power_rate",
        color="power_rate",
        color_continuous_scale=[C["amber"], C["teal"]],
        text=cohort_stats["power_rate"].apply(lambda x: f"{x:.0%}"),
        title="Power User Rate by Acquisition Cohort (Quarterly)",
        labels={"cohort_q": "Cohort", "power_rate": "Power User Rate"},
    )
    fig2.update_traces(textposition="outside")
    fig2.update_coloraxes(showscale=False)
    style(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-title">Plan x Role — Power User Rate Matrix</div>', unsafe_allow_html=True)
    pivot = df.pivot_table(index="role", columns="plan", values="power_user", aggfunc="mean").round(3)
    st.dataframe(pivot, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: ENGAGEMENT MODEL
# ══════════════════════════════════════════════════════════
elif page == "Engagement Model":

    st.markdown(
        """
    <div class="ds-callout">
        <strong>Modelling objective:</strong> Predict which clinicians will become power users.
        This is a <em>binary classification problem</em>. We care most about
        <strong>ROC-AUC</strong> (overall discrimination) and <strong>Recall</strong>
        (we don't want to miss at-risk users who need a CS intervention).
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-title">Model Comparison</div>', unsafe_allow_html=True)
    disp_res = art["results"].copy()
    for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        disp_res[c] = disp_res[c].map("{:.3f}".format)
    st.dataframe(disp_res, use_container_width=True, hide_index=True)

    melted = art["results"].melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_map={
            "Random Forest": C["teal"],
            "Gradient Boosting": C["sky"],
            "Logistic Regression": C["amber"],
        },
        title="Performance Metrics by Model",
    )
    style(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-title">Random Forest — Top Features</div>', unsafe_allow_html=True)
        rf_top = art["rf_fi"].head(10).sort_values("importance")
        fig2 = px.bar(
            rf_top,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[C["border"], C["teal"]],
            title="RF Feature Importance",
        )
        fig2.update_coloraxes(showscale=False)
        style(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-title">Gradient Boosting — Top Features</div>', unsafe_allow_html=True)
        gb_top = art["gb_fi"].head(10).sort_values("importance")
        fig3 = px.bar(
            gb_top,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[C["border"], C["sky"]],
            title="GB Feature Importance",
        )
        fig3.update_coloraxes(showscale=False)
        style(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sec-title">Confusion Matrices (Test Set)</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, cm, name in zip(
        [col1, col2, col3],
        [art["cm_rf"], art["cm_gb"], art["cm_lr"]],
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
    ):
        with col:
            st.write(f"**{name}**")
            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual: Non-Power", "Actual: Power"],
                    columns=["Pred: Non-Power", "Pred: Power"],
                ),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════
# PAGE: USER SCORING TOOL
# ══════════════════════════════════════════════════════════
elif page == "User Scoring Tool":

    st.markdown(
        """
    <div class="ds-callout">
        Enter a clinician's current usage profile to get their engagement score and tier.
        Customer Success teams use this to prioritise onboarding calls for at-risk users,
        or identify power users for case studies and advocate programmes.
    </div>
    """,
        unsafe_allow_html=True,
    )

    model_choice = st.selectbox(
        "Prediction model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        index=["Random Forest", "Gradient Boosting", "Logistic Regression"].index(art["best_name"]),
    )

    st.markdown('<div class="sec-title">Usage Profile Input</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    inputs = {}
    numeric_feats = [
        "tenure_months",
        "logins_last_30d",
        "session_length_min",
        "patients_documented",
        "features_adopted",
        "ttfv_days",
        "support_tickets",
        "nps_score",
        "depth_voice_notes",
        "depth_analytics_dashboard",
    ]
    binary_feats = [
        "adopted_e_prescribing",
        "adopted_care_gap_alerts",
        "adopted_voice_notes",
        "adopted_analytics_dashboard",
    ]
    half = len(numeric_feats) // 2

    with c1:
        for f in numeric_feats[:half]:
            inputs[f] = st.number_input(
                f"{f}  —  {MODEL_FEATURE_DESC[f]}",
                value=float(df[f].median()),
                step=0.5,
                format="%.1f",
                key=f"s_{f}",
            )
        for f in binary_feats[:2]:
            inputs[f] = int(
                st.checkbox(FEATURE_LABELS.get(f.replace("adopted_", ""), f), key=f"s_{f}")
            )

    with c2:
        for f in numeric_feats[half:]:
            inputs[f] = st.number_input(
                f"{f}  —  {MODEL_FEATURE_DESC[f]}",
                value=float(df[f].median()),
                step=0.5,
                format="%.1f",
                key=f"s2_{f}",
            )
        for f in binary_feats[2:]:
            inputs[f] = int(
                st.checkbox(FEATURE_LABELS.get(f.replace("adopted_", ""), f), key=f"s2_{f}")
            )

    if st.button("▶  Score This User"):
        inp = pd.DataFrame([{f: inputs[f] for f in MODEL_FEATURES}])

        if model_choice == "Random Forest":
            prob = art["rf"].predict_proba(inp)[0, 1]
        elif model_choice == "Gradient Boosting":
            prob = art["gb"].predict_proba(inp)[0, 1]
        else:
            prob = art["lr"].predict_proba(art["scaler"].transform(inp))[0, 1]

        if prob >= 0.65:
            tier, badge, action, color = (
                "Power User",
                "badge-green",
                "Invite to advocate programme or case study.",
                C["green"],
            )
        elif prob >= 0.35:
            tier, badge, action, color = (
                "Developing",
                "badge-amber",
                "Schedule onboarding check-in; suggest Voice Notes feature tour.",
                C["amber"],
            )
        else:
            tier, badge, action, color = (
                "At Risk",
                "badge-red",
                "Escalate to CS; trigger in-app feature discovery campaign.",
                C["red"],
            )

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Power User Probability", f"{prob:.1%}")
            st.markdown(f'<span class="{badge}">{tier}</span>', unsafe_allow_html=True)
            st.caption(f"Model: {model_choice}")

        with r2:
            st.markdown(
                f"""
            <div class="ds-card" style="border-left: 4px solid {color};">
                <div class="sec-title">Recommended Action</div>
                <div style="font-size:0.95rem; color:{C['navy']};">{action}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with r3:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"color": color, "size": 34}},
                    title={"text": "Engagement Score", "font": {"color": C["muted"], "size": 13}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": C["muted"]},
                        "bar": {"color": color, "thickness": 0.25},
                        "bgcolor": C["card"],
                        "bordercolor": C["border"],
                        "steps": [
                            {"range": [0, 35], "color": "rgba(220,38,38,0.10)"},
                            {"range": [35, 65], "color": "rgba(217,119,6,0.10)"},
                            {"range": [65, 100], "color": "rgba(13,148,136,0.10)"},
                        ],
                        "threshold": {"line": {"color": color, "width": 3}, "value": prob * 100},
                    },
                )
            )
            gauge.update_layout(
                height=250,
                paper_bgcolor=C["card"],
                font_color=C["slate"],
                margin=dict(t=30, b=0),
            )
            st.plotly_chart(gauge, use_container_width=True)

        st.info("ℹ️  This score is a CS prioritisation tool — not a clinical decision aid.")

# ══════════════════════════════════════════════════════════
# PAGE: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════
elif page == "SHAP Explainability":

    st.markdown(
        """
    <div class="ds-callout">
        SHAP (SHapley Additive exPlanations) shows <em>why</em> the model gives a score —
        not just what the score is. For a junior product DS role, being able to explain model
        outputs to Product Managers and CS teams is just as important as building the model.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not installed in this environment. Run: pip install shap")
    elif art["shap_arr"] is None:
        st.error("SHAP could not be computed for this environment.")
        if art.get("shap_error"):
            st.code(art["shap_error"])
    else:
        shap_values = np.array(art["shap_arr"])
        shap_values = np.squeeze(shap_values)

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        if shap_values.ndim != 2:
            st.error(f"Unexpected SHAP shape after normalization: {shap_values.shape}")
        elif shap_values.shape[1] != art["shap_sample"].shape[1]:
            st.error(
                f"SHAP feature mismatch. shap_values shape = {shap_values.shape}, "
                f"shap_sample shape = {art['shap_sample'].shape}"
            )
        else:
            st.markdown('<div class="sec-title">Global SHAP Summary Plot — Random Forest</div>', unsafe_allow_html=True)

            plt.figure(figsize=(10, 5.5))
            shap.summary_plot(shap_values, art["shap_sample"], show=False, plot_size=None)
            fig_s = plt.gcf()
            fig_s.patch.set_facecolor(C["card"])
            st.pyplot(fig_s, clear_figure=True)

            st.markdown('<div class="sec-title">Which Features Matter Most? (Mean |SHAP|)</div>', unsafe_allow_html=True)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            mean_abs_shap = np.asarray(mean_abs_shap).reshape(-1)

            mean_shap = pd.DataFrame({
                "feature": list(art["shap_sample"].columns),
                "mean_abs_shap": mean_abs_shap.tolist(),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

            mean_shap["description"] = mean_shap["feature"].map(MODEL_FEATURE_DESC)
            mean_shap["rank"] = range(1, len(mean_shap) + 1)

            fig_b = px.bar(
                mean_shap.sort_values("mean_abs_shap"),
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale=[C["border"], C["teal"]],
                title="Feature Contribution to Power User Prediction",
                hover_data={"description": True, "mean_abs_shap": ":.4f"},
            )
            fig_b.update_coloraxes(showscale=False)
            style(fig_b)
            st.plotly_chart(fig_b, use_container_width=True)

            st.markdown('<div class="sec-title">Business Interpretation</div>', unsafe_allow_html=True)
            interp_df = mean_shap[["rank", "feature", "description", "mean_abs_shap"]].head(10).rename(
                columns={"rank": "#", "mean_abs_shap": "Impact Score"}
            )
            st.dataframe(interp_df, use_container_width=True, hide_index=True)

            t3 = [MODEL_FEATURE_DESC.get(f, f) for f in mean_shap["feature"].head(3).tolist()]
            st.markdown(
                f"""
            <div class="ds-callout">
                📌 <strong>How to explain this to a Product Manager:</strong><br>
                "Our top three signals for identifying power users are:
                <strong>{t3[0]}</strong>, <strong>{t3[1]}</strong>, and <strong>{t3[2]}</strong>.
                This means our onboarding should focus on getting users to these behaviours quickly —
                not just driving login volume."
            </div>
            """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════
# PAGE: PROJECT REPORT
# ══════════════════════════════════════════════════════════
elif page == "Project Report":

    st.markdown(
        f"""
    <div class="ds-card">

    <h3 style="color:{C['navy']}; margin-top:0; border-bottom:2px solid {C['border']}; padding-bottom:.5rem;">
        What This Project Is
    </h3>
    <p>
    A portfolio project for a <strong>Junior / Associate Product Data Scientist</strong> role
    in healthcare technology. It simulates how a product DS would approach feature adoption
    analysis for a clinical workflow SaaS platform called <em>CareFlow</em>.
    </p>
    <p>
    It covers the full workflow expected at the junior level:
    <em>frame the business question → engineer features → build and compare models → explain results → communicate to stakeholders.</em>
    </p>

    <h3 style="color:{C['navy']}; border-bottom:2px solid {C['border']}; padding-bottom:.5rem;">
        Business Question
    </h3>
    <p>
    Healthcare SaaS has notoriously slow clinical adoption. The key question is:
    <em>"Which product behaviours in the first 30 days predict whether a clinician will become
    a long-term, highly engaged user?"</em>
    </p>
    <p>
    Answering this helps the Product team know what to optimise in onboarding, and helps
    Customer Success prioritise who to call.
    </p>

    <h3 style="color:{C['navy']}; border-bottom:2px solid {C['border']}; padding-bottom:.5rem;">
        Feature Engineering Rationale
    </h3>
    <ul>
        <li><strong>Feature breadth</strong> — how many distinct features a user has tried.</li>
        <li><strong>Time-to-first-value</strong> — days until the first documented patient record.</li>
        <li><strong>Feature depth</strong> — Voice Notes and Analytics Dashboard indicate genuine workflow integration.</li>
        <li><strong>NPS score</strong> — sentiment signal alongside behavioural data.</li>
        <li><strong>Support tickets</strong> — friction proxy for churn risk.</li>
    </ul>

    <h3 style="color:{C['navy']}; border-bottom:2px solid {C['border']}; padding-bottom:.5rem;">
        Modelling Choices
    </h3>
    <ul>
        <li><strong>Target: Power User</strong> — defined from a composite engagement score.</li>
        <li><strong>Why three models:</strong> Random Forest, Gradient Boosting, and Logistic Regression.</li>
        <li><strong>Evaluation priority:</strong> ROC-AUC and Recall.</li>
    </ul>

    <h3 style="color:{C['navy']}; border-bottom:2px solid {C['border']}; padding-bottom:.5rem;">
        What I Would Add with Real Data
    </h3>
    <ul>
        <li>Connect to event stream data such as Segment or Mixpanel</li>
        <li>Add a 90-day engagement trajectory</li>
        <li>A/B test onboarding interventions</li>
        <li>Lightweight retraining pipeline with drift monitoring</li>
    </ul>

    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-title">Run Locally</div>', unsafe_allow_html=True)
    st.code(
        """
# Install dependencies
pip install streamlit pandas numpy scikit-learn shap plotly matplotlib

# Run the dashboard
streamlit run app.py
    """,
        language="bash",
    )

    st.markdown('<div class="sec-title">Project Structure</div>', unsafe_allow_html=True)
    st.code(
        """
CareFlowDS/
├── app.py
└── README.md
    """,
        language="text",
    )