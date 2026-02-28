import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Clinical Survival Analysis",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Clinical Survival Analysis Dashboard")
st.markdown("**Predicting patient survival outcomes using Cox Proportional Hazards modeling**")

# -------------------------
# Generate Sample Data
# -------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'age': np.random.randint(30, 85, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'diagnosis': np.random.choice(['Type A', 'Type B', 'Type C'], n),
        'comorbidities': np.random.randint(0, 5, n),
        'treatment': np.random.choice(['Treatment 1', 'Treatment 2', 'Control'], n),
        'duration': np.random.exponential(365, n).astype(int),
        'event_observed': np.random.choice([0, 1], n, p=[0.4, 0.6])
    })
    df['risk_group'] = pd.cut(df['age'], bins=[0, 50, 65, 100],
                               labels=['Low Risk', 'Medium Risk', 'High Risk'])
    return df

df = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("🔍 Filter Patients")
gender_filter = st.sidebar.multiselect("Gender", df['gender'].unique(),
                                        default=df['gender'].unique())
diagnosis_filter = st.sidebar.multiselect("Diagnosis", df['diagnosis'].unique(),
                                           default=df['diagnosis'].unique())
treatment_filter = st.sidebar.multiselect("Treatment", df['treatment'].unique(),
                                           default=df['treatment'].unique())

filtered_df = df[
    df['gender'].isin(gender_filter) &
    df['diagnosis'].isin(diagnosis_filter) &
    df['treatment'].isin(treatment_filter)
]

# -------------------------
# KPIs
# -------------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", len(filtered_df))
col2.metric("Events Observed", filtered_df['event_observed'].sum())
col3.metric("Median Age", int(filtered_df['age'].median()))
col4.metric("Avg Duration (days)", int(filtered_df['duration'].mean()))
st.markdown("---")

# -------------------------
# Kaplan-Meier Curves
# -------------------------
st.subheader("📈 Kaplan-Meier Survival Curves by Risk Group")

fig, ax = plt.subplots(figsize=(10, 5))
kmf = KaplanMeierFitter()

colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
for group in filtered_df['risk_group'].dropna().unique():
    mask = filtered_df['risk_group'] == group
    kmf.fit(filtered_df[mask]['duration'],
            filtered_df[mask]['event_observed'],
            label=str(group))
    kmf.plot_survival_function(ax=ax, color=colors.get(str(group), 'blue'), ci_show=True)

ax.set_title("Survival Probability by Risk Group")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Survival Probability")
ax.legend()
st.pyplot(fig)

# -------------------------
# Cox PH Model
# -------------------------
st.subheader("🔬 Cox Proportional Hazards Model")

cox_df = filtered_df.copy()
le = LabelEncoder()
cox_df['gender_enc'] = le.fit_transform(cox_df['gender'])
cox_df['diagnosis_enc'] = le.fit_transform(cox_df['diagnosis'])
cox_df['treatment_enc'] = le.fit_transform(cox_df['treatment'])

cox_features = cox_df[['duration', 'event_observed', 'age',
                         'gender_enc', 'comorbidities',
                         'diagnosis_enc', 'treatment_enc']]

cph = CoxPHFitter()
cph.fit(cox_features, duration_col='duration', event_col='event_observed')

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Model Summary**")
    summary = cph.summary[['coef', 'exp(coef)', 'p']].round(4)
    summary.columns = ['Coefficient', 'Hazard Ratio', 'P-Value']
    summary.index = ['Age', 'Gender', 'Comorbidities', 'Diagnosis', 'Treatment']
    st.dataframe(summary, use_container_width=True)

with col2:
    st.markdown("**Concordance Index**")
    st.metric("C-Index", round(cph.concordance_index_, 3))
    st.markdown("**Interpretation:** A C-Index > 0.7 indicates good model discrimination.")

# -------------------------
# Hazard Ratios Plot
# -------------------------
st.subheader("📊 Hazard Ratios by Feature")
fig2, ax2 = plt.subplots(figsize=(8, 4))
hazard_ratios = cph.summary['exp(coef)']
hazard_ratios.index = ['Age', 'Gender', 'Comorbidities', 'Diagnosis', 'Treatment']
colors_hr = ['red' if hr > 1 else 'green' for hr in hazard_ratios]
hazard_ratios.plot(kind='barh', ax=ax2, color=colors_hr)
ax2.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax2.set_title("Hazard Ratios (>1 = increased risk)")
ax2.set_xlabel("Hazard Ratio")
st.pyplot(fig2)

# -------------------------
# Patient Data Table
# -------------------------
st.subheader("🗂 Patient Data")
st.dataframe(filtered_df.head(50), use_container_width=True)

st.markdown("---")
st.markdown("Built by **Tharun Kumar Reddy Byreddy** | M.S. Statistical Data Science | SFSU")
