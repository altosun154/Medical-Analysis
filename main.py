import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pointbiserialr
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

st.set_page_config(page_title="Cancer Diagnosis Data Analysis", layout="wide")
st.title("🩺 Cancer Diagnosis Data Analysis Program")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV file skipping first row (header=1 like your original code)
    df = pd.read_csv(uploaded_file, header=1)
    st.success("✅ Dataset loaded successfully!")

    # Show preview of the data
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Stats for each column
    # -------------------------------
    st.subheader("📊 Column Statistics & Graphs")
    for col in df.columns:
        if col == "Patient_ID":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            st.markdown(f"**{col}** (Numeric)")
            st.write(f"Count: {df[col].count()}")
            st.write(f"Mean: {df[col].mean():.2f}")
            st.write(f"Median: {df[col].median():.2f}")
            st.write(f"Std Dev: {df[col].std():.2f}")
            st.write(f"Min: {df[col].min()}")
            st.write(f"Max: {df[col].max()}")

            # Histogram
            fig = px.histogram(df, x=col, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)

            # Boxplot
            fig = px.box(df, y=col, title=f"Boxplot of {col}")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(f"**{col}** (Categorical)")
            st.write(f"Count: {df[col].count()}")
            st.write(f"Unique values: {df[col].nunique()}")
            st.write(f"Most common: {df[col].mode()[0]}")
            st.write(df[col].value_counts().head(5))

            # Bar chart
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']
            fig = px.bar(value_counts, x=col, y='count', title=f"Bar chart of {col}")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Survival Status vs Numeric Variables
    # -------------------------------
    st.subheader("📈 Survival Analysis Settings")

# Check if FollowUp_Months column already exists
    if "FollowUp_Months" in df.columns:
        st.success("✅ Found 'FollowUp_Months' column in uploaded dataset.")
        followup_df = df.copy()
    
    else:
        st.warning("⚠ No 'FollowUp_Months' column found in dataset.")
        followup_option = st.radio(
            "How do you want to provide follow-up times?",
            ["Use default time for all patients",
             "Upload CSV with follow-up times",
             "Manually enter per-patient times"]
        )
    
        if followup_option == "Use default time for all patients":
            default_time = st.number_input(
                "Enter default follow-up time (months):",
                min_value=0.0, step=1.0, value=36.0
            )
            df["FollowUp_Months"] = default_time
            followup_df = df.copy()
    
        elif followup_option == "Upload CSV with follow-up times":
            uploaded_followup = st.file_uploader(
                "Upload CSV with Patient_ID and FollowUp_Months",
                type=["csv"]
            )
            if uploaded_followup is not None:
                followup_data = pd.read_csv(uploaded_followup)
                if "Patient_ID" in followup_data.columns and "FollowUp_Months" in followup_data.columns:
                    followup_df = df.merge(
                        followup_data[["Patient_ID", "FollowUp_Months"]],
                        on="Patient_ID",
                        how="left"
                    )
                    st.success("✅ Follow-up times merged successfully!")
                else:
                    st.error("❌ CSV must contain 'Patient_ID' and 'FollowUp_Months' columns.")
    
        elif followup_option == "Manually enter per-patient times":
            followup_times = []
            for pid in df["Patient_ID"]:
                time = st.number_input(
                    f"Follow-up time (months) for Patient {pid}:",
                    min_value=0.0, step=1.0, value=36.0,
                    key=f"time_{pid}"
                )
                followup_times.append(time)
            df["FollowUp_Months"] = followup_times
            followup_df = df.copy()



    # Kaplan-Meier survival analysis
    st.subheader("📉 Kaplan–Meier Survival Curve")
    kmf = KaplanMeierFitter()
    kmf.fit(df["FollowUp_Months"], event_observed=df[event_col], label="Survival Probability")
    
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)
    
    # Cox Proportional Hazards model
    st.subheader("📊 Cox Regression Analysis")
    cox_df = df[["FollowUp_Months", event_col, "Age", "Tumor_Size(cm)"]].dropna()
    if not cox_df.empty:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time", event_col=event_col)
        st.write(cph.summary)
    else:
        st.warning("Not enough data for Cox regression.")
    # -------------------------------
    # Survival by Treatment
    # -------------------------------
    st.subheader("📊 Survival Status by Treatment")
    survival_by_treatment = df.groupby(["Treatment", "Survival_Status"]).size().reset_index(name="Count")
    fig = px.bar(
        survival_by_treatment,
        x="Treatment",
        y="Count",
        color="Survival_Status",
        barmode="group",
        title="Survival Status by Treatment"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Age Distribution by Survival Status
    # -------------------------------
    st.subheader("📊 Age Distribution by Survival Status")
    fig = px.box(
        df,
        x="Survival_Status",
        y="Age",
        title="Age Distribution by Survival Status",
        points="all"
    )
    st.plotly_chart(fig, use_container_width=True)
