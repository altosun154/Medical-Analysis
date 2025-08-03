import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pointbiserialr

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
    st.subheader("📈 Survival Status vs Numeric Variables")

    df["Death_Event"] = df["Survival_Status"].apply(
        lambda x: 1 if str(x).lower() == "deceased" else 0
    )

    results = []
    for col in df.select_dtypes(include=np.number).columns:
        if col in ["Patient_ID", "Death_Event"]:
            continue
        mean_survived = df.loc[df["Death_Event"] == 0, col].mean()
        mean_deceased = df.loc[df["Death_Event"] == 1, col].mean()
        corr, p_value = pointbiserialr(df[col], df["Death_Event"])
        results.append([
            col,
            round(mean_survived, 2),
            round(mean_deceased, 2),
            round(corr, 4),
            round(p_value, 4)
        ])

    results_df = pd.DataFrame(
        results,
        columns=["Variable", "Mean Survived", "Mean Deceased", "Corr", "P-value"]
    )
    st.dataframe(results_df)

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
