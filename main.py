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
with st.sidebar:
    st.header("Settings")
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=1)
    st.success("✅ Dataset loaded successfully!")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Column Statistics",
        "📈 Survival Analysis",
        "📉 Cox Regression",
        "📂 Raw Data"
    ])

    with tab1:
        st.title("📊 Medical Data Analysis Dashboard")

        # Split the main area into two large columns
        col_interactive, col_columnwise = st.columns(2)

# ------------------------- LEFT: INTERACTIVE ANALYSIS -------------------------
        with col_interactive:
            # -------------------------------
            # Stats for each column
            # -------------------------------
            st.subheader("Interactive Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                stat_choices = st.multiselect(
                    "Choose statistical summaries:",
                    ["Mean", "Median", "Std Dev", "Min", "Max"],
                    default=["Mean", "Median"],
                    key="main_stat_choices"
                )
            
            with col2:
                plot_choices = st.multiselect(
                    "Choose plots to display:",
                    ["Histogram", "Box Plot", "Bar Plot"],
                    default=["Histogram"],
                    key="main_plot_choices"
                )
            
            selected_vars = st.multiselect(
                "Choose variables to compare:",
                df.columns.drop("Patient_ID"),
                key="main_var_choices"
            )
            for col in selected_vars:
                st.markdown(f"### {col}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    if "Mean" in stat_choices:
                        st.write(f"Mean: {df[col].mean():.2f}")
                    if "Median" in stat_choices:
                        st.write(f"Median: {df[col].median():.2f}")
                    if "Std Dev" in stat_choices:
                        st.write(f"Std Dev: {df[col].std():.2f}")
                    if "Min" in stat_choices:
                        st.write(f"Min: {df[col].min()}")
                    if "Max" in stat_choices:
                        st.write(f"Max: {df[col].max()}")
            
                    if "Histogram" in plot_choices:
                        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
                        st.plotly_chart(fig, use_container_width=True)
            
                    if "Boxplot" in plot_choices:
                        fig = px.box(df, y=col, title=f"Boxplot of {col}")
                        st.plotly_chart(fig, use_container_width=True)
            
                else:
                    st.write(f"Unique values: {df[col].nunique()}")
                    st.write(f"Most common: {df[col].mode()[0]}")
                    if "Bar Chart" in plot_choices:
                        value_counts = df[col].value_counts().reset_index()
                        value_counts.columns = [col, 'count']
                        fig = px.bar(value_counts, x=col, y='count', title=f"Bar chart of {col}")
                        st.plotly_chart(fig, use_container_width=True)
# ------------------------- RIGHT: COLUMN-WISE SUMMARY -------------------------
        with col_columnwise:                
            st.subheader("Custom Variable Comparison")
    
            # Grouping variable (like Survival_Status)
            group_col = st.selectbox("Choose grouping variable:", df.select_dtypes(include='object').columns)
            
            # Comparison variable(s)
            compare_vars = st.multiselect(
                "Choose variables to compare (categorical or numeric):",
                df.columns.drop(["Patient_ID", group_col])
            )
            
            # Plot type
            plot_type = st.radio("Choose plot type:", ["Bar Plot", "Box Plot", "Histogram"])
            
            # Display plots
            for col in compare_vars:
                st.markdown(f"### {col} by {group_col}")
                
                if plot_type == "Bar Plot" and df[col].dtype == "object":
                    fig = px.histogram(df, x=col, color=group_col, barmode='group')
                    st.plotly_chart(fig)
            
                elif plot_type == "Box Plot" and pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.box(df, x=group_col, y=col, points="all")
                    st.plotly_chart(fig)
            
                elif plot_type == "Histogram" and pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.histogram(df, x=col, color=group_col, barmode="overlay", opacity=0.7)
                    st.plotly_chart(fig)
            
                else:
                    st.warning(f"❌ {col} is not compatible with {plot_type}. Try a different plot type.")


    
    

        with tab2:
        # -------------------------------
        # Survival Status vs Numeric Variables
        # -------------------------------
            with st.sidebar:
                st.subheader("📈 Survival Analysis Settings")
            
                # STEP 1 — Event indicator
                df["Death_Event"] = df["Survival_Status"].apply(
                    lambda x: 1 if str(x).lower() == "deceased" else 0
                )
                event_col = "Death_Event"
            
                # STEP 2 — Get follow-up times
                if "FollowUp_Months" in df.columns:
                    st.success("✅ Found 'FollowUp_Months' in uploaded dataset.")
                    df["time"] = df["FollowUp_Months"]
            
                else:
                    st.warning("⚠ No 'FollowUp_Months' found.")
                    followup_option = st.radio(
                        "How do you want to provide follow-up times?",
                        [
                            "Use default time for all patients",
                            "Upload CSV with follow-up times",
                            "Manually enter per-patient times"
                        ]
                    )
            
                    if followup_option == "Use default time for all patients":
                        default_time = st.number_input(
                            "Enter default follow-up time (months):",
                            min_value=0.0, step=1.0, value=36.0
                        )
                        df["time"] = default_time
            
                    elif followup_option == "Upload CSV with follow-up times":
                        uploaded_followup = st.file_uploader(
                            "Upload CSV with Patient_ID and FollowUp_Months",
                            type=["csv"]
                        )
                        if uploaded_followup is not None:
                            followup_data = pd.read_csv(uploaded_followup)
                            if {"Patient_ID", "FollowUp_Months"}.issubset(followup_data.columns):
                                df = df.merge(
                                    followup_data[["Patient_ID", "FollowUp_Months"]],
                                    on="Patient_ID",
                                    how="left"
                                )
                                df.rename(columns={"FollowUp_Months": "time"}, inplace=True)
                                st.success("✅ Follow-up times merged successfully!")
                            else:
                                st.error("❌ CSV must have 'Patient_ID' and 'FollowUp_Months' columns.")
            
                    elif followup_option == "Manually enter per-patient times":
                        times = []
                        for pid in df["Patient_ID"]:
                            t = st.number_input(
                                f"Follow-up for Patient {pid} (months):",
                                min_value=0.0, step=1.0, value=36.0,
                                key=f"time_{pid}"
                            )
                            times.append(t)
                        df["time"] = times
            # Kaplan–Meier Survival Analysis
            st.subheader("📈 Kaplan–Meier Survival Curve")
            kmf = KaplanMeierFitter()
            
            # Fit using the single, consistent time column
            kmf.fit(df["time"], event_observed=df[event_col], label="Survival Probability")
            
            # Plot
            fig, ax = plt.subplots()
            kmf.plot_survival_function(ax=ax)
            ax.set_title("Kaplan–Meier Survival Curve")
            ax.set_xlabel("Time (months)")
            ax.set_ylabel("Survival Probability")
            st.pyplot(fig)
        with tab3:
            # Prepare Cox model data
            cox_df = df[["time", event_col, "Age", "Tumor_Size(cm)"]].copy()
            
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="time", event_col=event_col)
            
            # Show summary
            st.subheader("📊 Cox Proportional Hazards Model Summary")
            st.write(cph.summary)
        
        with tab4:
            st.subheader("Raw Dataset Preview")
            st.dataframe(df)
            # Load CSV file skipping first row (header=1 like your original code)
            
