import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import numpy as np
print("Welcome to the Cancer Diagnosis Data Analysis Program!")



# Load the dataset
file_name = input("Enter which file you want to analyze (e.g., 'cancer_diagnosis_data.csv'): ")
df = pd.read_csv(file_name, header=1)
print("\nDataset loaded successfully!")



for col in df.columns:
    if col == "Patient_ID":  # or whatever it's called
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"\n📊 Stats for numeric column: {col}")
        print(f"Count: {df[col].count()}")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Std Dev: {df[col].std():.2f}")
        print(f"Min: {df[col].min()}")
        print(f"Max: {df[col].max()}")
        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        fig.show()

        fig = px.box(df, y=col, title=f"Boxplot of {col}")
        fig.show()


    else:
        print(f"\n📋 Stats for categorical column: {col}")
        print(f"Count: {df[col].count()}")
        print(f"Unique values: {df[col].nunique()}")
        print(f"Most common: {df[col].mode()[0]}")
        print(df[col].value_counts().head(5))
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']
        fig = px.bar(value_counts, x=col, y='count', title=f"Bar chart of {col}")
        fig.show()



print("\n📊 Survival Status vs Numeric Variables")
print("-----------------------------------------------------------")

# Create Death_Event numeric flag (1 = Deceased, 0 = Survived)
df["Death_Event"] = df["Survival_Status"].apply(
    lambda x: 1 if str(x).lower() == "deceased" else 0
)

results = []

# Loop through numeric columns except Patient_ID and Death_Event
for col in df.select_dtypes(include=np.number).columns:
    if col in ["Patient_ID", "Death_Event"]:
        continue

    # Means for each survival status
    mean_survived = df.loc[df["Death_Event"] == 0, col].mean()
    mean_deceased = df.loc[df["Death_Event"] == 1, col].mean()

    # Point-biserial correlation
    corr, p_value = pointbiserialr(df[col], df["Death_Event"])

    results.append([
        col,
        round(mean_survived, 2),
        round(mean_deceased, 2),
        round(corr, 4),
        round(p_value, 4)
    ])

# Convert results to DataFrame for pretty printing
results_df = pd.DataFrame(
    results,
    columns=["Variable", "Mean Survived", "Mean Deceased", "Corr", "P-value"]
)

print(results_df.to_string(index=False))

survival_by_treatment = df.groupby(["Treatment", "Survival_Status"]).size().reset_index(name="Count")

fig = px.bar(
    survival_by_treatment,
    x="Treatment",
    y="Count",
    color="Survival_Status",
    barmode="group",
    title="Survival Status by Treatment"
)
fig.show()


fig = px.box(
    df,
    x="Survival_Status",
    y="Age",
    title="Age Distribution by Survival Status",
    points="all"  # shows individual points
)
fig.show()

