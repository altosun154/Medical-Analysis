import pandas as pd
import plotly.express as px


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
