import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\INTI\Sem 8\Machine Learning\6001CMD_CW1\PhiUSIIL_Phishing_URL_Dataset.csv")

# Check how many rows and columns in the dataset
print("Dataset Rows and Columns:")
print(df.shape)

# Show the first 5 rows of the dataset
print("\nFirst 5 Rows:")
print(df.head())

# Show the column names of the dataset
print("\nColumn Names:")
print(df.columns)

# Show the overall information of the dataset
print("\nDataset Information:")
print(df.info())

# Calculate the summary statistics of the dataset
print("\nSummary Statistics:")
print(df.describe())

# Double confirm if there are any missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Check the distribution of the target variable
print("\nTarget Variable Distribution:")
print(df["label"].value_counts())

# Show the distribution of the target variable as percentages
print("\nTarget Variable Percentages:")
print(df["label"].value_counts(normalize=True) * 100)


# EDA - Numerical Feature Analysis
# Select numerical columns
numeric_df = df.select_dtypes(include=np.number)
print("\nNumerical columns:")
print(numeric_df.columns.tolist())

# Calculate skewness
skewness = numeric_df.skew()
print("\nSkewness:")
print(skewness.sort_values(ascending=False))


# Outlier Analysis using IQR
outlier_results = []
# For each numerical column, calculate the IQR and identify outliers
for col in numeric_df.columns:
    Q1 = numeric_df[col].quantile(0.25) 
    Q3 = numeric_df[col].quantile(0.75) 

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers based on the calculated bounds
    outliers = (
        (numeric_df[col] < lower_bound) |
        (numeric_df[col] > upper_bound)
    )

    # Count the number of outliers and calculate the percentage of outliers
    outlier_count = outliers.sum()
    outlier_percentage = (outlier_count / len(numeric_df)) * 100

    # Store the results in a list of dictionaries for later display
    outlier_results.append({
        "Feature": col,
        "Outlier Count": outlier_count,
        "Outlier Percentage": outlier_percentage
    })

# Create a DataFrame to display the outlier analysis results
outlier_df = pd.DataFrame(outlier_results)

# Show outlier analysis results
print("\nOutlier Analysis:")
print(
    outlier_df
    .sort_values("Outlier Percentage", ascending=False)
    .to_string(index=False)
)

# Correlation Analysis
correlation_matrix = numeric_df.corr()

print("\nCorrelation with target label:")

target_correlation = (
    correlation_matrix["label"]
    .drop("label")
    .sort_values(key=abs, ascending=False)
)

print(target_correlation)


# ==========================================
# Figure 1 - Target Class Distribution
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="label")

plt.title("Target Class Distribution")
plt.xlabel("Label (0 = Phishing, 1 = Legitimate)")
plt.ylabel("Number of URLs")

plt.tight_layout()

plt.savefig("Figure1_Target_Class_Distribution.png", dpi=300)

plt.show()

# ==========================================
# Figure 2 - Correlation Heatmap
# ==========================================

top_features = target_correlation.head(10).index.tolist()

heatmap_columns = top_features + ["label"]

plt.figure(figsize=(10, 8))

sns.heatmap(
    df[heatmap_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Key Features")

plt.tight_layout()

plt.savefig("Figure2_Correlation_Heatmap.png", dpi=300)

plt.show()

# ==========================================
# Figure 3 - Boxplots
# ==========================================

boxplot_features = [
    "NoOfSubDomain",
    "Pay",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "IsHTTPS",
    "NoOfEmptyRef"
]

plt.figure(figsize=(12, 7))

sns.boxplot(data=df[boxplot_features])

plt.xticks(rotation=45)

plt.title("Boxplots of Selected Numerical Features")
plt.ylabel("Value")

plt.tight_layout()

plt.savefig("Figure3_Boxplots.png", dpi=300)

plt.show()