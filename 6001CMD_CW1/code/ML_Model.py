import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(
    r"C:\Users\User\OneDrive\Desktop\INTI\Sem 8\Machine Learning\6001CMD_CW1\PhiUSIIL_Phishing_URL_Dataset.csv"
)

# Check the total number of TLD (Top Level Domain) unique values in the dataset
print("TLD unique values:")
print(df["TLD"].nunique())

# Check the most common TLDs in the dataset
print("\nMost common TLDs:")
print(df["TLD"].value_counts().head(20))


# Separate features and target
drop_columns = [
    "FILENAME",
    "URL",
    "Domain",
    "TLD",
    "Title"
]

# X = Features (all columns except the target and dropped columns)
# y = Target (the "label" column)
X = df.drop(columns=drop_columns + ["label"])
y = df["label"]

# Check the shapes of the features and target
print("X shape:", X.shape)
print("y shape:", y.shape)

# Check all the feature columns
print("\nFeature columns:")
print(X.columns.tolist())


#=======================================================

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y # Stratified sampling to maintain the same class distribution in both training and testing sets
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting label distribution:")
print(y_test.value_counts(normalize=True) * 100)


#=======================================================
# Create preprocessing + model pipeline
model = Pipeline([
    ("scaler", RobustScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Make predictions
y_pred = model.predict(X_test)

# Probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Phishing", "Legitimate"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Get Logistic Regression coefficients
coefficients = model.named_steps["classifier"].coef_[0]

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": coefficients,
    "Absolute_Coefficient": abs(coefficients)
})

feature_importance = feature_importance.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\nTop 15 most influential features:")
print(feature_importance.head(15))

from sklearn.metrics import precision_score, recall_score, f1_score

phishing_precision = precision_score(
    y_test, y_pred, pos_label=0
)

phishing_recall = recall_score(
    y_test, y_pred, pos_label=0
)

phishing_f1 = f1_score(
    y_test, y_pred, pos_label=0
)

print("Phishing Precision:", phishing_precision)
print("Phishing Recall:", phishing_recall)
print("Phishing F1-score:", phishing_f1)

#=======================================================
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully.")

# Make predictions
rf_pred = rf_model.predict(X_test)

# Probability predictions
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, pos_label=0)
rf_recall = recall_score(y_test, rf_pred, pos_label=0)
rf_f1 = f1_score(y_test, rf_pred, pos_label=0)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("\nRandom Forest Results")
print("--------------------")
print("Accuracy:", rf_accuracy)
print("Phishing Precision:", rf_precision)
print("Phishing Recall:", rf_recall)
print("Phishing F1-score:", rf_f1)
print("ROC-AUC:", rf_roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Random Forest feature importance

rf_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Random Forest Features:")
print(rf_importance.head(15))

# --------------------------------------------------
# Feature Ablation Experiment


ablation_features = [
    "URLSimilarityIndex",
    "NoOfExternalRef",
    "LineOfCode",
    "NoOfSelfRef",
    "NoOfImage",
]

X_ablation = X.drop(columns=ablation_features)

X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(
    X_ablation,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create a new Random Forest
rf_ablation = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train
rf_ablation.fit(X_train_ab, y_train_ab)

# Predict
ab_pred = rf_ablation.predict(X_test_ab)
ab_prob = rf_ablation.predict_proba(X_test_ab)[:, 1]

# Evaluate
ab_accuracy = accuracy_score(y_test_ab, ab_pred)
ab_precision = precision_score(y_test_ab, ab_pred, pos_label=0)
ab_recall = recall_score(y_test_ab, ab_pred, pos_label=0)
ab_f1 = f1_score(y_test_ab, ab_pred, pos_label=0)
ab_roc_auc = roc_auc_score(y_test_ab, ab_prob)

print("\nFeature Ablation Results")
print("------------------------")
print("Removed:", ablation_features)
print("Accuracy:", ab_accuracy)
print("Phishing Precision:", ab_precision)
print("Phishing Recall:", ab_recall)
print("Phishing F1-score:", ab_f1)
print("ROC-AUC:", ab_roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_ab, ab_pred))