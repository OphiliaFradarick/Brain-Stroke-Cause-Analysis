from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, roc_auc_score
import warnings

df = pd.read_csv("cleaned_health_data(in).csv")

# Target and features
y = df['stroke']
X = df.drop(columns='stroke')

# Define categorical and numerical columns [not the binary categorical columns with int data type]
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_cols = ['age', 'bmi', 'avg_glucose_level_transformed']

# Data Transformation - Standardization and One-Hot Encoding
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

# logistic regression
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Predict 
# y_pred = pipeline.predict(X_test)
# report = classification_report(y_test, y_pred, output_dict=True)
# report_df = pd.DataFrame(report).transpose()

# accuracy = report_df.loc["accuracy", "f1-score"]
# precision_stroke = report_df.loc["1", "precision"]

#UNDER SAMPLING
# print(f"Accuracy: {accuracy:.2%}")
# print(f"Precision (Stroke): {precision_stroke:.2%}")

#Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
# disp.plot(cmap="Blues", values_format='d')
# plt.title("Confusion Matrix - Logistic Regression", fontsize=14)
# plt.grid(False)
# plt.tight_layout()
# plt.show()

#OVER SAMPLING
# Step 1: Transform X using the preprocessor
X_processed = preprocessor.fit_transform(X)

# Step 2: Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 4: Train the Logistic Regression model
log_reg_oversampling = LogisticRegression(max_iter=1000)
log_reg_oversampling.fit(X_train, y_train)

# Step 5: Predict and Evaluate
y_pred_oversampling = log_reg_oversampling.predict(X_test)

# Step 6: print accuracy and precision
accuracy = accuracy_score(y_test, y_pred_oversampling)
print(f"Accuracy: {accuracy:.2%}")
precision_stroke = precision_score(y_test, y_pred_oversampling, pos_label=1)
print(f"Precision (Stroke): {precision_stroke:.2%}")


# Confusion Matrix for Oversampling
cm_oversampling = confusion_matrix(y_test, y_pred_oversampling)
disp_oversampling = ConfusionMatrixDisplay(confusion_matrix=cm_oversampling, display_labels=["No Stroke", "Stroke"])
disp_oversampling.plot(cmap="Blues", values_format='d')
# plt.title("Confusion Matrix - Logistic Regression (Oversampling)", fontsize=14)
# plt.grid(False)
# plt.tight_layout()
# plt.show()

# Get predicted probabilities for the positive class (stroke = 1)
y_prob = log_reg_oversampling.predict_proba(X_test)[:, 1]

# Compute ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()



