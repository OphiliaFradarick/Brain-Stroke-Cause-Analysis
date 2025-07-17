from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv("cleaned_health_data(in).csv")

# ----- Binary Encoding of categorical variables -----
binary_categorical_cols = ['gender', 'ever_married', 'Residence_type']

for col in binary_categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ----- Multi-categorical Encoding of categorical variables -----
multi_categorical_cols = ['work_type', 'smoking_status']
df = pd.get_dummies(df, columns=multi_categorical_cols, drop_first=True)

# ----- Standardization of numerical variables -----
numerical_cols = ['age', 'bmi', 'avg_glucose_level_transformed']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# ----- Splitting the dataset into features and target variable -----
y = df['stroke']
X = df.drop(columns='stroke')   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----- Logistic Regression Model -----
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)

# Predict on test data
y_pred = log_reg.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# print(f"Overall Accuracy of the Logistic Regression model: {accuracy:.2%}")

# Precision for No Stroke (class 0) and Stroke (class 1)
precision_no_stroke = precision_score(y_test, y_pred, pos_label=0)
precision_stroke = precision_score(y_test, y_pred, pos_label=1)

# Print precision for both classes
# print(f"\nPrecision (No Stroke): {precision_no_stroke:.2%}")
# print(f"Precision (Stroke): {precision_stroke:.2%}")

# Calculate accuracy for No Stroke (class 0) and Stroke (class 1) separately
accuracy_no_stroke = (y_pred == 0) & (y_test == 0)  # True Positives for No Stroke
accuracy_no_stroke = accuracy_no_stroke.sum() / (y_test == 0).sum()  # Proportion of correctly predicted No Stroke

accuracy_stroke = (y_pred == 1) & (y_test == 1)  # True Positives for Stroke
accuracy_stroke = accuracy_stroke.sum() / (y_test == 1).sum()  # Proportion of correctly predicted Stroke

# print(f"\nAccuracy for No Stroke: {accuracy_no_stroke:.2%}")
# print(f"Accuracy for Stroke: {accuracy_stroke:.2%}")

#------Checking for imbalance in the dataset------
# Check the distribution of the target variable
# print("\nDistribution of the target variable:")
# print(df['stroke'].value_counts(normalize=True))  
# print("\nPercentage of Stroke cases in the dataset:")
# print(df['stroke'].value_counts(normalize=True)[1] * 100) 
# print("\nPercentage of No Stroke cases in the dataset:")
# print(df['stroke'].value_counts(normalize=True)[0] * 100)


# Over Sampling using SMOTE
# Define SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to balance the dataset
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the Logistic Regression model on the resampled data
log_reg_oversampling = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg_oversampling.fit(X_train1, y_train1)

# Predict and Evaluate
# Predict and Evaluate for the oversampled data
y_pred_log_oversampling = log_reg_oversampling.predict(X_test1)

# Calculate accuracy for the oversampled model
accuracy_oversampling = accuracy_score(y_test1, y_pred_log_oversampling)

# Precision for No Stroke (class 0) and Stroke (class 1) in the oversampled data
precision_no_stroke_oversampling = precision_score(y_test1, y_pred_log_oversampling, pos_label=0)
precision_stroke_oversampling = precision_score(y_test1, y_pred_log_oversampling, pos_label=1)

# Accuracy for No Stroke (class 0) and Stroke (class 1) in the oversampled data
accuracy_no_stroke_oversampling = (y_pred_log_oversampling == 0) & (y_test1 == 0)  # True Positives for No Stroke
accuracy_no_stroke_oversampling = accuracy_no_stroke_oversampling.sum() / (y_test1 == 0).sum()  # Proportion of correctly predicted No Stroke

accuracy_stroke_oversampling = (y_pred_log_oversampling == 1) & (y_test1 == 1)  # True Positives for Stroke
accuracy_stroke_oversampling = accuracy_stroke_oversampling.sum() / (y_test1 == 1).sum()  # Proportion of correctly predicted Stroke

# Print the results after SMOTE
# print(f"\nAccuracy of the Logistic Regression model after SMOTE: {accuracy_oversampling:.2%}")
# print(f"Precision (No Stroke) after SMOTE: {precision_no_stroke_oversampling:.2%}")
# print(f"Precision (Stroke) after SMOTE: {precision_stroke_oversampling:.2%}")
# print(f"\nAccuracy for No Stroke after SMOTE: {accuracy_no_stroke_oversampling:.2%}")
# print(f"Accuracy for Stroke after SMOTE: {accuracy_stroke_oversampling:.2%}")

#-----UNDERSAMPLING-----
# Define undersampling
undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Apply undersampling to balance the dataset
X_resampled, y_resampled = undersample.fit_resample(X, y)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Logistic Regression Model after Undersampling
log_reg_undersampled = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg_undersampled.fit(X_train2, y_train2)

# Predict and Evaluate on the undersampled dataset
y_pred_log_undersampled = log_reg_undersampled.predict(X_test2)

# Calculate accuracy for the undersampled model
accuracy_undersampling = accuracy_score(y_test2, y_pred_log_undersampled)

# Precision for No Stroke (class 0) and Stroke (class 1) in the undersampled data
precision_no_stroke_undersampling = precision_score(y_test2, y_pred_log_undersampled, pos_label=0)
precision_stroke_undersampling = precision_score(y_test2, y_pred_log_undersampled, pos_label=1)

# Accuracy for No Stroke (class 0) and Stroke (class 1) in the undersampled data
accuracy_no_stroke_undersampling = (y_pred_log_undersampled == 0) & (y_test2 == 0)
accuracy_no_stroke_undersampling = accuracy_no_stroke_undersampling.sum() / (y_test2 == 0).sum()

accuracy_stroke_undersampling = (y_pred_log_undersampled == 1) & (y_test2 == 1)
accuracy_stroke_undersampling = accuracy_stroke_undersampling.sum() / (y_test2 == 1).sum()

# Print the results after Undersampling
# print(f"\nAccuracy of the Logistic Regression model after Undersampling: {accuracy_undersampling:.2%}")
# print(f"Precision (No Stroke) after Undersampling: {precision_no_stroke_undersampling:.2%}")
# print(f"Precision (Stroke) after Undersampling: {precision_stroke_undersampling:.2%}")
# print(f"\nAccuracy for No Stroke after Undersampling: {accuracy_no_stroke_undersampling:.2%}")
# print(f"Accuracy for Stroke after Undersampling: {accuracy_stroke_undersampling:.2%}")

#-----------Confusion Matrix for Oversampling-----------
cm_oversampling = confusion_matrix(y_test1, y_pred_log_oversampling)
disp_oversampling = ConfusionMatrixDisplay(confusion_matrix=cm_oversampling, display_labels=["No Stroke", "Stroke"])  
# disp_oversampling.plot(cmap="Blues", values_format='d')
# plt.title("Confusion Matrix - Logistic Regression (Oversampling)", fontsize=14)
# plt.grid(False)
# plt.tight_layout()
# plt.show()

#---------- Confusion Matrix for Undersampling-----------
cm_undersampling = confusion_matrix(y_test2, y_pred_log_undersampled)
disp_undersampling = ConfusionMatrixDisplay(confusion_matrix=cm_undersampling, display_labels=["No Stroke", "Stroke"])
# disp_undersampling.plot(cmap="Blues", values_format='d')
# plt.title("Confusion Matrix - Logistic Regression (Undersampling)", fontsize=14)
# plt.grid(False)
# plt.tight_layout()
# plt.show()

#----------------- Get feature importance (coefficients) from Logistic Regression ---------------------
feature_importance_log_reg = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(log_reg.coef_[0])})
feature_importance_log_reg = feature_importance_log_reg.sort_values(by='Importance', ascending=False)

# Plot feature importance for Logistic Regression
# plt.figure(figsize=(10,6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_log_reg)
# plt.title("Feature Importance (Logistic Regression)")
# # Adjust the layout to center the plot
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# plt.savefig("feature_importance_logistic_regression.png", bbox_inches='tight', dpi=300)
# plt.show()

