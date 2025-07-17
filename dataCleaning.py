import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

url = "https://raw.githubusercontent.com/CoderOphilia/DataFile/refs/heads/main/healthcare-dataset-stroke-data(in).csv"
df = pd.read_csv(url)

#Drop the ID column because it add no value to the analysis
# 'id' is just an unique identifier, not useful for prediction
# It does not provide any information about the patient or the stroke.
# It might confuse the model and lead to overfitting.
#df.drop(columns=['id'], inplace=True) # drop the 'id' column from the DataFrame

#Identify and Handle Missing values
missing_data = df.isnull() # True if missing value, False if not missing value

# for column in missing_data.columns.values.tolist():
#     print(column)
#     print(missing_data[column].value_counts())
#     print("")
#print(df.isnull().sum()) # to check the missing values in the columns
# Only 'bmi' column has missing values (201 missing values)

#Dealing with missing data
'''
1. The stroke column do not have missing values, so we do not need to worry about it. 
2. The 'bmi' column has 201 missing values. We can replace the missing values with the mean of the column.

To confirm whether we need to replace the missing values with the mean of the column, we can check the distribution of the 'bmi' column.

'''
#print(df.head())
#print(df.dtypes) # to check the data types of the columns

# id                     int64 
# gender                object
# age                  float64  (int would be better)
# hypertension           int64  (bool would be better)
# heart_disease          int64  (bool would be better)
# ever_married          object
# work_type             object
# Residence_type        object
# avg_glucose_level    float64
# bmi                  float64
# smoking_status        object
# stroke                 int64  (bool would be better)
# dtype: object

#print(df.isnull().sum()) # to check the missing values in the columns   # bmi has 201 missing values
#print(df.describe(include='all')) # to check the summary statistics of the data; provides count, mean, standard deviation, min, max, and quartile ranges for numerical columns.
#print(df.info()) # to check the information of the data;  gives an overview of the top and bottom 30 rows of the DataFrame, useful for quick visual inspection.
#print(df.shape) # to check the shape of the data; returns the number of rows and columns in the DataFrame.
#df1 = df.dropna(subset=["bmi"], axis=0) # drop rows with missing values in the column 'bmi' 
#print(df1.shape) # to check the shape of the data after dropping the missing values


#If we want nin-biased imputations, median is a better choice than mean.
# The mean is sensitive to outliers, while the median is not.
df['bmi'].fillna(df['bmi'].median(), inplace=True) #Median is more robust to outliers than mean, so we use median to fill the missing values in the 'bmi' column.
# print(df.isnull().sum())

# unknown_count = (df['smoking_status'] == 'Unknown').sum()
# print("Number of Unknown smoking status:", unknown_count) # Number of Unknown smoking status: 1544

smoke_status = df['smoking_status'].value_counts()
# print(smoke_status) # to check the smoking status counts
df['smoking_status'].fillna('Unknown', inplace=True) # Fill missing 'smoking_status' with 'Unknown' (categorical placeholder) , this will preserve the data instead of dropping it.

# One - hot encodding for categorical variables
# One-hot encoding is a technique used to convert categorical variables into a numerical format that can be used by machine learning algorithms.
# It creates binary columns for each category in the original column, allowing the model to learn from categorical data.
# This is important because most machine learning algorithms work with numerical data.
# In this case, we will use One-hot encoding for the categorical variables in the dataset.
# We will use the OneHotEncoder class from the sklearn.preprocessing module to perform one-hot encoding.
# We will also use the ColumnTransformer class from the sklearn.compose module to apply the OneHotEncoder to the categorical variables in the dataset.
# We will use the Pipeline class from the sklearn.pipeline module to create a pipeline for preprocessing the data.  
# We will use the StandardScaler class from the sklearn.preprocessing module to scale the numerical variables in the dataset.
# We will use the SimpleImputer class from the sklearn.impute module to impute the missing values in the dataset.
# We will use the ColumnTransformer class from the sklearn.compose module to apply the SimpleImputer to the numerical variables in the dataset.
# We will use the Pipeline class from the sklearn.pipeline module to create a pipeline for preprocessing the data.
# We will use the StandardScaler class from the sklearn.preprocessing module to scale the numerical variables in the dataset.
# 1st step of encoding: --> define categorical and numerical columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
# Check original DataFrame (before encoding)
#print(df[categorical_cols].head())

# 2nd step: --> create a pipeline for preprocessing the data
# Build a preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

# 3. Separate features and target
X = df.drop(columns='stroke') #Keeps the 'id' column in the DataFrame
X_model = X.drop(columns=['id']) #Drop the 'id' column from the DataFrame for modeling
y = df['stroke']

#BOX PLOTS
# Create box plots for numerical columns
sns.boxplot(data=df, x='stroke', y='age')
plt.title('Box plot of Age vs Stroke')      
plt.xlabel('Stroke')
plt.ylabel('Age')   
plt.show()

# Apply it to your modeling data
X_transformed = preprocessor.fit_transform(X_model)


