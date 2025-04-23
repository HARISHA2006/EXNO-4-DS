# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, RobustScaler
```
```
df=pd.read_csv("/content/bmi.csv")
print(df.head())
```
![image](https://github.com/user-attachments/assets/4f907939-9a40-4457-b695-5cd102c9dfce)
```
df = df.dropna()
print("Max Height:", df["Height"].max())
print("Max Weight:", df["Weight"].max())
```
![image](https://github.com/user-attachments/assets/09ac6563-ec56-4abe-9693-2c20bc19509e)
```
# Perform MinMax Scaler
minmax = MinMaxScaler()
df_minmax = minmax.fit_transform(df[["Height", "Weight"]])
print("\nMinMaxScaler Result:\n", df_minmax[:5])
```
![image](https://github.com/user-attachments/assets/7cd05a12-b87b-4d5c-9c13-949760cb5fc9)
```
# Perform Standard Scaler
standard = StandardScaler()
df_standard = standard.fit_transform(df[["Height", "Weight"]])
print("\nStandardScaler Result:\n", df_standard[:5])
```
![image](https://github.com/user-attachments/assets/aad97786-ab25-446a-959d-9a81288bc6a5)
```
# Perform Normalizer
normalizer = Normalizer()
df_normalized = normalizer.fit_transform(df[["Height", "Weight"]])
print("\nNormalizer Result:\n", df_normalized[:5])
```
![image](https://github.com/user-attachments/assets/b29366a9-5c03-4413-a374-30dd75441ef8)
```
#Perform MaxAbsScaler
max_abs = MaxAbsScaler()
df_maxabs = max_abs.fit_transform(df[["Height", "Weight"]])
print("\nMaxAbsScaler Result:\n", df_maxabs[:5])
```
![image](https://github.com/user-attachments/assets/b90258c2-66a2-4892-b6d3-3ec6ffb995a3)
```
#  Perform RobustScaler
robust = RobustScaler()
df_robust = robust.fit_transform(df[["Height", "Weight"]])
print("\nRobustScaler Result:\n",df_robust[:5])
```
![image](https://github.com/user-attachments/assets/2cdd9e18-5cd8-4d8b-ae01-5e70c5b12438)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2
```
```
# Load Titanic dataset
df = pd.read_csv('/content/titanic_dataset.csv')
# Display column names
print(df.columns)
```
![image](https://github.com/user-attachments/assets/8c63eccc-c4f3-44a8-b4b7-83f81f3e3c6a)
```
print("Shape:", df.shape)
```
![image](https://github.com/user-attachments/assets/b56054ad-7424-4aae-8016-6ea5b93390f4)
```
# Drop irrelevant columns
df1 = df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
# Check null values in Age
print("Missing Age values before:", df1['Age'].isnull().sum())
```
![image](https://github.com/user-attachments/assets/ecf71977-1e7a-4c02-bc44-b5e35505a62c)
```
# Fill null values in Age using forward fill
df1['Age'] = df1['Age'].fillna(method='ffill')
```
![image](https://github.com/user-attachments/assets/05f4cbc9-31c0-4edc-a649-1c855711108e)
```
# Check again
print("Missing Age values after:", df1['Age'].isnull().sum())
```
![image](https://github.com/user-attachments/assets/fb78fb47-2f91-4806-aa51-9080796bab5b)
```
# Apply SelectKBest for top 3 features
feature = SelectKBest(mutual_info_classif, k=3)
# Reorder columns as required
df1 = df1[['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived']]
# Define feature matrix and target vector
X = df1.iloc[:, 0:6]
y = df1.iloc[:, 6]
# Confirm columns
print("X Columns:", X.columns)
y = y.to_frame()
print("y Columns:", y.columns)
```
![image](https://github.com/user-attachments/assets/9cfadaf6-5faf-484a-adc3-7588adf8a3db)
```
# Fit SelectKBest
feature.fit(X, y.values.ravel())
```
![image](https://github.com/user-attachments/assets/7f0253b8-3cb9-45c8-a16a-f5b98e23de34)
```
# Get selected feature scores
scores = pd.DataFrame({"Feature": X.columns, "Score": feature.scores_})
print("\nFeature Scores:\n", scores.sort_values(by="Score", ascending=False))
```
![image](https://github.com/user-attachments/assets/613b34f2-9c3d-4047-a007-73a17e8c3196)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
