#%%[markdown]
'''Data Mining Final Project
Team 6
Collaborators: Abhradeep Das, Disha Kacha, Neelima Puthanveetil, Devarsh Sheth '''

#%%
# Import necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime

################### Sklearn ####################################
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#%%
# Read in the dataset: 
stroke_orig = pd.read_csv('healthcare-dataset-stroke-data.csv', na_values='N/A')
stroke = stroke_orig

#%%[markdown]
'''Data Dictionary:''' # (From the data source on kaggle)
'''
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient
'''
#%%[markdown]
'''Let's first try to understand our dataset by using basic visualization techniques.'''

#%%
stroke.info()
stroke.head()
stroke.tail()

#%%[makdown]
'''We first want to check whether there any N/A values in our data,'''

#%%
# Checking for N/A Values
# if True in stroke.isna(): 
#     print('There are N/A values in the dataset')
# else:
#     print('No N/A values in our dataset!')
stroke.dropna(inplace = True)
print(stroke.isna().sum())

#%%
# Checking for duplicate records
if True in stroke.duplicated():
    print('There are duplicate rows')
else:
    print('There are no duplicate rows')

#%%
# Checking for outliers
outliercols = ['age', 'avg_glucose_level', 'bmi']

for col in outliercols:
    Q1 = stroke[col].quantile(0.25)
    Q3 = stroke[col].quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5
    outliers = stroke[(stroke[col] < Q1 - threshold * IQR) | (stroke[col] > Q3 + threshold * IQR)]
    if len(outliers) > 0:
        print('There were outliers in', col, 'column')
        print('We will drop the outliers')
        stroke = stroke.drop(outliers.index)
        stroke.info()
    else:
        print('No outliers in', col, 'column')
stroke.info()

#%%
# Just checking to see how many data points we have for those had and did not have strokes:
total = len(stroke['stroke'])
num_stroke = stroke['stroke'].sum()

print('Of our', total, 'data points', num_stroke, 'had a stroke')

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%
# Visualization
# We can replace some of these plots with density plots, boxplots, and violin plots (or anything else!)

# Gender:
gender_counts = stroke['gender'].value_counts(normalize=True)
gender_counts.plot.pie(y='Gender', autopct="%.1f%%")
plt.title('Distribution of Gender')
plt.show()

# plot for stroke with hue based on gender

# Age:
plt.hist(stroke['age'], alpha=.5, label='Age')
plt.title('Distribution of Age')
plt.show()

# stacked histogram for age based on stroke? or boxplot

# Hypertension:
plt.bar([0,1], list(stroke['hypertension'].value_counts())) # Have to check why the x-axis is not being changed to 0 and 1
plt.xlabel("Have Hypertension") 
plt.ylabel("Number of Individuals") 
plt.title("Counts of Hypertension Cases") 
plt.show() 

# Heart disease:
plt.bar([0,1], list(stroke['heart_disease'].value_counts())) # Have to check why the x-axis is not being changed to 0 and 1
plt.xlabel("Have Heart Disease") 
plt.ylabel("Number of Individuals") 
plt.title("Counts of Heart Disease Cases") 
plt.show() 

#%%[markdown]
'''We see that the bar charts for heart disease and hypertension look very similar; these variables might be highly correlated; we want to be careful of multicollinearity when building models'''
# We can add a plot of hypertension = 1, heart disease = 1

#%%
# Ever Married:
plt.bar([0,1], list(stroke['ever_married'].value_counts())) # Have to check why the x-axis is not being changed to 0 and 1
plt.xlabel("Have been married before") 
plt.ylabel("Number of Individuals") 
plt.title("Counts of Individuals Who Have Been Married Before") 
plt.show()

# Work type
work_counts = stroke['work_type'].value_counts(normalize=True)
work_counts.plot.pie(y='Work Type', autopct="%.1f%%")
plt.title('Distribution of Work Type')
plt.show()

# Residence type
plt.bar([0,1], list(stroke['Residence_type'].value_counts())) # Have to check why the x-axis is not being changed to 0 and 1
plt.xlabel("Type of Residence") 
plt.ylabel("Number of Individuals") 
plt.title("Counts of Individuals' Residence Type") 
plt.show()

# Glucose level
plt.hist(stroke['avg_glucose_level'], alpha=.5, label='Avg Glucose Level')
plt.title('Distribution of Average Glucose Level in Blood')
plt.show()
# stacked histogram or boxplot for avg glucose level

# BMI
plt.hist(stroke['bmi'], alpha=.5, label='BMI')
plt.title('Distribution of Body Mass Index')
plt.show()

# Smoking status
plt.bar([0, 1, 2, 3], list(stroke['smoking_status'].value_counts())) # Have to check why the x-axis is not being changed to 0 and 1
plt.xlabel("Smoking Status") 
plt.ylabel("Number of Individuals") 
plt.title("Counts of Individuals' Smoking Status") 
plt.show()
# I think we might need another plot here to visually determine its association with stroke

#%%[markdown]
'''[comments about visualization]'''

#%%[markdown]
'''Now that we have cleaned our dataset and visualized the key components, we can begin with exploratory data analysis.'''

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%[markdown]
'''EDA Section'''
# EDA

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%[markdown]
'''Hypothesis Testing Section'''
# Hypothesis testing

#%%
# SMART Question 1: What is the impact of lifestyle choices, particularly BMI and 
# smoking status, on the probability of having a stroke? Are there significant differences 
# among different groups? 
# Evaluate the relationship between lifestyle factors (BMI, smoking status) and stroke occurrences, 
# considering variations across different groups like gender, age, or residential areas.

'''Since BMI is a numerical variable, while status of stroke is categorical, (and the population standard
deviation of BMI is unknown) we can conduct a t test (as opposed to a z test).'''

# Null hypothesis, alternative hypothesis, assumptions

bmi_stroke0 = (stroke['bmi'])[stroke.stroke == 0]
bmi_stroke1 = (stroke['bmi'])[stroke.stroke == 1]

t_stat, p_value = stats.ttest_ind(a=bmi_stroke0, b=bmi_stroke1, equal_var=True)
print('Our t test statistic is', t_stat, 'with p-value', p_value) # Still getting nan values; we have not addressed the N/A values

'''In a similar vein, because stroke and smoking status are both categorical variables, we can conduct
a chi square test to determine whether the two variables are associated with each other.

First, we need to create a contingency table.'''

#%%
# SMART Question 2: How do the levels of hypertension and heart disease individually, or in combination, impact the probability of stroke among different work types? 
# Investigate the relationship between hypertension, heart disease, and their interaction among various work types to discern if certain professions are more susceptible to strokes due to these health factors. 

#%%
# SMART Question 3: Is it possible to assess the influence of residence type, occupation, and smoking habits on stroke frequency? 
# Evaluate the potential influence of smoking patterns and occupation on stroke risk among urban and rural residents. 

#%%
# SMART Question 4: Is there a significant correlation between marital status, gender, and age in this analysis? 
# Analyze marital status data to identify patterns within specific age groups among the married, and assess gender-based stroke frequency differences. 

le = LabelEncoder()
stroke['ever_married_encoded'] = le.fit_transform(stroke['ever_married'])
stroke['gender_encoded'] = le.fit_transform(stroke['gender'])

plt.figure(figsize=(10, 8))
correlation_matrix = stroke[['age', 'stroke', 'ever_married_encoded', 'gender_encoded']].corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix: Age and Stroke')
plt.show()

print('''Age and Stroke (0.245257):

There is a positive correlation of approximately 0.25 between age and stroke.
This suggests that, on average, as age increases, the likelihood of having a stroke also tends to increase. However, the correlation is not extremely strong.

       
Ever Married and Stroke (0.108340):

There is a positive correlation of approximately 0.11 between ever_married_encoded and stroke.
This suggests a weak positive relationship between being ever married and the likelihood of having a stroke. 

Gender and Stroke (0.008929):

There is a very weak positive correlation of approximately 0.009 between gender_encoded and stroke.
This suggests a minimal relationship between gender and the likelihood of having a stroke.           
''')
# %%

age_bins = [0, 18, 35, 50, 65, 100]
age_labels = ['0-18', '19-35', '36-50', '51-65', '66-100']

# Add a new column 'age_group' to the DataFrame
stroke['age_group'] = pd.cut(stroke['age'], bins=age_bins, labels=age_labels, right=False)

# Filter the DataFrame for married individuals
married_df = stroke[stroke['ever_married'] == 'Yes']

# Create a cross-tabulation (crosstab) between age groups, marital status, and stroke
married_age_stroke = pd.crosstab([married_df['age_group'], married_df['stroke']], married_df['ever_married'])
print(married_age_stroke)
# Plot the bar chart
married_age_stroke.unstack().plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(10, 6))
plt.title('Age Distribution within Married Group with Stroke')
plt.xlabel('Age Group and Stroke')
plt.ylabel('Count')
plt.show()


print('''Age Group 19-35:

Individuals in this age group who are ever married have a low incidence of strokes (1 case).
The majority (374 cases) of individuals in this age group who are ever married do not have strokes.


Age Group 36-50:

There is a slightly higher incidence of strokes in this age group (12 cases) compared to the 19-35 age group.
The majority (803 cases) of individuals in this age group who are ever married do not have strokes.

      
Age Group 51-65:

The incidence of strokes increases in this age group (43 cases), indicating a higher risk compared to the younger age groups.
The majority (824 cases) of individuals in this age group who are ever married do not have strokes.

      
Age Group 66-100:

This age group has the highest incidence of strokes (85 cases) among individuals who are ever married.
The majority (593 cases) of individuals in this age group who are ever married do not have strokes.
''')
#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%[markdown]
'''Unique Values and Categorical changes and Normalization'''
#%%
columns_temp = ['gender', 'ever_married', 'work_type', 'smoking_status', 'Residence_type']

for col in columns_temp :
    print('column :', col)
    for index, unique in enumerate(stroke[col].unique()) :
        print(unique, ':', index)
    print('_'*45)

# gender
stroke_numerical = stroke.replace(
    {'gender' : {'Male' : 0, 'Female' : 1, 'Other' : 2}}
)

# ever_married
stroke_numerical =  stroke_numerical.replace(
    {'ever_married' : {'Yes' : 0, 'No' : 1}}
)

# work_type
stroke_numerical =  stroke_numerical.replace(
    {'work_type' : {'Private' : 0, 'Self-employed' : 1, 'Govt_job' : 2, 'children' : 3, 'Never_worked' : 4}}
)

# smoking_status
stroke_numerical =  stroke_numerical.replace(
    {'smoking_status' : {'formerly smoked' : 0, 'never smoked' : 1, 'smokes' : 2, 'Unknown' : 3}}
)

# Residence_type
stroke_numerical =  stroke_numerical.replace(
    {'Residence_type' : {'Urban' : 0, 'Rural' : 1}}
)

stroke_numerical.head()

#%%
numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

X_temp = stroke_numerical[numerical_columns]
y = stroke_numerical['stroke']

#%%
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_temp), columns=numerical_columns)

# Concatenate the scaled numerical features with the encoded categorical features
X = pd.concat([X_scaled, stroke_numerical.drop(columns=numerical_columns + ['stroke'])], axis=1)

print(X.describe())
#%%[markdown]
'''Modeling Section'''

#%%
# Decision tree classifier - Nema

#%%
# Logistic regression - Devarsh

#%%
# SVC - Disha

#%%
# Random Forest - Abhradeep