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
from scipy.stats import chi2_contingency
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
# stroke_orig = pd.read_csv('/Users/dishakacha/Downloads/healthcare-dataset-stroke-data.csv', na_values='N/A')
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
# plot color palette
deep_palette = sns.color_palette("deep")
#%%
# SMART Question 1: What is the impact of lifestyle choices, particularly BMI and 
# smoking status, on the probability of having a stroke? Are there significant differences 
# among different groups? 
# Evaluate the relationship between lifestyle factors (BMI, smoking status) and stroke occurrences, 
# considering variations across different groups like gender, age, or residential areas.

#%%[markdown]
'''We first visualize the distribution of BMI based on whether a person had a stroke or not.'''

# #%%
# bmi_stroke0 = (stroke['bmi'])[stroke.stroke == 0]
# plt.hist(bmi_stroke0, alpha=.5)
# plt.title('Distribution of Body Mass Index for Individuals Without a Stroke')
# plt.legend()
# plt.show()

# #%%
# bmi_stroke1 = (stroke['bmi'])[stroke.stroke == 1]
# plt.hist(bmi_stroke1, alpha=.5, color=deep_palette[1])
# plt.title('Distribution of Body Mass Index Who Have Had a Stroke')
# plt.legend()
# plt.show()

#%%
plt.figure(figsize=(10, 6))
sns.kdeplot(bmi_stroke0, label='No Stroke', fill=True, alpha=0.5)
sns.kdeplot(bmi_stroke1, label='Stroke', fill=True, alpha=0.5)
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Density')
plt.title('Density Plot of Body Mass Index Based on Stroke Status')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# sns.violinplot(x='stroke', y='bmi', data=stroke, inner='quartile', palette='muted')
# plt.xlabel('Stroke Status')
# plt.ylabel('Body Mass Index (BMI)')
# plt.title('Violin Plot of Body Mass Index Based on Stroke Status')
# plt.show()

#%%[markdown]
'''Since BMI is a numerical variable, while status of stroke is categorical, (and the population standard
deviation of BMI is unknown) we can conduct a t test (as opposed to a z test).'''

#%%
# Null hypothesis, alternative hypothesis, assumptions
print('average BMI for stroke0:', np.average(bmi_stroke0))
print('average BMI for stroke1:', np.average(bmi_stroke1))
t_stat, p_value = stats.ttest_ind(a=bmi_stroke0, b=bmi_stroke1, equal_var=True)
print('Our t test statistic is', t_stat, 'with p-value', p_value) # Still getting nan values; we have not addressed the N/A values

if p_value < .05:
    conclusion = '''At an alpha level of .05, we reject the null hypothesis and conclude that 
    the average BMI of those who had a stroke is significantly different from those individuals
    who did not have a stroke. We can say that BMI and stroke are associated with one another.
    '''
else:
    conclusion = '''At an alpha level of .05, we fail to reject the null hypothesis and conclude that 
    the average BMI of those who had a stroke is not significantly different from those individuals
    who did not have a stroke. We can say that BMI and stroke are not associated with one another.
    '''
#%%
print(conclusion)

#%%[markdown]
'''In a similar vein, because stroke and smoking status are both categorical variables, we can now conduct
a chi square test to determine whether the two variables are associated with each other.
# Null hypothesis alternative hypothesis, assumptions
First, we need to create a contingency table.'''

#%%
# EDA Plot for second part of SMART Q1
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(x='smoking_status', hue='stroke', data=stroke)
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.title('Bar Plot of Stroke Status by Smoking Status')
plt.legend(title='Stroke Status', loc='upper right')
plt.show()

#%%
stroke0 = stroke[stroke['stroke'] == 0]
stroke1 = stroke[stroke['stroke'] == 1]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
smokecounts_stroke0 = stroke0['smoking_status'].value_counts(normalize=True)
axes[0].pie(smokecounts_stroke0, labels=smokecounts_stroke0.index, autopct="%.1f%%", startangle=90)
axes[0].set_title('Distribution of Smoking Status (No Stroke)')
smokecounts_stroke1 = stroke1['smoking_status'].value_counts(normalize=True)
axes[1].pie(smokecounts_stroke1, labels=smokecounts_stroke1.index, autopct="%.1f%%", startangle=90)
axes[1].set_title('Distribution of Smoking Status (Stroke)')
plt.tight_layout()
plt.show()

#%%
# smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"

#%%
stroke0former = stroke[(stroke['stroke'] == 0) & (stroke['smoking_status'] == 'formerly smoked')]
stroke1former = stroke[(stroke['stroke'] == 1) & (stroke['smoking_status'] == 'formerly smoked')]
stroke0never = stroke[(stroke['stroke'] == 0) & (stroke['smoking_status'] == 'never smoked')]
stroke1never = stroke[(stroke['stroke'] == 1) & (stroke['smoking_status'] == 'never smoked')]
stroke0smokes = stroke[(stroke['stroke'] == 0) & (stroke['smoking_status'] == 'smokes')]
stroke1smokes = stroke[(stroke['stroke'] == 1) & (stroke['smoking_status'] == 'smokes')]
stroke0unknown = stroke[(stroke['stroke'] == 0) & (stroke['smoking_status'] == 'Unknown')]
stroke1unknown = stroke[(stroke['stroke'] == 1) & (stroke['smoking_status'] == 'Unknown')]
#%%
cont_tab = np.array([[len(stroke0former), len(stroke1former)], [len(stroke0never), len(stroke1never)], [len(stroke0smokes), len(stroke1smokes)], [len(stroke0unknown), len(stroke1unknown)]])
#%%
# from scipy.stats import chi2_contingency
chisquare_val, p_value, _, _ = chi2_contingency(cont_tab)
print('Our chi square test statistic is', chisquare_val, 'with p-value', p_value)

if p_value < .05:
    conclusion = '''At an alpha level of .05, we reject the null hypothesis and conclude that 
    status of smoking and whether an individual has a stroke or not are associated with one another.
    '''
else:
    conclusion = '''At an alpha level of .05, we fail to reject the null hypothesis and conclude that 
    status of smoking and whether an individual has a stroke or not are not associated with one another.'''
#%%
print(conclusion)

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%
# SMART Question 2: How do the levels of hypertension and heart disease individually, or in combination, impact the probability of stroke among different work types? 
# Investigate the relationship between hypertension, heart disease, and their interaction among various work types to discern if certain professions are more susceptible to strokes due to these health factors. 

#%%
# SMART Question 3: Is it possible to assess the influence of residence type, occupation, and smoking habits on stroke frequency? 
# Evaluate the potential influence of smoking patterns and occupation on stroke risk among urban and rural residents. 

stroke_1 = stroke[stroke['stroke'] == 1]

# Countplot for smoking status and occupation distribution for stroke=1
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='smoking_status', hue='work_type', data=stroke_1)
plt.title('Smoking Status and Occupation Distribution')
plt.legend(title='Work Type')

# Pie chart for the distribution of smoking status for stroke=1
plt.subplot(1, 2, 2)
smoking_status_counts = stroke_1['smoking_status'].value_counts()
plt.pie(smoking_status_counts, labels=smoking_status_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Smoking Status Distribution')

plt.show()

# Countplot for residence type and occupation distribution for stroke=1
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='Residence_type', hue='work_type', data=stroke_1)
plt.title('Residence Type and Occupation Distribution for Stroke=1')
plt.legend(title='Work Type')

# Pie chart for the distribution of residence type for stroke=1
plt.subplot(1, 2, 2)
residence_type_counts = stroke_1['Residence_type'].value_counts()
plt.pie(residence_type_counts, labels=residence_type_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Residence Type Distribution for Stroke=1')

plt.show()

# Adding another categorical variable using hue
plt.show()

plt.figure(figsize=(15, 7))

sns.countplot(x='work_type', hue='smoking_status', data=stroke_1, palette='viridis')
plt.title('Occupation and Smoking Status Distribution for Stroke=1')

# Adding another categorical variable using hue
plt.show()

#%%
# Create a contingency table
contingency_table = pd.crosstab(stroke['work_type'], stroke['stroke'])

# Perform chi-squared test
chi2, p, _, _ = chi2_contingency(contingency_table)

print("Chi-squared:", chi2)
print("P-value:", p)

print("""The chi-squared statistic being significantly different from zero suggests that there is a notable discrepancy between the observed distribution of stroke cases across different work types and the distribution expected under the assumption of independence.
The small p-value (close to zero) suggests strong evidence against the null hypothesis, indicating that there is a significant association between work type and the occurrence of stroke.""")

print("""The small p-value (close to zero) suggests strong evidence against the null hypothesis, indicating that there is a significant association between work type and the occurrence of stroke.""")

print("""The results provide statistical support for the hypothesis that work type and stroke are associated in the dataset. In practical terms, it implies that the distribution of stroke cases is not uniform across different work types, and there may be a relationship between the two variables.""")
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
X = stroke.loc[:, stroke.columns != 'stroke']
y = stroke['stroke']

X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=1) # 80-20 split

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%%
# Logistic regression - Devarsh

#%%
# SVC - Disha

#%%
# Random Forest - Abhradeep
