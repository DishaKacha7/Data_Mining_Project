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
import matplotlib.pyplot as plt

################### Sklearn ####################################
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import plot_tree

# pip install imbalanced-learn
import imblearn
from imblearn.over_sampling import SMOTE

#%%
# plot color palette
deep_palette = sns.color_palette("deep")

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
'''We first want to check whether there any N/A, duplicate records, or outlier values in our data.'''

#%%
# Dropping N/A Values
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
# How many data points we have for those who did and did not have strokes:
total = len(stroke['stroke'])
num_stroke = stroke['stroke'].sum()

print('Of our', total, 'data points', num_stroke, 'had a stroke')

#%%[markdown]
'''Notice the severe imbalance in our target variable. We will address this with appropriate
sampling techniques before proceeding with our modeling section.'''

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%
# Visualization

# Gender:
gender_counts = stroke['gender'].value_counts(normalize=True)
gender_counts.plot.pie(y='Gender', autopct="%.1f%%")
plt.title('Distribution of Gender')
plt.show()

# Age:
plt.hist(stroke['age'], alpha=.5, label='Age')
plt.title('Distribution of Age')
plt.show()

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
'''We see that the bar charts for heart disease and hypertension look very similar; these variables might be highly correlated; we want to be careful of multicollinearity when building models.'''

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

#%%[markdown]
'''Now that we have cleaned our dataset and visualized the key components, we can begin with exploratory data analysis.'''

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%[markdown]
'''EDA and Hypothesis Testing Section'''

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%
# SMART Question 1: What is the impact of lifestyle choices, particularly BMI and 
# smoking status, on the probability of having a stroke?

#%%[markdown]
'''We first visualize the distribution of BMI based on whether a person had a stroke or not.'''

# #%%
bmi_stroke0 = (stroke['bmi'])[stroke.stroke == 0]
bmi_stroke1 = (stroke['bmi'])[stroke.stroke == 1]

plt.figure(figsize=(10, 6))
sns.kdeplot(bmi_stroke0, label='No Stroke', fill=True, alpha=0.5)
sns.kdeplot(bmi_stroke1, label='Stroke', fill=True, alpha=0.5)
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Density')
plt.title('Density Plot of Body Mass Index Based on Stroke Status')
plt.legend()
plt.show()

#%%[markdown]
'''Since BMI is a numerical variable, while status of stroke is categorical, (and the population standard
deviation of BMI is unknown) we can conduct a t test (as opposed to a z test).'''

#%%
print('average BMI for stroke0:', np.average(bmi_stroke0))
print('average BMI for stroke1:', np.average(bmi_stroke1))
t_stat, p_value = stats.ttest_ind(a=bmi_stroke0, b=bmi_stroke1, equal_var=True)
print('Our t test statistic is', t_stat, 'with p-value', p_value)

#%%[markdown]
'''We now want to understand how smoking status and having a stroke may be associated. We first
visualize the data.'''

#%%
plt.figure(figsize=(10, 6))
sns.countplot(x='smoking_status', hue='stroke', data=stroke)
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.title('Bar Plot of Stroke Status by Smoking Status')
plt.legend(title='Stroke Status', loc='upper right')
plt.show()

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

#%%[markdown]
'''In a similar vein as for BMI, because stroke and smoking status are both categorical variables, we can now conduct
a chi square test to determine whether the two variables are associated with each other. First, we need to create a contingency table.'''

#%%
stroke0former = stroke[(stroke['stroke'] == 0) & (stroke['smoking_status'] == 'formerly smoked')]
stroke1former = stroke[(stroke['stroke'] == 1) & (stroke['smoking_status'] == 'formerly smoked')]
stroke0never = stroke[(stroke['stroke'] == 0) & (stroke['smoking_status'] == 'never smoked')]
stroke1never = stroke[(stroke['stroke'] == 1) & (stroke['smoking_status'] == 'never smoked')]
stroke0smokes = stroke[(stroke['stroke'] == 0) & (stroke['smoking_status'] == 'smokes')]
stroke1smokes = stroke[(stroke['stroke'] == 1) & (stroke['smoking_status'] == 'smokes')]
stroke0unknown = stroke[(stroke['stroke'] == 0) & (stroke['smoking_status'] == 'Unknown')]
stroke1unknown = stroke[(stroke['stroke'] == 1) & (stroke['smoking_status'] == 'Unknown')]

cont_tab = np.array([[len(stroke0former), len(stroke1former)], [len(stroke0never), len(stroke1never)], [len(stroke0smokes), len(stroke1smokes)], [len(stroke0unknown), len(stroke1unknown)]])

chisquare_val, p_value, _, _ = chi2_contingency(cont_tab)
print('Our chi square test statistic is', chisquare_val, 'with p-value', p_value)

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%
# SMART Question 2: How do the levels of hypertension and heart disease individually, or in combination, impact the probability of stroke among different work types? 
# Investigate the relationship between hypertension, heart disease, and their interaction among various work types to discern if certain professions are more susceptible to strokes due to these health factors. 

# We are going to check the relation between work type and hypertension and heart disease.
plt.figure(figsize=(12, 8))
sns.countplot(x="work_type", hue="hypertension", data=stroke, palette="Blues", edgecolor="k")
plt.title("Distribution of Hypertension by Work Type")
plt.xlabel("Work Type")
plt.ylabel("Count")
plt.legend(title="Hypertension", labels=["No Hypertension", "Hypertension"])
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x="work_type", hue="heart_disease", data=stroke, palette="Greens", edgecolor="k")
plt.title("Distribution of Heart Disease by Work Type")
plt.xlabel("Work Type")
plt.ylabel("Count")
plt.legend(title="Heart Disease", labels=["No Heart Disease", "Heart Disease"])
plt.show()

# We are creating a new variable representing the combination of hypertension and heart disease

stroke['hypertension_heart_disease_interaction'] = 0  # 0 for no condition

# Set values for specific conditions
stroke.loc[(stroke['hypertension'] == 1) & (stroke['heart_disease'] == 0), 'hypertension_heart_disease_interaction'] = 1  # 1 for hypertension only
stroke.loc[(stroke['hypertension'] == 0) & (stroke['heart_disease'] == 1), 'hypertension_heart_disease_interaction'] = 2  # 2 for heart disease only
stroke.loc[(stroke['hypertension'] == 1) & (stroke['heart_disease'] == 1), 'hypertension_heart_disease_interaction'] = 3  # 3 for both conditions

print(stroke[['hypertension', 'heart_disease', 'hypertension_heart_disease_interaction']])

#%%
grouped_by_work_type = stroke.groupby('work_type')
grouped_by_work_type
descriptive_stats = grouped_by_work_type[['hypertension_heart_disease_interaction', 'stroke']].mean()
print(descriptive_stats)

custom_palette = sns.color_palette("Set2")

sns.set_style("darkgrid")

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='work_type',
    y='stroke',
    hue='hypertension_heart_disease_interaction',
    data=stroke,
    palette=custom_palette
)

plt.title("Stroke Probability by Work Type and Hypertension-Heart Disease Interaction")
plt.xlabel("Work Type")
plt.ylabel("Probability of Stroke")
plt.legend(title="Hypertension-Heart Disease Interaction")
plt.tight_layout()
plt.show()

# Performing the Chi-Square test between the new variable and stroke.
from scipy.stats import chi2_contingency

work_types = stroke['work_type'].unique()

for work_type in work_types:
    contingency_table = pd.crosstab(
        index=stroke[(stroke['work_type'] == work_type)]['hypertension_heart_disease_interaction'],
        columns=stroke[(stroke['work_type'] == work_type)]['stroke'],
        margins=False
    )

    print(f"\nWork Type: {work_type}")
    print("Contingency Table:")
    print(contingency_table)

    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"\nChi-Squared Value: {chi2}")
    print(f"P-Value: {p}")

    # Visualization
    custom_palette = sns.color_palette("Set2")
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
    x='hypertension_heart_disease_interaction',
    y='stroke',
    data=stroke[stroke['work_type'] == work_type],
    palette=custom_palette
)
    plt.title(f'Stroke Probability by Hypertension-Heart Disease Interaction in {work_type} Work Type')
    plt.xlabel('Hypertension-Heart Disease Interaction')
    plt.ylabel('Probability of Stroke')
    plt.tight_layout()
    plt.show()

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

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
print(contingency_table)
# Perform chi-squared test
chi2, p, _, _ = chi2_contingency(contingency_table)

print("Chi-squared:", chi2)
print("P-value:", p)

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

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

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%[markdown]
'''Categorical changes, Standardization, and Normalization'''
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
# numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
# numerical_columns = ['age', 'heart_disease', 'avg_glucose_level', 'bmi']
numerical_columns = ['age', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi', 'smoking_status']

X_temp = stroke_numerical[numerical_columns]
y = stroke_numerical['stroke']

#%%
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_temp), columns=numerical_columns)

# # Concatenate the scaled numerical features with the encoded categorical features
# X = pd.concat([X_scaled, stroke_numerical.drop(columns=numerical_columns + ['stroke'])], axis=1)
X = X_scaled

print(X.describe())

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%[markdown]
'''Modeling Section'''

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%[markdown]
'''What is important to note is the severe imbalance in our dataset's target variable. Before
we perform any machine learning algorithms, we want to use SMOTE analysis to balance our dataset.
This places more weight on the minority data (here, when patients had a stroke), to ensure that
the algorithm has enough data from both subsets to properly learn how to predict for the two outcomes.'''

#%%
# SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train_orig, X_test_orig, y_train_orig, y_test_orig = X_train, X_test, y_train, y_test

smt = SMOTE()
X_train, y_train = smt.fit_resample(X_train, y_train)

#%%[markdown]
'''We also realize the grave importance of our analysis. The model that we build can help predict
whether an individual has a stroke or not. While normally we can choose a model evaluation metric
based on personal preference or one guided by the nature of our dataset, we override this with
the dataset's context. In the medical field, particularly when trying to determine the probability of
having a stroke, we need to place particular importance on correctly predicting that a person will have
a stroke when they, in fact, will. That is, we want our analysis to be honed in on the true positive rate.
As such, we use the recall rate of each model, while also keeping in mind each model's F1 score (a combination
of recall and precision).'''

#%%
# Decision tree classifier
clf = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 50, 100],
    'min_samples_leaf': [1, 2, 4, 10, 25]
}
grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='recall')
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Plotting the tree
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(best_clf, 
                   feature_names=stroke.columns[:-1],  # Exclude the target column
                   class_names=['No Stroke', 'Stroke'],  # Assuming binary classification
                   filled=True)
plt.show()

print("Best Parameters:", grid_search.best_params_)

t1 = datetime.now()
dt = DecisionTreeClassifier(**grid_search.best_params_).fit(X_train, y_train)
t2 = datetime.now()
y_pred_dt = dt.predict(X_test)
cr = metrics.classification_report(y_test, y_pred_dt)
print(cr)
delta = t2-t1
delta_dt = round(delta.total_seconds(), 3)
print('DecisionTree takes : ', delta_dt, 'Seconds')

cm = confusion_matrix(y_test, y_pred_dt)
print(cm)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(dt, X_test, y_test, cmap=plt.cm.Blues, display_labels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix')
plt.show()

feature_importances = dt.feature_importances_
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Decision Tree - Feature Importance")
plt.show()

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%
# Logistic regression
lr = LogisticRegression()
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'class_weight' : ['balanced'],
    'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

lr_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10, scoring = 'recall').fit(X_train, y_train)

print('Tuned hyper parameters : ', lr_cv.best_params_)

t1 = datetime.now()
lr = LogisticRegression(**lr_cv.best_params_).fit(X_train, y_train)
t2 = datetime.now()
y_pred_lr = lr.predict(X_test)
cr = metrics.classification_report(y_test, y_pred_lr)
print(cr)
delta = t2-t1
delta_lr = round(delta.total_seconds(), 3)
print('LogisticRegression takes : ', delta_lr, 'Seconds')

cm = confusion_matrix(y_test, y_pred_lr)
print(cm)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(lr, X_test, y_test, cmap=plt.cm.Blues, display_labels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix')
plt.show()

coefficients = lr.coef_[0]

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%
# SVC
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
}

svc = SVC()
svc_cv = GridSearchCV(estimator=svc, param_grid=parameters, cv=10, scoring = 'recall').fit(X_train, y_train)

print('Tuned hyper parameters : ', svc_cv.best_params_)
print('Accuracy : ', svc_cv.best_score_)

t1 = datetime.now()
svc = SVC(**svc_cv.best_params_).fit(X_train, y_train)
t2 = datetime.now()

y_pred_svc = svc.predict(X_test)

delta = t2-t1
delta_svc = round(delta.total_seconds(), 3)
print('SVC : ', delta_svc, 'Seconds')

metrics.pair_confusion_matrix(y_test, y_pred_svc)

cr = metrics.classification_report(y_test, y_pred_svc)
print(cr)

cm = confusion_matrix(y_test, y_pred_svc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dont Have Stroke', 'Have Stroke'])

# Adjust layout for better visibility
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='', ax=ax)
ax.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()

# No feature selection for svc with a non-linear kernel

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

# %%[markdown]
# Random Forest
rf = RandomForestClassifier(random_state=42)
parameters_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}

# Perform Grid Search to find the best hyperparameters
rf_cv = GridSearchCV(estimator=rf, param_grid=parameters_rf, cv=10, scoring='recall').fit(X_train, y_train)

print('Tuned hyperparameters for Random Forest:', rf_cv.best_params_)
print('Best accuracy for Random Forest:', rf_cv.best_score_)

# Train Random Forest with the best hyperparameters
t1_rf = datetime.now()
rf_best = RandomForestClassifier(**rf_cv.best_params_, random_state=42).fit(X_train, y_train)
t2_rf = datetime.now()

y_pred_rf = rf_best.predict(X_test)
cr_rf = metrics.classification_report(y_test, y_pred_rf)
print(cr_rf)

delta_rf = round((t2_rf - t1_rf).total_seconds(), 3)
print('Random Forest takes:', delta_rf, 'Seconds')

# Plotting the confusion matrix for Random Forest
plt.figure(figsize=(8, 6))
plot_confusion_matrix(rf_best, X_test, y_test, cmap=plt.cm.Blues, display_labels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix for Random Forest')
plt.show()

# Feature Importance Plot from Random forest
feature_importances = rf_best.feature_importances_
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.show()

# Visualize one tree from the forest
plt.figure(figsize=(20, 10))
plot_tree(rf_best.estimators_[0], feature_names=X_train.columns, filled=True, rounded=True, class_names=['No Stroke', 'Stroke'])
plt.title("Example Decision Tree from Random Forest")
plt.show()

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''
# %%[markdown]
# Gaussian Naive Bayes Classifier
start_time=time.time()
naive_bayes = GaussianNB()

naive_bayes.fit(X_train, y_train)
y_pred_naive_bayes = naive_bayes.predict(X_test)
end_time=time.time()

cr_naive_bayes = metrics.classification_report(y_test, y_pred_naive_bayes)
print(cr_naive_bayes)

cm_naive_bayes = confusion_matrix(y_test, y_pred_naive_bayes)
print('Confusion Matrix for Naive Bayes:\n', cm_naive_bayes)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(naive_bayes, X_test, y_test, cmap=plt.cm.Blues, display_labels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix for Naive Bayes')
plt.show()

# Time the naive bayes algorithm
print(f"Training time: {end_time - start_time} seconds")
# No feature importance for Naive Bayes

#%%[markdown]
'''-----------------------------------------------------------------------------------------'''

#%%[markdown]
'''Feature importance analysis from two of our five models tells us that
age, average glucose level, and BMI are the top three most important explanatory
variables to predict whether an individual will have a stroke or not. Feature importance
for logistic regression (our best model after fitting with all six of our selected parameters) 
specifically indicates that age, smoking_status, and heart_disease are the best features. 
We fit these two models and compare them to see whether we can get a model with the top 
three parameters predicting stroke occurences with reasonable performance.'''

#%%
# Logistic Regression after feature selection - age, avg_glucose_level, bmi
X_reduced = X[['age', 'avg_glucose_level', 'bmi']]
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=1)

X_train_orig, X_test_orig, y_train_orig, y_test_orig = X_train, X_test, y_train, y_test

smt = SMOTE()
X_train, y_train = smt.fit_resample(X_train, y_train)

# %%
t1 = datetime.now()
lr = LogisticRegression(**lr_cv.best_params_).fit(X_train, y_train)
t2 = datetime.now()
y_pred_lr = lr.predict(X_test)
cr = metrics.classification_report(y_test, y_pred_lr)
print(cr)
delta = t2-t1
delta_lr = round(delta.total_seconds(), 3)
print('LogisticRegression takes : ', delta_lr, 'Seconds')

cm = confusion_matrix(y_test, y_pred_lr)
print(cm)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(lr, X_test, y_test, cmap=plt.cm.Blues, display_labels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix')
plt.show()

#%%
# Logistic Regression after feature selection - age, smoking_status, heart_disease
X_reduced = X[['age', 'smoking_status', 'heart_disease']]
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=1)

X_train_orig, X_test_orig, y_train_orig, y_test_orig = X_train, X_test, y_train, y_test

smt = SMOTE()
X_train, y_train = smt.fit_resample(X_train, y_train)

# %%
t1 = datetime.now()
lr = LogisticRegression(**lr_cv.best_params_).fit(X_train, y_train)
t2 = datetime.now()
y_pred_lr = lr.predict(X_test)
cr = metrics.classification_report(y_test, y_pred_lr)
print(cr)
delta = t2-t1
delta_lr = round(delta.total_seconds(), 3)
print('LogisticRegression takes : ', delta_lr, 'Seconds')

cm = confusion_matrix(y_test, y_pred_lr)
print(cm)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(lr, X_test, y_test, cmap=plt.cm.Blues, display_labels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix')
plt.show()
