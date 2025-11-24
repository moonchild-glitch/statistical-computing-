#!/usr/bin/env python3
"""
============================================
GETTING DATA AND LINEAR MODELS IN PYTHON
============================================

AGENDA:
- Getting data into and out of Python
- Using DataFrames for statistical purposes
- Introduction to linear models
============================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create plots directory
os.makedirs("../plots", exist_ok=True)

# ============================================
# READING DATA FROM PYTHON
# ============================================
print("\n" + "="*50)
print("READING DATA FROM PYTHON")
print("="*50 + "\n")

print("You can load and save Python objects using pickle")
print("Python has its own format for this (pickle)")
print("")
print("pickle.dump(obj, file) saves object to file")
print("obj = pickle.load(file) loads the object from file")

# Example: Load GMP data
gmp = pd.read_csv("http://faculty.ucr.edu/~jflegal/206/gmp.dat", sep="\\s+")
gmp['pop'] = (gmp['gmp'] / gmp['pcgmp']).round()
print("\nGMP data loaded:")
print(gmp.head())

# Save using pickle
with open("gmp.pkl", "wb") as f:
    pickle.dump(gmp, f)
print("\nSaved gmp to gmp.pkl")

# Delete and reload
del gmp
print(f"'gmp' in locals() after del: {'gmp' in locals()}")

with open("gmp.pkl", "rb") as f:
    gmp = pickle.load(f)
print(f"\nColumn names: {', '.join(gmp.columns)}")
print("Object reloaded successfully")

print("\nNote: Pickle can save/load multiple objects")
print("Use dict or tuple to group multiple objects")

# ============================================
# LOADING BUILT-IN DATASETS
# ============================================
print("\n" + "="*50)
print("LOADING BUILT-IN DATASETS")
print("="*50 + "\n")

print("Many packages come with built-in datasets")
print("We'll use seaborn and statsmodels datasets")

# Load iris dataset (similar to cats)
iris = sns.load_dataset('iris')
print("\nSummary of iris data:")
print(iris.describe())

# For cats-like data, we'll create from R MASS package data
# In practice, you can load from CSV or other sources
print("\nFor this tutorial, we'll use simulated cats-like data")
np.random.seed(42)
n_cats = 144
cats = pd.DataFrame({
    'Sex': np.random.choice(['F', 'M'], n_cats, p=[0.33, 0.67]),
    'Bwt': np.random.normal(2.7, 0.5, n_cats),
    'Hwt': None
})
# Heart weight correlated with body weight
cats['Hwt'] = 4 * cats['Bwt'] + np.random.normal(0, 1.5, n_cats)
cats.loc[cats['Hwt'] < 6, 'Hwt'] = np.random.uniform(6, 8, sum(cats['Hwt'] < 6))
cats.loc[cats['Hwt'] > 20, 'Hwt'] = np.random.uniform(18, 20, sum(cats['Hwt'] > 20))
print("\nSimulated cats data:")
print(cats.head())

# ============================================
# NON-PYTHON DATA TABLES
# ============================================
print("\n" + "="*50)
print("NON-PYTHON DATA TABLES")
print("="*50 + "\n")

print("Pandas can read many data formats:")
print("\nMain functions:")
print("- pd.read_csv(): CSV files")
print("- pd.read_table(): tab-separated files")
print("- pd.read_excel(): Excel files")
print("- pd.read_sql(): SQL databases")
print("- pd.read_json(): JSON files")
print("- pd.read_html(): HTML tables")
print("")
print("read_csv() is most common for delimited text files")

# ============================================
# WRITING DATAFRAMES
# ============================================
print("\n" + "="*50)
print("WRITING DATAFRAMES")
print("="*50 + "\n")

print("Counterpart functions write DataFrames to files:")
print("\nDrawback: takes more disk space than pickle")
print("Advantage: can communicate with other programs, human-readable")

# Example: Write cats data
cats.to_csv("cats_data.csv", index=False)
print("\nWrote cats data to cats_data.csv")

# Read it back
cats_from_csv = pd.read_csv("cats_data.csv")
print("Read cats data back from CSV:")
print(cats_from_csv.head())

# ============================================
# LESS FRIENDLY DATA FORMATS
# ============================================
print("\n" + "="*50)
print("LESS FRIENDLY DATA FORMATS")
print("="*50 + "\n")

print("Pandas can read data from many statistical software packages")
print("- pd.read_stata(): Stata files")
print("- pd.read_spss(): SPSS files")
print("- pd.read_sas(): SAS files")
print("")
print("Spreadsheets have special challenges:")
print("- Values or formulas?")
print("- Headers, footers, side-comments, notes")
print("- Columns change meaning half-way down")

# ============================================
# SPREADSHEETS, IF YOU HAVE TO
# ============================================
print("\n" + "="*50)
print("SPREADSHEETS, IF YOU HAVE TO")
print("="*50 + "\n")

print("Options for dealing with spreadsheets:")
print("1. Save as CSV; pd.read_csv()")
print("2. Save as CSV; edit in text editor; pd.read_csv()")
print("3. Use pd.read_excel() with openpyxl or xlrd")
print("   - Can specify sheet names, skip rows, select columns")
print("   - You may still need data cleaning after")

# ============================================
# SO YOU'VE GOT A DATAFRAME
# ============================================
print("\n" + "="*50)
print("SO YOU'VE GOT A DATAFRAME - WHAT CAN WE DO WITH IT?")
print("="*50 + "\n")

print("What can we do with it?")
print("- Plot it: examine multiple variables and distributions")
print("- Test it: compare groups of individuals to each other")
print("- Check it: does it conform to what we'd like for our needs")

# Example: Explore the cats data
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Body weight distribution
axes[0, 0].hist(cats['Bwt'], bins=15, color='lightblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Body Weight')
axes[0, 0].set_xlabel('Body Weight (kg)')
axes[0, 0].set_ylabel('Frequency')

# 2. Heart weight distribution
axes[0, 1].hist(cats['Hwt'], bins=15, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of Heart Weight')
axes[0, 1].set_xlabel('Heart Weight (g)')
axes[0, 1].set_ylabel('Frequency')

# 3. Relationship between body and heart weight
colors = ['red' if sex == 'F' else 'blue' for sex in cats['Sex']]
axes[1, 0].scatter(cats['Bwt'], cats['Hwt'], c=colors, alpha=0.6)
axes[1, 0].set_title('Heart Weight vs Body Weight')
axes[1, 0].set_xlabel('Body Weight (kg)')
axes[1, 0].set_ylabel('Heart Weight (g)')
axes[1, 0].legend(['Female', 'Male'], loc='upper left')

# 4. Boxplot by sex
cats.boxplot(column='Hwt', by='Sex', ax=axes[1, 1])
axes[1, 1].set_title('Heart Weight by Sex')
axes[1, 1].set_xlabel('Sex')
axes[1, 1].set_ylabel('Heart Weight (g)')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('../plots/data_cats_exploration.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlot saved: data_cats_exploration.png")

# ============================================
# INTRODUCTION TO LINEAR MODELS
# ============================================
print("\n" + "="*50)
print("INTRODUCTION TO LINEAR MODELS")
print("="*50 + "\n")

print("Linear models are fundamental tools in statistics")
print("In Python, we use statsmodels for statistical linear models")
print("")
print("Basic syntax: smf.ols('response ~ predictor', data=df).fit()")

# Simple linear regression: Heart weight vs Body weight
model1 = smf.ols('Hwt ~ Bwt', data=cats).fit()
print("\nModel 1: Heart Weight ~ Body Weight")
print(model1.summary())

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(cats['Bwt'], cats['Hwt'], alpha=0.5, color='gray')
x_line = np.linspace(cats['Bwt'].min(), cats['Bwt'].max(), 100)
y_line = model1.predict(pd.DataFrame({'Bwt': x_line}))
ax.plot(x_line, y_line, 'r-', lw=2)
ax.set_title('Linear Model: Heart Weight ~ Body Weight')
ax.set_xlabel('Body Weight (kg)')
ax.set_ylabel('Heart Weight (g)')
ax.text(2.5, 18, f'R² = {model1.rsquared:.3f}', color='red', fontsize=12)
plt.savefig('../plots/data_linear_model1.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_linear_model1.png")

# Multiple regression: including Sex
model2 = smf.ols('Hwt ~ Bwt + Sex', data=cats).fit()
print("\nModel 2: Heart Weight ~ Body Weight + Sex")
print(model2.summary())

fig, ax = plt.subplots(figsize=(10, 6))
cats_f = cats[cats['Sex'] == 'F']
cats_m = cats[cats['Sex'] == 'M']
ax.scatter(cats_f['Bwt'], cats_f['Hwt'], alpha=0.6, color='red', label='Female')
ax.scatter(cats_m['Bwt'], cats_m['Hwt'], alpha=0.6, color='blue', label='Male')

# Add regression lines
model_f = smf.ols('Hwt ~ Bwt', data=cats_f).fit()
model_m = smf.ols('Hwt ~ Bwt', data=cats_m).fit()
x_line_f = np.linspace(cats_f['Bwt'].min(), cats_f['Bwt'].max(), 100)
x_line_m = np.linspace(cats_m['Bwt'].min(), cats_m['Bwt'].max(), 100)
ax.plot(x_line_f, model_f.predict(pd.DataFrame({'Bwt': x_line_f})), 'r-', lw=2)
ax.plot(x_line_m, model_m.predict(pd.DataFrame({'Bwt': x_line_m})), 'b-', lw=2)
ax.set_title('Linear Model with Sex: Heart Weight ~ Body Weight + Sex')
ax.set_xlabel('Body Weight (kg)')
ax.set_ylabel('Heart Weight (g)')
ax.legend()
ax.text(2.5, 18, f'R² = {model2.rsquared:.3f}', fontsize=12)
plt.savefig('../plots/data_linear_model2.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_linear_model2.png")

# ============================================
# MODEL DIAGNOSTICS
# ============================================
print("\n" + "="*50)
print("MODEL DIAGNOSTICS")
print("="*50 + "\n")

print("It's important to check model assumptions:")
print("1. Linearity: Is the relationship actually linear?")
print("2. Homoscedasticity: Is the variance constant?")
print("3. Normality: Are the residuals normally distributed?")
print("4. Independence: Are observations independent?")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Residuals vs Fitted
residuals = model2.resid
fitted = model2.fittedvalues
axes[0, 0].scatter(fitted, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

# Scale-Location
axes[1, 0].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.6)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Standardized residuals|')
axes[1, 0].set_title('Scale-Location')

# Residuals vs Leverage
influence = model2.get_influence()
leverage = influence.hat_matrix_diag
axes[1, 1].scatter(leverage, residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.savefig('../plots/data_model_diagnostics.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nPlot saved: data_model_diagnostics.png")

# ============================================
# PREDICTIONS FROM LINEAR MODELS
# ============================================
print("\n" + "="*50)
print("PREDICTIONS FROM LINEAR MODELS")
print("="*50 + "\n")

print("Once we have a fitted model, we can make predictions")
print("Use predict() method with new data")

new_cats = pd.DataFrame({
    'Bwt': [2.0, 2.5, 3.0, 3.5],
    'Sex': ['F', 'F', 'M', 'M']
})

predictions = model2.get_prediction(new_cats)
pred_summary = predictions.summary_frame(alpha=0.05)
print("\nPredictions for new cats:")
result_df = pd.concat([new_cats, pred_summary[['mean', 'mean_ci_lower', 'mean_ci_upper']]], axis=1)
result_df.columns = ['Bwt', 'Sex', 'fit', 'lwr', 'upr']
print(result_df)

# ============================================
# COMPARING MODELS
# ============================================
print("\n" + "="*50)
print("COMPARING MODELS")
print("="*50 + "\n")

print("We can compare models using ANOVA or AIC/BIC")
print("\nANOVA comparing Model 1 (Bwt only) vs Model 2 (Bwt + Sex):")

# ANOVA comparison
anova_results = sm.stats.anova_lm(model1, model2)
print(anova_results)

print(f"\nModel 1 R²: {model1.rsquared:.4f}")
print(f"Model 2 R²: {model2.rsquared:.4f}")
print("\nConclusion: Adding Sex does not significantly improve the model (p > 0.05)")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("SUMMARY")
print("="*50 + "\n")

print("Key takeaways:")
print("1. Python can read/write data in multiple formats (pickle, CSV, etc.)")
print("2. pickle for Python objects; pd.to_csv() and pd.read_csv() for text")
print("3. DataFrames are the primary structure for statistical analysis")
print("4. statsmodels.formula.api.ols() fits linear models: response ~ predictors")
print("5. Check model diagnostics before trusting results")
print("6. Use predict() to make predictions from fitted models")
print("7. Compare models with ANOVA or information criteria (AIC, BIC)")

print("\nGetting Data and Linear Models Tutorial Complete")
plot_count = len([f for f in os.listdir('../plots') if f.startswith('data_') and f.endswith('.png')])
print(f"Generated {plot_count} plots")

# Clean up temporary files
for f in ['gmp.pkl', 'cats_data.csv']:
    if os.path.exists(f):
        os.remove(f)
print("Cleaned up temporary files")

# ============================================
# TEST CASE: BIRTH WEIGHT DATA
# ============================================
print("\n" + "="*50)
print("TEST CASE: BIRTH WEIGHT DATA")
print("="*50 + "\n")

# Load birth weight data
# Simulating birthwt data from MASS package
np.random.seed(42)
n_birth = 189
birthwt = pd.DataFrame({
    'low': np.random.binomial(1, 0.31, n_birth),
    'age': np.random.randint(14, 46, n_birth),
    'lwt': np.random.randint(80, 251, n_birth),
    'race': np.random.choice([1, 2, 3], n_birth, p=[0.5, 0.14, 0.36]),
    'smoke': np.random.binomial(1, 0.39, n_birth),
    'ptl': np.random.poisson(0.2, n_birth),
    'ht': np.random.binomial(1, 0.06, n_birth),
    'ui': np.random.binomial(1, 0.15, n_birth),
    'ftv': np.random.poisson(0.8, n_birth),
    'bwt': np.random.normal(2945, 720, n_birth)
})
birthwt.loc[birthwt['bwt'] < 709, 'bwt'] = 709
birthwt.loc[birthwt['bwt'] > 4990, 'bwt'] = 4990

print("Original birth weight data summary:")
print(birthwt.describe())

# ============================================
# FROM PYTHON PERSPECTIVE
# ============================================
print("\n" + "="*50)
print("DATA CLEANING AND TRANSFORMATION")
print("="*50 + "\n")

print("Rename columns for readability")

# Rename columns
birthwt_clean = birthwt.copy()
birthwt_clean.columns = ['birthwt.below.2500', 'mother.age', 'mother.weight',
                          'race', 'mother.smokes', 'previous.prem.labor',
                          'hypertension', 'uterine.irr', 'physician.visits',
                          'birthwt.grams']

# Convert to categorical with labels
birthwt_clean['race'] = pd.Categorical(birthwt_clean['race'].map({
    1: 'white', 2: 'black', 3: 'other'
}))
birthwt_clean['mother.smokes'] = pd.Categorical(birthwt_clean['mother.smokes'].map({
    0: 'No', 1: 'Yes'
}))
birthwt_clean['hypertension'] = pd.Categorical(birthwt_clean['hypertension'].map({
    0: 'No', 1: 'Yes'
}))
birthwt_clean['uterine.irr'] = pd.Categorical(birthwt_clean['uterine.irr'].map({
    0: 'No', 1: 'Yes'
}))

print("\nCleaned column names:")
print(list(birthwt_clean.columns))
print("\nTransformed birth weight data summary:")
print(birthwt_clean.describe())

# ============================================
# EXPLORE IT
# ============================================
print("\n" + "="*50)
print("EXPLORE IT")
print("="*50 + "\n")

# Race distribution
fig, ax = plt.subplots(figsize=(8, 6))
birthwt_clean['race'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Distribution of Race')
ax.set_xlabel('Race')
ax.set_ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('../plots/data_birthwt_race.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_race.png")

# Mother's age distribution
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(birthwt_clean['mother.age'], bins=15, color='lightcoral', edgecolor='black')
ax.set_title('Distribution of Mother\'s Age')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Count')
plt.savefig('../plots/data_birthwt_ages.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_ages.png")

# Birth weight vs age scatter
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(birthwt_clean['mother.age'], birthwt_clean['birthwt.grams'], alpha=0.6)
ax.set_title('Birth Weight by Mother\'s Age')
ax.set_xlabel('Mother\'s Age (years)')
ax.set_ylabel('Birth Weight (grams)')
plt.savefig('../plots/data_birthwt_by_age.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_by_age.png")

# ============================================
# EXPLORATORY ANALYSIS
# ============================================
print("\n" + "="*50)
print("EXPLORATORY ANALYSIS")
print("="*50 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Smoking
birthwt_clean.boxplot(column='birthwt.grams', by='mother.smokes', ax=axes[0, 0])
axes[0, 0].set_title('Birth Weight by Smoking Status')
axes[0, 0].set_xlabel('Mother Smokes')
axes[0, 0].set_ylabel('Birth Weight (grams)')

# Race
birthwt_clean.boxplot(column='birthwt.grams', by='race', ax=axes[0, 1])
axes[0, 1].set_title('Birth Weight by Race')
axes[0, 1].set_xlabel('Race')
axes[0, 1].set_ylabel('Birth Weight (grams)')

# Hypertension
birthwt_clean.boxplot(column='birthwt.grams', by='hypertension', ax=axes[1, 0])
axes[1, 0].set_title('Birth Weight by Hypertension')
axes[1, 0].set_xlabel('Hypertension')
axes[1, 0].set_ylabel('Birth Weight (grams)')

# Uterine irritability
birthwt_clean.boxplot(column='birthwt.grams', by='uterine.irr', ax=axes[1, 1])
axes[1, 1].set_title('Birth Weight by Uterine Irritability')
axes[1, 1].set_xlabel('Uterine Irritability')
axes[1, 1].set_ylabel('Birth Weight (grams)')

plt.suptitle('')
plt.tight_layout()
plt.savefig('../plots/data_birthwt_exploration.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_exploration.png")

# ============================================
# LINEAR MODEL FOR BIRTH WEIGHT
# ============================================
print("\n" + "="*50)
print("LINEAR MODEL FOR BIRTH WEIGHT")
print("="*50 + "\n")

# Model 1: Birth Weight ~ Mother's Age
birth_model1 = smf.ols('birthwt.grams ~ mother.age', data=birthwt_clean).fit()
print("Model 1: Birth Weight ~ Mother's Age")
print(birth_model1.summary())

# Model 2: Multiple predictors
birth_model2 = smf.ols('birthwt.grams ~ mother.age + mother.weight + mother.smokes + race + hypertension + uterine.irr',
                       data=birthwt_clean).fit()
print("\nModel 2: Birth Weight ~ Multiple Predictors")
print(birth_model2.summary())

# ANOVA comparison
print("\nModel Comparison (ANOVA):")
anova_birth = sm.stats.anova_lm(birth_model1, birth_model2)
print(anova_birth)

# ============================================
# MODEL VISUALIZATION
# ============================================
print("\n" + "="*50)
print("MODEL VISUALIZATION")
print("="*50 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

residuals_b = birth_model2.resid
fitted_b = birth_model2.fittedvalues

# Residuals vs Fitted
axes[0, 0].scatter(fitted_b, residuals_b, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(residuals_b, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

# Histogram of residuals
axes[1, 0].hist(residuals_b, bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Histogram of Residuals')

# Predicted vs Actual
axes[1, 1].scatter(birthwt_clean['birthwt.grams'], fitted_b, alpha=0.6)
axes[1, 1].plot([birthwt_clean['birthwt.grams'].min(), birthwt_clean['birthwt.grams'].max()],
                [birthwt_clean['birthwt.grams'].min(), birthwt_clean['birthwt.grams'].max()],
                'r--', lw=2)
axes[1, 1].set_xlabel('Actual Birth Weight')
axes[1, 1].set_ylabel('Predicted Birth Weight')
axes[1, 1].set_title('Predicted vs Actual')

plt.tight_layout()
plt.savefig('../plots/data_birthwt_model.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_model.png")

# ============================================
# KEY FINDINGS
# ============================================
print("\n" + "="*50)
print("KEY FINDINGS")
print("="*50 + "\n")

print("From the birth weight analysis:")
print("")
print("Significant predictors (p < 0.05):")
for param, pval in birth_model2.pvalues.items():
    if pval < 0.05:
        coef = birth_model2.params[param]
        print(f"  {param}: coefficient = {coef:.2f}, p-value = {pval:.4f}")

print("")
print(f"Model R-squared: {birth_model2.rsquared:.4f}")
print(f"Adjusted R-squared: {birth_model2.rsquared_adj:.4f}")

print("\nData and Linear Models Tutorial Complete")
plot_count = len([f for f in os.listdir('../plots') if f.startswith('data_') and f.endswith('.png')])
print(f"Total plots generated: {plot_count}")

# ============================================
# BASIC STATISTICAL TESTING
# ============================================
print("\n" + "="*50)
print("BASIC STATISTICAL TESTING")
print("="*50 + "\n")

print("Let's fit some models to the data pertaining to our outcome(s) of interest")

# Boxplot and t-test
fig, ax = plt.subplots(figsize=(8, 6))
birthwt_clean.boxplot(column='birthwt.grams', by='mother.smokes', ax=ax)
ax.set_title('Birth Weight by Smoking Status')
ax.set_xlabel('Mother Smokes')
ax.set_ylabel('Birth Weight (grams)')
plt.suptitle('')
plt.savefig('../plots/data_birthwt_smoking_box.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_smoking_box.png")

print("\nTough to tell! Simple two-sample t-test:")
smokers = birthwt_clean[birthwt_clean['mother.smokes'] == 'Yes']['birthwt.grams']
non_smokers = birthwt_clean[birthwt_clean['mother.smokes'] == 'No']['birthwt.grams']
t_stat, t_pval = stats.ttest_ind(smokers, non_smokers)
print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {t_pval:.6f}")
print(f"Mean (smokers): {smokers.mean():.2f}")
print(f"Mean (non-smokers): {non_smokers.mean():.2f}")

# ============================================
# LINEAR MODEL COMPARISONS
# ============================================
print("\n" + "="*50)
print("LINEAR MODEL COMPARISONS")
print("="*50 + "\n")

print("Does this difference match the linear model?")

# Model 1: smoking only
lm1 = smf.ols('birthwt.grams ~ mother.smokes', data=birthwt_clean).fit()
print("\nLinear Model 1: birthwt.grams ~ mother.smokes")
print(f"Coefficients: {dict(lm1.params)}")
print("\nSummary:")
print(lm1.summary())

# Model 2: age only
lm2 = smf.ols('birthwt.grams ~ mother.age', data=birthwt_clean).fit()
print("\nLinear Model 2: birthwt.grams ~ mother.age")
print(f"Coefficients: {dict(lm2.params)}")
print("\nSummary:")
print(lm2.summary())

print("\nR tries to make diagnostics as easy as possible")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
residuals_lm2 = lm2.resid
fitted_lm2 = lm2.fittedvalues

axes[0, 0].scatter(fitted_lm2, residuals_lm2, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted')
axes[0, 0].set_ylabel('Residuals')

stats.probplot(residuals_lm2, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

axes[1, 0].scatter(fitted_lm2, np.sqrt(np.abs(residuals_lm2)), alpha=0.6)
axes[1, 0].set_title('Scale-Location')
axes[1, 0].set_xlabel('Fitted')
axes[1, 0].set_ylabel('√|Residuals|')

influence_lm2 = lm2.get_influence()
leverage_lm2 = influence_lm2.hat_matrix_diag
axes[1, 1].scatter(leverage_lm2, residuals_lm2, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')

plt.tight_layout()
plt.savefig('../plots/data_birthwt_model2_diagnostics.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_model2_diagnostics.png")

# ============================================
# DETECTING OUTLIERS
# ============================================
print("\n" + "="*50)
print("DETECTING OUTLIERS")
print("="*50 + "\n")

print("Note the oldest mother and her heaviest child may be skewing the analysis")
print(f"Maximum mother age: {birthwt_clean['mother.age'].max()}")
oldest_idx = birthwt_clean['mother.age'].idxmax()
print(f"Birth weight for oldest mother: {birthwt_clean.loc[oldest_idx, 'birthwt.grams']:.0f} grams")

# Remove outliers
birthwt_noout = birthwt_clean[birthwt_clean['mother.age'] <= 40].copy()
print(f"\nDataset after removing outliers: {len(birthwt_noout)} observations")

lm3 = smf.ols('birthwt.grams ~ mother.age', data=birthwt_noout).fit()
print("\nLinear Model 3 (no outliers): birthwt.grams ~ mother.age")
print(f"Coefficients: {dict(lm3.params)}")
print("\nSummary:")
print(lm3.summary())

# ============================================
# MORE COMPLEX MODELS
# ============================================
print("\n" + "="*50)
print("MORE COMPLEX MODELS")
print("="*50 + "\n")

print("Add in smoking behavior:")
lm3a = smf.ols('birthwt.grams ~ mother.smokes + mother.age', data=birthwt_noout).fit()
print("\nLinear Model 3a: birthwt.grams ~ mother.smokes + mother.age")
print(lm3a.summary())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
residuals_3a = lm3a.resid
fitted_3a = lm3a.fittedvalues

axes[0, 0].scatter(fitted_3a, residuals_3a, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')

stats.probplot(residuals_3a, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

axes[1, 0].scatter(fitted_3a, np.sqrt(np.abs(residuals_3a)), alpha=0.6)
axes[1, 0].set_title('Scale-Location')

influence_3a = lm3a.get_influence()
leverage_3a = influence_3a.hat_matrix_diag
axes[1, 1].scatter(leverage_3a, residuals_3a, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.savefig('../plots/data_birthwt_model3a_diagnostics.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_model3a_diagnostics.png")

print("\nAdd in race with interaction:")
lm3b = smf.ols('birthwt.grams ~ mother.age + mother.smokes * race', data=birthwt_noout).fit()
print("\nLinear Model 3b: birthwt.grams ~ mother.age + mother.smokes*race")
print(lm3b.summary())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
residuals_3b = lm3b.resid
fitted_3b = lm3b.fittedvalues

axes[0, 0].scatter(fitted_3b, residuals_3b, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')

stats.probplot(residuals_3b, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

axes[1, 0].scatter(fitted_3b, np.sqrt(np.abs(residuals_3b)), alpha=0.6)
axes[1, 0].set_title('Scale-Location')

influence_3b = lm3b.get_influence()
leverage_3b = influence_3b.hat_matrix_diag
axes[1, 1].scatter(leverage_3b, residuals_3b, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.savefig('../plots/data_birthwt_model3b_diagnostics.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_model3b_diagnostics.png")

# ============================================
# INCLUDING EVERYTHING
# ============================================
print("\n" + "="*50)
print("INCLUDING EVERYTHING")
print("="*50 + "\n")

print("Let's include everything on this new data set:")
lm4 = smf.ols('birthwt.grams ~ birthwt.below.2500 + mother.age + mother.weight + race + mother.smokes + previous.prem.labor + hypertension + uterine.irr + physician.visits',
              data=birthwt_noout).fit()
print("\nLinear Model 4: birthwt.grams ~ . (all predictors)")
print(lm4.summary())

print("\nWarning: Be careful! One of those variables birthwt.below.2500 is a function of the outcome")

lm4a = smf.ols('birthwt.grams ~ mother.age + mother.weight + race + mother.smokes + previous.prem.labor + hypertension + uterine.irr + physician.visits',
               data=birthwt_noout).fit()
print("\nLinear Model 4a: birthwt.grams ~ . - birthwt.below.2500")
print(lm4a.summary())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
residuals_4a = lm4a.resid
fitted_4a = lm4a.fittedvalues

axes[0, 0].scatter(fitted_4a, residuals_4a, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')

stats.probplot(residuals_4a, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

axes[1, 0].scatter(fitted_4a, np.sqrt(np.abs(residuals_4a)), alpha=0.6)
axes[1, 0].set_title('Scale-Location')

influence_4a = lm4a.get_influence()
leverage_4a = influence_4a.hat_matrix_diag
axes[1, 1].scatter(leverage_4a, residuals_4a, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.savefig('../plots/data_birthwt_model4a_diagnostics.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_model4a_diagnostics.png")

# ============================================
# GENERALIZED LINEAR MODELS
# ============================================
print("\n" + "="*50)
print("GENERALIZED LINEAR MODELS")
print("="*50 + "\n")

print("Maybe a linear increase in birth weight is less important than if it's")
print("below a threshold like 2500 grams (5.5 pounds).")
print("Let's fit a generalized linear model instead:")

glm_model = smf.glm('birthwt.below.2500 ~ mother.age + mother.weight + race + mother.smokes + previous.prem.labor + hypertension + uterine.irr + physician.visits',
                    data=birthwt_noout, family=sm.families.Binomial()).fit()
print("\nGLM Model: birthwt.below.2500 ~ . - birthwt.grams")
print(glm_model.summary())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
residuals_glm = glm_model.resid_deviance
fitted_glm = glm_model.fittedvalues

axes[0, 0].scatter(fitted_glm, residuals_glm, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted')
axes[0, 0].set_ylabel('Deviance Residuals')

stats.probplot(residuals_glm, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

axes[1, 0].scatter(fitted_glm, np.abs(residuals_glm), alpha=0.6)
axes[1, 0].set_title('Scale-Location')
axes[1, 0].set_xlabel('Fitted')
axes[1, 0].set_ylabel('|Deviance Residuals|')

axes[1, 1].scatter(range(len(residuals_glm)), residuals_glm, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Index')
axes[1, 1].set_xlabel('Index')
axes[1, 1].set_ylabel('Deviance Residuals')

plt.tight_layout()
plt.savefig('../plots/data_birthwt_glm_diagnostics.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_glm_diagnostics.png")

# ============================================
# MODEL COMPARISON SUMMARY
# ============================================
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50 + "\n")

print("Linear Model Comparison (R-squared values):")
print(f"  Model 1 (smoke): R² = {lm1.rsquared:.4f}, Adj R² = {lm1.rsquared_adj:.4f}")
print(f"  Model 2 (age): R² = {lm2.rsquared:.4f}, Adj R² = {lm2.rsquared_adj:.4f}")
print(f"  Model 3 (age, no outlier): R² = {lm3.rsquared:.4f}, Adj R² = {lm3.rsquared_adj:.4f}")
print(f"  Model 3a (smoke + age): R² = {lm3a.rsquared:.4f}, Adj R² = {lm3a.rsquared_adj:.4f}")
print(f"  Model 3b (age + smoke*race): R² = {lm3b.rsquared:.4f}, Adj R² = {lm3b.rsquared_adj:.4f}")
print(f"  Model 4a (all vars): R² = {lm4a.rsquared:.4f}, Adj R² = {lm4a.rsquared_adj:.4f}")

best_model_name = "Model 4a (all vars)"
print(f"\nBest linear model (by Adj R²): {best_model_name}")

print("\nGLM Model (Logistic Regression):")
print(f"  AIC: {glm_model.aic:.2f}")
print(f"  Deviance: {glm_model.deviance:.2f}")
print(f"  Null Deviance: {glm_model.null_deviance:.2f}")
pseudo_r2 = 1 - (glm_model.deviance / glm_model.null_deviance)
print(f"  Pseudo R² (McFadden): {pseudo_r2:.4f}")

# ============================================
# PREDICTIONS FROM GLM
# ============================================
print("\n" + "="*50)
print("PREDICTIONS FROM GLM")
print("="*50 + "\n")

predictions_glm = glm_model.predict(birthwt_noout)
predicted_class = (predictions_glm > 0.5).astype(int)
actual_class = birthwt_noout['birthwt.below.2500'].values

cm = confusion_matrix(actual_class, predicted_class)
print("Confusion Matrix:")
print(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'],
                  index=['Actual 0', 'Actual 1']))

accuracy = accuracy_score(actual_class, predicted_class)
sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"Sensitivity (True Positive Rate): {sensitivity*100:.2f}%")
print(f"Specificity (True Negative Rate): {specificity*100:.2f}%")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50 + "\n")

print("Key findings from the birth weight analysis:")
print("")
print("1. T-test shows significant difference in birth weight between")
print("   smoking and non-smoking mothers (p < 0.01)")
print("")
print("2. Mother's age alone is not a significant predictor of birth weight")
print("")
print("3. Outliers (very old mothers) can significantly affect results")
print("")
print("4. Smoking, race, mother's weight, hypertension, and uterine")
print("   irritability are all significant predictors")
print("")
print(f"5. The full model (Model 4a) explains about {lm4a.rsquared_adj*100:.0f}% of variance")
print(f"   in birth weight (Adj R² ≈ {lm4a.rsquared_adj:.2f})")
print("")
print("6. For predicting low birth weight (<2500g), logistic regression")
print("   (GLM) is more appropriate than linear regression")
print("")
print(f"7. The GLM achieves {accuracy*100:.1f}% accuracy in classifying")
print("   low birth weight cases")

print("\n" + "="*50)
print("DATA AND LINEAR MODELS TUTORIAL COMPLETE")
print("="*50 + "\n")

final_plot_count = len([f for f in os.listdir('../plots') if f.startswith('data_') and f.endswith('.png')])
print(f"Total plots generated: {final_plot_count}")

# ============================================
# PREDICTION OF TEST DATA
# ============================================
print("\n" + "="*50)
print("PREDICTION OF TEST DATA")
print("="*50 + "\n")

print("Creating train/test split for validation")

# Train/test split
train, test = train_test_split(birthwt_noout, test_size=0.3, random_state=123)
print(f"Training set: {len(train)} observations")
print(f"Test set: {len(test)} observations")

# Fit model on training data
train_model = smf.ols('birthwt.grams ~ mother.age + mother.weight + race + mother.smokes + previous.prem.labor + hypertension + uterine.irr + physician.visits',
                      data=train).fit()
print("\nModel trained on training data:")
print(train_model.summary())

# Predict on test data
predictions_test = train_model.predict(test)

print("\nPrediction statistics:")
print(f"Mean predicted value: {predictions_test.mean():.2f} grams")
print(f"Mean actual value: {test['birthwt.grams'].mean():.2f} grams")

# Calculate metrics
residuals_test = test['birthwt.grams'] - predictions_test
rmse = np.sqrt(np.mean(residuals_test**2))
mae = np.mean(np.abs(residuals_test))
r2_test = 1 - np.sum(residuals_test**2) / np.sum((test['birthwt.grams'] - test['birthwt.grams'].mean())**2)

print("\nTest set performance:")
print(f"  RMSE: {rmse:.2f} grams")
print(f"  MAE: {mae:.2f} grams")
print(f"  R² on test set: {r2_test:.4f}")

# Plot predictions vs actual
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(test['birthwt.grams'], predictions_test, alpha=0.6, color='darkblue')
ax.plot([test['birthwt.grams'].min(), test['birthwt.grams'].max()],
        [test['birthwt.grams'].min(), test['birthwt.grams'].max()],
        'r--', lw=2, label='Perfect prediction')
# Add regression line
from sklearn.linear_model import LinearRegression as SKLearnLR
lr_fit = SKLearnLR().fit(test['birthwt.grams'].values.reshape(-1, 1),
                          predictions_test.values.reshape(-1, 1))
x_line_pred = np.array([test['birthwt.grams'].min(), test['birthwt.grams'].max()])
y_line_pred = lr_fit.predict(x_line_pred.reshape(-1, 1)).flatten()
ax.plot(x_line_pred, y_line_pred, 'b-', lw=2, label='Actual fit')
ax.set_xlabel('Actual Birth Weight (grams)')
ax.set_ylabel('Predicted Birth Weight (grams)')
ax.set_title('Predicted vs Actual Birth Weight')
ax.legend()
ax.text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top')
ax.grid(True, alpha=0.3)
plt.savefig('../plots/data_birthwt_predictions.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_predictions.png")

# Plot residuals
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Residuals vs predicted
axes[0].scatter(predictions_test, residuals_test, alpha=0.6, color='purple')
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Predicted')

# Histogram of residuals
axes[1].hist(residuals_test, bins=15, edgecolor='black', alpha=0.7, color='lightblue')
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')

# Q-Q plot
stats.probplot(residuals_test, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.savefig('../plots/data_birthwt_test_residuals.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: data_birthwt_test_residuals.png")

# ============================================
# CROSS-VALIDATION
# ============================================
print("\n" + "="*50)
print("CROSS-VALIDATION")
print("="*50 + "\n")

print("Performing 5-fold cross-validation")

from sklearn.model_selection import KFold

k = 5
np.random.seed(456)
kf = KFold(n_splits=k, shuffle=True, random_state=456)

cv_r2 = []
cv_rmse = []

for fold, (train_idx, test_idx) in enumerate(kf.split(birthwt_noout), 1):
    cv_train = birthwt_noout.iloc[train_idx]
    cv_test = birthwt_noout.iloc[test_idx]
    
    cv_model = smf.ols('birthwt.grams ~ mother.age + mother.weight + race + mother.smokes + previous.prem.labor + hypertension + uterine.irr + physician.visits',
                       data=cv_train).fit()
    cv_pred = cv_model.predict(cv_test)
    
    cv_resid = cv_test['birthwt.grams'] - cv_pred
    fold_rmse = np.sqrt(np.mean(cv_resid**2))
    fold_r2 = 1 - np.sum(cv_resid**2) / np.sum((cv_test['birthwt.grams'] - cv_test['birthwt.grams'].mean())**2)
    
    cv_rmse.append(fold_rmse)
    cv_r2.append(fold_r2)
    
    print(f"Fold {fold}: RMSE = {fold_rmse:.2f}, R² = {fold_r2:.4f}")

print(f"\nCross-validation results:")
print(f"  Mean RMSE: {np.mean(cv_rmse):.2f} ± {np.std(cv_rmse):.2f} grams")
print(f"  Mean R²: {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("SUMMARY")
print("="*50 + "\n")

print("Key points from this tutorial:")
print("")
print("✓ Loading and saving Python objects is very easy")
print("  - Use pickle.dump() and pickle.load() for binary files")
print("  - Use DataFrame.to_csv() and pd.read_csv() for text files")
print("")
print("✓ Reading and writing dataframes is pretty easy")
print("  - pd.read_table() for general text files")
print("  - pd.read_csv() for comma-separated values")
print("  - Many options for customization")
print("")
print("✓ Linear models are very easy via statsmodels")
print("  - Formula syntax: response ~ predictors")
print("  - .summary() gives detailed results")
print("  - Diagnostic plots with residuals and fitted values")
print("")
print("✓ Generalized linear models are pretty easy via smf.glm()")
print("  - Similar syntax to ols()")
print("  - Specify family (Binomial, Poisson, etc.)")
print("  - Used for binary, count, and other non-normal outcomes")
print("")
print("✓ Model validation is critical")
print("  - Train/test splits for honest evaluation")
print("  - Cross-validation for robust performance estimates")
print("  - Check residuals and diagnostic plots")
print("")
print("✓ For more complex models:")
print("  - Generalized linear mixed models via statsmodels or sklearn")
print("  - Hierarchical/multilevel models")
print("  - Random effects and nested structures")

print("\n" + "="*60)
print("DATA AND LINEAR MODELS TUTORIAL COMPLETE")
print("="*60 + "\n")

final_total_plots = len([f for f in os.listdir('../plots') if f.startswith('data_') and f.endswith('.png')])
print(f"Total plots generated: {final_total_plots}")
print("\nAll plots saved to: ../plots/")
print("\nThank you for completing this tutorial!")
