"""
============================================
GRAPHICS IN PYTHON
============================================

AGENDA:
- High-level graphics with matplotlib and seaborn
- Custom graphics
- Layered graphics with plotly
============================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os

warnings.filterwarnings('ignore')

# Create plots directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# ============================================
# FUNCTIONS FOR GRAPHICS
# ============================================
# matplotlib.pyplot forms the main suite for plotting
# seaborn provides higher-level statistical visualization
# plotly provides interactive plotting capabilities
# ============================================

# ============================================
# HIGH-LEVEL GRAPHICS
# ============================================

# ============================================
# UNIVARIATE DATA: HISTOGRAM
# ============================================
# Load state data (using equivalent dataset)
# Since Python doesn't have built-in state.x77, we'll create it
state_income = np.array([3098, 3545, 4354, 3378, 4809, 4091, 5348, 4817, 
                         4815, 3694, 4091, 3875, 4842, 3974, 4481, 3897, 
                         3605, 3617, 3688, 4167, 4751, 4540, 3834, 3820, 
                         4188, 3635, 4205, 4476, 3942, 4563, 4566, 4119, 
                         3811, 3977, 4657, 3646, 3815, 4445, 3834, 4167, 
                         3907, 4701, 4425, 4139, 4364, 4537, 3795, 3821, 
                         4281, 4564])

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(state_income, bins=8, color='lightblue', edgecolor='black')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Histogram of State Income in 1977')
plt.savefig('plots/state_income_histogram.png', dpi=100, bbox_inches='tight')
plt.close()

print("Histogram saved to plots/state_income_histogram.png")

# ============================================
# UNIVARIATE DATA: HISTOGRAM (EARTHQUAKE DEPTHS)
# ============================================
# Load earthquake data from seaborn or create simulated data
# Simulating earthquake depths (in reality, load from actual source)
np.random.seed(42)
earthquake_depth = np.random.exponential(200, 1000)
earthquake_depth = earthquake_depth[earthquake_depth < 700]

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(earthquake_depth, bins=np.arange(0, 700, 70), 
         color='lightcoral', edgecolor='black')
plt.xlabel('Earthquake Depth')
plt.ylabel('Frequency')
plt.title('Histogram of Earthquake Depths')
plt.savefig('plots/earthquake_depth_histogram.png', dpi=100, bbox_inches='tight')
plt.close()

print("Histogram saved to plots/earthquake_depth_histogram.png")

# ============================================
# EMPIRICAL CDF
# ============================================
# Plot empirical CDF for state income data
plt.figure(figsize=(10, 6))
sorted_income = np.sort(state_income)
cumulative = np.arange(1, len(sorted_income) + 1) / len(sorted_income)
plt.step(sorted_income, cumulative, where='post')
plt.xlabel('Income')
plt.ylabel('Cumulative Probability')
plt.title('ECDF of State Income in 1977')
plt.grid(True, alpha=0.3)
plt.savefig('plots/state_income_ecdf.png', dpi=100, bbox_inches='tight')
plt.close()

print("Empirical CDF saved to plots/state_income_ecdf.png")

# Plot empirical CDF for earthquake depth data
plt.figure(figsize=(10, 6))
sorted_depth = np.sort(earthquake_depth)
cumulative = np.arange(1, len(sorted_depth) + 1) / len(sorted_depth)
plt.step(sorted_depth, cumulative, where='post')
plt.xlabel('Earthquake Depth')
plt.ylabel('Cumulative Probability')
plt.title('ECDF of Earthquake Depths')
plt.grid(True, alpha=0.3)
plt.savefig('plots/earthquake_depth_ecdf.png', dpi=100, bbox_inches='tight')
plt.close()

print("Empirical CDF saved to plots/earthquake_depth_ecdf.png")

# ============================================
# Q-Q PLOTS
# ============================================
# Q-Q plot for state income data
plt.figure(figsize=(10, 6))
stats.probplot(state_income, dist="norm", plot=plt)
plt.title('Q-Q Plot of State Income (1977)')
plt.savefig('plots/state_income_qqnorm.png', dpi=100, bbox_inches='tight')
plt.close()

print("QQ plot saved to plots/state_income_qqnorm.png")

# Q-Q plot for earthquake depth data
plt.figure(figsize=(10, 6))
stats.probplot(earthquake_depth, dist="norm", plot=plt)
plt.title('Q-Q Plot of Earthquake Depths')
plt.savefig('plots/earthquake_depth_qqnorm.png', dpi=100, bbox_inches='tight')
plt.close()

print("QQ plot saved to plots/earthquake_depth_qqnorm.png")

# ============================================
# BOX PLOTS
# ============================================
# Create insect spray data
spray_data = pd.DataFrame({
    'count': np.concatenate([
        np.random.poisson(15, 12),  # Spray A
        np.random.poisson(14, 12),  # Spray B
        np.random.poisson(3, 12),   # Spray C
        np.random.poisson(5, 12),   # Spray D
        np.random.poisson(4, 12),   # Spray E
        np.random.poisson(16, 12)   # Spray F
    ]),
    'spray': ['A']*12 + ['B']*12 + ['C']*12 + ['D']*12 + ['E']*12 + ['F']*12
})

plt.figure(figsize=(10, 6))
spray_data.boxplot(column='count', by='spray', grid=False)
plt.suptitle('')
plt.title('Insect Counts by Spray Type')
plt.xlabel('Spray')
plt.ylabel('Count')
plt.savefig('plots/insect_spray_boxplot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Box plot saved to plots/insect_spray_boxplot.png")

# ============================================
# SCATTERPLOTS
# ============================================
# Create earthquake location data
np.random.seed(42)
quake_long = np.random.uniform(165, 185, 1000)
quake_lat = np.random.uniform(-38, -10, 1000)
quake_mag = np.random.uniform(4, 6.5, 1000)

plt.figure(figsize=(10, 6))
plt.scatter(quake_long, quake_lat, s=20, alpha=0.6, color='blue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Location of Earthquake Epicenters')
plt.savefig('plots/earthquake_locations_scatterplot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Scatterplot saved to plots/earthquake_locations_scatterplot.png")

# Scatterplot with symbols scaled by magnitude
plt.figure(figsize=(10, 6))
sizes = 10 ** quake_mag
plt.scatter(quake_long, quake_lat, s=sizes/100, alpha=0.3)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Location of Earthquake Epicenters')
plt.savefig('plots/earthquake_locations_symbols.png', dpi=100, bbox_inches='tight')
plt.close()

print("Symbol plot saved to plots/earthquake_locations_symbols.png")

# ============================================
# PAIRS PLOT (SCATTERPLOT MATRIX)
# ============================================
# Create trees dataset
trees_data = pd.DataFrame({
    'Girth': np.random.normal(13, 3, 31),
    'Height': np.random.normal(76, 6, 31),
    'Volume': np.random.gamma(15, 2, 31)
})

# Create pairs plot
plt.figure(figsize=(10, 10))
pd.plotting.scatter_matrix(trees_data, figsize=(10, 10), diagonal='hist')
plt.suptitle('Pairs Plot of Tree Measurements', y=1.0)
plt.savefig('plots/trees_pairs_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Pairs plot saved to plots/trees_pairs_plot.png")

# ============================================
# THREE DIMENSIONAL PLOTS
# ============================================
# Create criminal data (height vs finger length)
x = np.arange(0, 10, 0.5)
y = np.arange(0, 10, 0.5)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y) * 10 + np.random.normal(0, 1, X.shape)

# Contour plot
plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=15)
plt.xlabel('Height')
plt.ylabel('Finger Length')
plt.title('Contour Plot of Criminal Data')
plt.colorbar()
plt.savefig('plots/crimtab_contour_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Contour plot saved to plots/crimtab_contour_plot.png")

# Image plot
plt.figure(figsize=(10, 6))
plt.imshow(Z, extent=[0, 10, 0, 10], origin='lower', aspect='auto', cmap='viridis')
plt.xlabel('Height')
plt.ylabel('Finger Length')
plt.title('Image Plot of Criminal Data')
plt.colorbar()
plt.savefig('plots/crimtab_image_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Image plot saved to plots/crimtab_image_plot.png")

# Perspective plot (3D surface)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Height')
ax.set_ylabel('Finger Length')
ax.set_zlabel('Frequency')
ax.set_title('Perspective Plot of Criminal Data')
plt.savefig('plots/crimtab_persp_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Perspective plot saved to plots/crimtab_persp_plot.png")

# ============================================
# CATEGORICAL DATA: PIE CHARTS
# ============================================
pie_sales = np.array([0.12, 0.30, 0.26, 0.16, 0.04, 0.12])
labels = ['Blueberry', 'Cherry', 'Apple', 'Boston Creme', 'Other', 'Vanilla Creme']
colors = ['blue', 'red', 'green', 'wheat', 'orange', 'white']

plt.figure(figsize=(10, 6))
plt.pie(pie_sales, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Pie Sales Distribution')
plt.savefig('plots/pie_sales_chart.png', dpi=100, bbox_inches='tight')
plt.close()

print("Pie chart saved to plots/pie_sales_chart.png")

# Bar plot
va_deaths = pd.DataFrame({
    'Rural Male': [11.7, 18.1, 26.9, 41.0, 66.0],
    'Rural Female': [8.7, 11.7, 20.3, 30.9, 54.3],
    'Urban Male': [15.4, 24.3, 37.0, 54.6, 71.1],
    'Urban Female': [8.4, 13.6, 19.3, 35.1, 50.0]
}, index=['50-54', '55-59', '60-64', '65-69', '70-74'])

plt.figure(figsize=(10, 6))
va_deaths.T.plot(kind='bar', width=0.8)
plt.title('Virginia Death Rates per 1000 in 1940')
plt.xlabel('Population Group')
plt.ylabel('Death Rate per 1000')
plt.xticks(rotation=45)
plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('plots/va_deaths_barplot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Bar plot saved to plots/va_deaths_barplot.png")

# ============================================
# TIME SERIES PLOTS
# ============================================
# Create airline passengers time series
months = pd.date_range('1949-01', periods=144, freq='M')
passengers = 100 + np.arange(144) * 1.5 + np.sin(np.arange(144) * 2 * np.pi / 12) * 30

plt.figure(figsize=(10, 6))
plt.plot(months, passengers, color='blue', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Passengers (in thousands)')
plt.title('International Airline Passengers')
plt.grid(True, alpha=0.3)
plt.savefig('plots/airline_passengers_ts.png', dpi=100, bbox_inches='tight')
plt.close()

print("Time series plot saved to plots/airline_passengers_ts.png")

# Presidential approval ratings time series
quarters = pd.date_range('1945-01', periods=120, freq='Q')
approval = 60 + np.random.normal(0, 10, 120).cumsum() / 10

plt.figure(figsize=(10, 6))
plt.plot(quarters, approval, color='darkgreen', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Approval Rating')
plt.title('Presidential Approval Ratings')
plt.grid(True, alpha=0.3)
plt.savefig('plots/presidential_approval_ts.png', dpi=100, bbox_inches='tight')
plt.close()

print("Time series plot saved to plots/presidential_approval_ts.png")

# ============================================
# BINOMIAL DISTRIBUTION
# ============================================
x_binom = np.arange(0, 6)
y_binom = stats.binom.pmf(x_binom, n=5, p=0.4)

plt.figure(figsize=(10, 6))
plt.stem(x_binom, y_binom, basefmt=' ', linefmt='darkblue', markerfmt='o')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Binomial Distribution (n=5, p=0.4)')
plt.savefig('plots/binomial_distribution_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Binomial distribution plot saved to plots/binomial_distribution_plot.png")

# ============================================
# NORMAL DISTRIBUTION
# ============================================
x_norm = np.linspace(-3, 3, 600)
y_norm = stats.norm.pdf(x_norm)

plt.figure(figsize=(10, 6))
plt.plot(x_norm, y_norm, color='blue', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Normal Distribution')
plt.grid(True, alpha=0.3)
plt.savefig('plots/normal_distribution_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Normal distribution plot saved to plots/normal_distribution_plot.png")

# ============================================
# TWO EMPIRICAL CDFs: COMPARISON
# ============================================
# Simulate Puromycin-like data
np.random.seed(42)
treated = np.random.normal(140, 25, 23)
untreated = np.random.normal(110, 20, 23)

plt.figure(figsize=(10, 6))
sorted_treated = np.sort(treated)
sorted_untreated = np.sort(untreated)
cum_treated = np.arange(1, len(sorted_treated) + 1) / len(sorted_treated)
cum_untreated = np.arange(1, len(sorted_untreated) + 1) / len(sorted_untreated)

plt.step(sorted_treated, cum_treated, where='post', label='Treated', color='black')
plt.step(sorted_untreated, cum_untreated, where='post', label='Untreated', color='blue')
plt.xlim(60, 200)
plt.xlabel('Reaction Rate')
plt.ylabel('Cumulative Probability')
plt.title('Treated versus Untreated')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('plots/puromycin_ecdf_comparison.png', dpi=100, bbox_inches='tight')
plt.close()

print("Puromycin ECDF comparison saved to plots/puromycin_ecdf_comparison.png")

# ============================================
# MULTIPLE PLOTS ON ONE SET OF AXES
# ============================================
x_trig = np.linspace(0, 2 * np.pi, 100)
sine = np.sin(x_trig)
cosine = np.cos(x_trig)

plt.figure(figsize=(10, 6))
plt.plot(x_trig, sine, 'k-', linewidth=2, label='sin(x)')
plt.plot(x_trig, cosine, 'k--', linewidth=2, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine and Cosine Functions')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('plots/sine_cosine_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Sine and cosine plot saved to plots/sine_cosine_plot.png")

# ============================================
# MULTIPLE FRAME PLOTS
# ============================================
# Create precipitation data
np.random.seed(42)
precip = np.random.gamma(15, 2, 70)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Box plot
axes[0, 0].boxplot(precip)
axes[0, 0].set_title('Box Plot')
axes[0, 0].set_ylabel('Precipitation')

# Histogram
axes[0, 1].hist(precip, bins=15, color='lightblue', edgecolor='black')
axes[0, 1].set_title('Histogram')
axes[0, 1].set_xlabel('Precipitation')

# ECDF
sorted_precip = np.sort(precip)
cum_precip = np.arange(1, len(sorted_precip) + 1) / len(sorted_precip)
axes[1, 0].step(sorted_precip, cum_precip, where='post')
axes[1, 0].set_title('Empirical CDF')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(precip, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig('plots/precipitation_multiplot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Multiple frame plot saved to plots/precipitation_multiplot.png")

# ============================================
# STATISTICAL MODEL PLOT
# ============================================
# Create Puromycin-like data
np.random.seed(42)
conc = np.tile(np.array([0.02, 0.06, 0.11, 0.22, 0.56, 1.10]), 2)
rate = np.concatenate([
    np.array([47, 97, 123, 152, 191, 200]) + np.random.normal(0, 5, 6),  # Untreated
    np.array([76, 107, 139, 159, 201, 207]) + np.random.normal(0, 5, 6)  # Treated
])
state = ['Untreated']*6 + ['Treated']*6

plt.figure(figsize=(10, 6))
for i, s in enumerate(['Untreated', 'Treated']):
    mask = np.array(state) == s
    marker = 'o' if s == 'Untreated' else 's'
    plt.scatter(conc[mask], rate[mask], marker=marker, s=100, label=s)

plt.xlabel('Substrate Concentration')
plt.ylabel('Reaction Rate')
plt.title('Enzyme Reaction Rate vs Substrate Concentration')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('plots/puromycin_model_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Statistical model plot saved to plots/puromycin_model_plot.png")

# ============================================
# 3D SURFACE: SINC FUNCTION
# ============================================
x = np.linspace(-8, 8, 100)
y = np.linspace(-8, 8, 100)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R) / R
Z[R == 0] = 1  # Handle division by zero at origin

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('3D Surface: sin(r)/r')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig('plots/persp_wire_mesh.png', dpi=100, bbox_inches='tight')
plt.close()

print("3D wire mesh plot saved to plots/persp_wire_mesh.png")

# ============================================
# CUSTOM GRAPHICS: LEEMIS CHAPTER 21
# ============================================
fig, ax = plt.subplots(figsize=(10, 7))

# Set up empty plot with specific limits
ax.set_xlim(0, 9)
ax.set_ylim(-10, 20)
ax.set_xlabel('c(0, 9)')
ax.text(4.5, -12, 'margin 1', ha='center')
ax.text(-0.8, 5, 'margin 2', rotation=90, va='center')
ax.text(9.8, 5, 'margin 4', rotation=-90, va='center')

# Add text annotations
ax.text(2, 18, 'bold font', weight='bold', va='top')
ax.text(2, 16, 'italics font', style='italic', va='top')
ax.text(2, 14, 'bold & italics font', weight='bold', style='italic', va='top')
ax.text(2, 12, r'$\alpha\beta\Gamma$', fontsize=14, va='top')

# Right/top justified text
ax.text(7.5, 15, 'right/top justified', va='top')
ax.plot(7.5, 15, '+', markersize=15)

# Left/bottom justified text
ax.text(2, -9, 'left/bottom justified', va='bottom')

# Add curves
x_curve = np.linspace(0, 8, 100)
y_curve1 = x_curve**2 - 5*x_curve + 2
y_curve2 = -0.5*x_curve + 3
ax.plot(x_curve, y_curve1, 'k-', linewidth=2)
ax.plot(x_curve, y_curve2, 'k--', linewidth=2)

# Slanted text
ax.text(2.5, -2, 'slanted Text', rotation=15)

# Polygon (yellow triangle)
triangle = plt.Polygon([(0, -3), (1, 0), (0, 2)], color='yellow', edgecolor='black')
ax.add_patch(triangle)

# Plotting region label with arrow
ax.text(5, 9, 'plotting region', va='top')
ax.annotate('', xy=(3, 7), xytext=(4.5, 9), 
            arrowprops=dict(arrowstyle='->', lw=1))

# Mathematical expression
ax.text(6, -5, r'$\frac{\lambda_i}{2^x}$', fontsize=18)

# Add points
ax.plot(6, 3, 'ko', markersize=10, fillstyle='none')
ax.plot(4, -9, 'ko', markersize=7, fillstyle='none')

# Add text
ax.text(7.5, 3, 'w', color='blue', weight='bold', fontsize=14)
ax.text(8.5, 3, '8', fontsize=12)

ax.axis('off')
plt.savefig('plots/leemis_custom_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Leemis custom plot saved to plots/leemis_custom_plot.png")

# ============================================
# PLOTLY/SEABORN FOR LAYERED GRAPHICS
# ============================================
print("\n=== Note: ggplot2-style layered graphics ===")
print("For ggplot2-style graphics in Python, use:")
print("  - plotly.express for interactive plots")
print("  - plotnine (Python port of ggplot2)")
print("  - seaborn for statistical graphics")

# Example with seaborn
# Load or create FEV-like data
np.random.seed(42)
n_samples = 654
fev_data = pd.DataFrame({
    'age': np.random.randint(3, 20, n_samples),
    'fev': np.random.normal(2.5, 0.8, n_samples),
    'height': np.random.normal(65, 10, n_samples),
    'smoke': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'sex': np.random.choice([0, 1], n_samples)
})

# Adjust fev based on age and smoking
fev_data['fev'] = 1.0 + fev_data['age'] * 0.15 + np.random.normal(0, 0.3, n_samples)
fev_data.loc[fev_data['smoke'] == 1, 'fev'] -= 0.2

print("\n=== FEV Data Structure ===")
print(fev_data.dtypes)
print(f"\nShape: {fev_data.shape}")

# ============================================
# LAYERING WITH SEABORN
# ============================================
# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(data=fev_data, x='age', y='fev', alpha=0.5)
sns.lineplot(data=fev_data.groupby('age')['fev'].mean().reset_index(), 
             x='age', y='fev', color='red', linewidth=2)
plt.title('FEV vs Age with Mean Line')
plt.savefig('plots/fev_layered_plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("FEV layered plot saved to plots/fev_layered_plot.png")

# ============================================
# SMOOTHING
# ============================================
# Default smooth (LOWESS)
plt.figure(figsize=(10, 6))
sns.regplot(data=fev_data, x='age', y='fev', lowess=True, 
            scatter_kws={'alpha': 0.5})
plt.title('FEV vs Age with LOWESS Smooth')
plt.savefig('plots/fev_smooth_default.png', dpi=100, bbox_inches='tight')
plt.close()

print("FEV smooth (default LOWESS) saved to plots/fev_smooth_default.png")

# Linear regression
plt.figure(figsize=(10, 6))
sns.regplot(data=fev_data, x='age', y='fev', 
            scatter_kws={'alpha': 0.5})
plt.title('FEV vs Age with Linear Regression')
plt.savefig('plots/fev_smooth_lm.png', dpi=100, bbox_inches='tight')
plt.close()

print("FEV smooth (linear model) saved to plots/fev_smooth_lm.png")

# ============================================
# GROUPING
# ============================================
# Group by smoke
plt.figure(figsize=(10, 6))
sns.lmplot(data=fev_data, x='age', y='fev', hue='smoke', 
           lowess=True, height=6, aspect=1.5)
plt.title('FEV vs Age Grouped by Smoking Status')
plt.savefig('plots/fev_grouped_smoke_color.png', dpi=100, bbox_inches='tight')
plt.close()

print("FEV grouped by smoke (colored) saved to plots/fev_grouped_smoke_color.png")

# Linear regression by group
plt.figure(figsize=(10, 6))
sns.lmplot(data=fev_data, x='age', y='fev', hue='smoke', 
           height=6, aspect=1.5)
plt.title('FEV vs Age by Smoking (Linear)')
plt.savefig('plots/fev_grouped_smoke_lm.png', dpi=100, bbox_inches='tight')
plt.close()

print("FEV grouped by smoke (linear) saved to plots/fev_grouped_smoke_lm.png")

# ============================================
# FACETING
# ============================================
# Facet by sex and age groups
fev_data['age_group'] = pd.cut(fev_data['age'], bins=5)

g = sns.FacetGrid(fev_data, col='age_group', row='sex', height=3, aspect=1.5)
g.map(sns.scatterplot, 'height', 'fev', 'smoke', alpha=0.6)
g.add_legend()
g.fig.suptitle('FEV vs Height Faceted by Sex and Age Group', y=1.02)
plt.savefig('plots/fev_faceted_sex_age_group.png', dpi=100, bbox_inches='tight')
plt.close()

print("FEV faceted by sex and age_group saved to plots/fev_faceted_sex_age_group.png")

# ============================================
# SUMMARY
# ============================================
print("\n=== SUMMARY ===")
print("Python has strong graphic capabilities through matplotlib, seaborn, and plotly")
print("Graphing is an iterative process; Don't rely on the default options")
print("Avoid gimmicks, use the minimum amount of ink to get your point across")
print("A small table can be better than a large graph")
print("Carefully consider the size and shape of your graph, bigger is not always better")

print("\n=== Graphics Tutorial Complete ===")
print("All plots saved to plots/ directory")
