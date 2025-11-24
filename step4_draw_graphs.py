import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
import matplotlib.patches as patches

# ==========================================
# Load Data
# ==========================================
base_dir = r'D:\projects\twitter\results'
output_dir = os.path.join(base_dir, 'Final_Graphs')

# File paths
l1_file = os.path.join(base_dir, r'results_GenderStance\annual_stance_difference_L1.csv')
ratio_file = os.path.join(base_dir, r'results_GenderStance\annual_gender_stance_ratios.csv')

print("Loading data...")
df_l1 = pd.read_csv(l1_file)
df_ratio = pd.read_csv(ratio_file)

# Merge datasets for easier plotting
df_wide = df_ratio.pivot(index='Year', columns='Gender', values=['Believer_Ratio', 'Denier_Ratio', 'Neutral_Ratio'])
df_wide.columns = [f"{col[1]}_{col[0].split('_')[0]}" for col in df_wide.columns]
df_wide = df_wide.reset_index()
# Add L1 data
df_merged = pd.merge(df_wide, df_l1, on='Year')

print("Data preparation complete.")

# ==========================================================
# Graph 1: Comprehensive Heatmap
# ==========================================================
print("Drawing Graph 1: Comprehensive Heatmap...")

heatmap_data = df_ratio.copy()
years = sorted(df_ratio['Year'].unique())
rows = ['Female_Believer', 'Male_Believer', 'Female_Neutral', 'Male_Neutral', 'Female_Denier', 'Male_Denier']
matrix_data = []

for r in rows:
    gender, stance = r.split('_')
    row_vals = []
    for y in years:
        val = df_ratio[(df_ratio['Year'] == y) & (df_ratio['Gender'] == gender)][f'{stance}_Ratio'].values[0]
        row_vals.append(val)
    matrix_data.append(row_vals)

df_heatmap = pd.DataFrame(matrix_data, columns=years, index=rows)

plt.figure(figsize=(14, 8))
sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="RdYlGn_r", linewidths=.5)

idx_2014 = list(years).index(2014)
plt.axvline(x=idx_2014, color='red', linewidth=3)
plt.axvline(x=idx_2014 + 2, color='red', linewidth=3)

plt.title('Comprehensive Gender-Stance Heatmap (2007-2019)\nEmphasizing 2014-2015 Structural Shift')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph1_Comprehensive_Heatmap.png'))
plt.close()


# ==========================================================
# Graph 2: Difference Heatmap
# ==========================================================
print("Drawing Graph 2: Difference Heatmap...")

diff_rows = ['Believer', 'Denier', 'Neutral']
diff_matrix = []

for stance in diff_rows:
    row_vals = []
    for y in years:
        f_val = df_ratio[(df_ratio['Year'] == y) & (df_ratio['Gender'] == 'Female')][f'{stance}_Ratio'].values[0]
        m_val = df_ratio[(df_ratio['Year'] == y) & (df_ratio['Gender'] == 'Male')][f'{stance}_Ratio'].values[0]
        row_vals.append(f_val - m_val)
    diff_matrix.append(row_vals)

df_diff_map = pd.DataFrame(diff_matrix, columns=years, index=diff_rows)

plt.figure(figsize=(12, 6))
ax = sns.heatmap(df_diff_map, annot=True, fmt=".3f", cmap="RdBu_r", center=0, linewidths=.5)

rect = patches.Rectangle((idx_2014, 0), 2, 3, linewidth=4, edgecolor='yellow', facecolor='none')
ax.add_patch(rect)

plt.title('Gender Stance Difference Heatmap (Female - Male)\nPositive: Female Higher, Negative: Male Higher')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph2_Difference_Heatmap.png'))
plt.close()


# ==========================================================
# Graph 3: Polar/Radar Charts
# ==========================================================
print("Drawing Graph 3: Polar Analysis...")

target_years = [2008, 2011, 2014, 2015, 2017, 2019]
labels = ['Believer', 'Neutral', 'Denier']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
axs = axs.flatten()

for i, y in enumerate(target_years):
    ax = axs[i]
    m_data = df_merged[df_merged['Year'] == y][['Male_Believer', 'Male_Neutral', 'Male_Denier']].values.flatten().tolist()
    f_data = df_merged[df_merged['Year'] == y][['Female_Believer', 'Female_Neutral', 'Female_Denier']].values.flatten().tolist()
    
    m_data += m_data[:1]
    f_data += f_data[:1]
    
    ax.plot(angles, m_data, color='blue', linewidth=1, label='Male')
    ax.fill(angles, m_data, color='blue', alpha=0.1)
    ax.plot(angles, f_data, color='red', linewidth=1, label='Female')
    ax.fill(angles, f_data, color='red', alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(str(y), fontweight='bold')
    
    if y in [2014, 2015]:
        ax.set_facecolor('#ffe6e6')
        ax.set_title(f"{y} (SHIFT PERIOD)", color='red', fontweight='bold')

    if i == 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.suptitle('Polar Analysis of Gender Stance Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph3_Polar_Charts.png'))
plt.close()


# ==========================================================
# Graph 4: Correlation Matrix
# ==========================================================
print("Drawing Graph 4: Correlation Matrix...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplots 1-3 (Correlations)
plot_configs = [
    (0, 0, 'Male_Believer', 'Female_Believer', 'Believer Stance Correlation', 'viridis'),
    (0, 1, 'Male_Denier', 'Female_Denier', 'Denier Stance Correlation', 'magma'),
    (1, 0, 'Male_Neutral', 'Female_Neutral', 'Neutral Stance Correlation', 'cividis')
]

for r_idx, c_idx, x_col, y_col, title, cmap in plot_configs:
    x = df_merged[x_col]
    y = df_merged[y_col]
    r, p = stats.pearsonr(x, y)
    
    axes[r_idx, c_idx].scatter(x, y, c=df_merged['Year'], cmap=cmap, s=50, edgecolors='k')
    m, b = np.polyfit(x, y, 1)
    axes[r_idx, c_idx].plot(x, m*x + b, 'r--')
    axes[r_idx, c_idx].set_title(f'{title}\nr = {r:.3f}, p = {p:.3f}')
    axes[r_idx, c_idx].set_xlabel('Male Ratio')
    axes[r_idx, c_idx].set_ylabel('Female Ratio')

# Subplot 4 (Split Time Series)
df_pre = df_merged[df_merged['Year'] < 2014]
df_shift = df_merged[(df_merged['Year'] >= 2014) & (df_merged['Year'] <= 2015)]
df_post = df_merged[df_merged['Year'] > 2015]

ax4 = axes[1, 1]
ax4.plot(df_pre['Year'], df_pre['Stance_Difference_L1'], 'bo-', label='Pre-2014')
ax4.plot(df_shift['Year'], df_shift['Stance_Difference_L1'], 'rs-', linewidth=3, label='2014-2015 Shift')
ax4.plot(df_post['Year'], df_post['Stance_Difference_L1'], 'g^-', label='Post-2015')

z = np.polyfit(df_merged['Year'], df_merged['Stance_Difference_L1'], 1)
p_trend = np.poly1d(z)
ax4.plot(df_merged['Year'], p_trend(df_merged['Year']), "k--", alpha=0.3, label='Overall Trend')

ax4.set_title('Temporal Stability with Highlighted Shift')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph4_Correlation_Matrix.png'))
plt.close()


# ==========================================================
# Graph 5: Dual Axis Analysis
# ==========================================================
print("Drawing Graph 5: Dual Axis Analysis...")

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df_merged['Year'], df_merged['Stance_Difference_L1'], 'o-', color='purple', linewidth=3, label='L1 Distance')
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('L1 Distance', color='purple', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='purple')
ax1.grid(True, axis='y', alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df_merged['Year'], df_merged['Male_Believer'], 's-', color='cyan', linewidth=2, label='Male Believer Ratio')
ax2.plot(df_merged['Year'], df_merged['Female_Believer'], '^-', color='#ff69b4', linewidth=2, label='Female Believer Ratio')
ax2.set_ylabel('Believer Ratio', color='black', fontsize=12, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.title('Dual-Axis Analysis: L1 Distance and Gender Believer Ratios', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph5_Dual_Axis.png'))
plt.close()


# ==========================================================
# [ADDED] Graph 6: Simple Trend Analysis (Temporal Evolution)
# Matches the missing screenshot
# ==========================================================
print("Drawing Graph 6: Trend Analysis...")

x = df_merged['Year']
y = df_merged['Stance_Difference_L1']

# Calculate Linear Regression
slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
trend_line = slope * x + intercept

# Determine Conclusion Text based on p-value
if p_val < 0.05:
    conclusion = "Significant change"
else:
    conclusion = "No significant change"

plt.figure(figsize=(10, 6))

# Plot Data Points
plt.plot(x, y, 'o-', color='purple', label='Stance Difference (L1 Norm)')

# Plot Trend Line
plt.plot(x, trend_line, '--', color='red', label=f'Trend (Slope={slope:.3f}, p={p_val:.3f})')

plt.title(f'Temporal Evolution of Gender Stance Difference ({x.min()}-{x.max()})\nConclusion: {conclusion}')
plt.xlabel('Year')
plt.ylabel('L1 Norm Difference in Stance Ratios')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph6_Trend_Analysis.png'))
plt.close()

print("All advanced graphs saved in 'Final_Graphs' folder!")
