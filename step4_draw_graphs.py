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
l1_file = os.path.join(base_dir, r'results_GenderStance/annual_stance_difference_L1.csv')
ratio_file = os.path.join(base_dir, r'results_GenderStance/annual_gender_stance_ratios.csv')

print("Loading data...")
df_l1 = pd.read_csv(l1_file)
df_ratio = pd.read_csv(ratio_file)

# Merge datasets for easier plotting
# We want one big table with columns: Year, Male_Believer, Female_Believer, etc.
df_wide = df_ratio.pivot(index='Year', columns='Gender', values=['Believer_Ratio', 'Denier_Ratio', 'Neutral_Ratio'])
# Flatten columns (e.g., ('Believer_Ratio', 'Male') -> 'Male_Believer')
df_wide.columns = [f"{col[1]}_{col[0].split('_')[0]}" for col in df_wide.columns]
df_wide = df_wide.reset_index()
# Add L1 data
df_merged = pd.merge(df_wide, df_l1, on='Year')

print("Data preparation complete.")

# ==========================================================
# Graph 1: Comprehensive Heatmap (Like Screenshot 3)
# Shows Stance Ratios for all years
# ==========================================================
print("Drawing Graph 1: Comprehensive Heatmap...")

# Prepare data for heatmap: Rows=Stance_Gender, Cols=Year
heatmap_data = df_ratio.copy()
heatmap_data['Label'] = heatmap_data['Gender'] + '_' + heatmap_data['Stance_Type'].apply(lambda x: x.split('_')[0]) if 'Stance_Type' in heatmap_data.columns else heatmap_data['Gender'] + '_' + "Stance" 

# Manual pivot because the data structure is simple
# We need rows: Female_Believer, Male_Believer, Female_Neutral...
# Columns: 2007, 2008...
years = sorted(df_ratio['Year'].unique())
rows = ['Female_Believer', 'Male_Believer', 'Female_Neutral', 'Male_Neutral', 'Female_Denier', 'Male_Denier']
matrix_data = []

for r in rows:
    gender, stance = r.split('_')
    row_vals = []
    for y in years:
        # Find value
        val = df_ratio[(df_ratio['Year'] == y) & (df_ratio['Gender'] == gender)][f'{stance}_Ratio'].values[0]
        row_vals.append(val)
    matrix_data.append(row_vals)

df_heatmap = pd.DataFrame(matrix_data, columns=years, index=rows)

plt.figure(figsize=(14, 8))
# Draw heatmap with numbers
sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="RdYlGn_r", linewidths=.5)

# Add Red Lines for "Shift Period" (2014-2015)
# In heatmap coordinates, 2014 is index 7 (if start 2007), etc. Find index manually.
# Let's say 2014 is the 7th column, 2015 is the 8th.
idx_2014 = list(years).index(2014)
plt.axvline(x=idx_2014, color='red', linewidth=3)
plt.axvline(x=idx_2014 + 2, color='red', linewidth=3) # End of 2015

plt.title('Comprehensive Gender-Stance Heatmap (2007-2019)\nEmphasizing 2014-2015 Structural Shift')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph1_Comprehensive_Heatmap.png'))
plt.close()


# ==========================================================
# Graph 2: Difference Heatmap (Like Screenshot 5)
# Shows (Female - Male) difference
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

# Add Yellow Box for 2014-2015
# x position, y position, width, height
# 2014 index is idx_2014
rect = patches.Rectangle((idx_2014, 0), 2, 3, linewidth=4, edgecolor='yellow', facecolor='none')
ax.add_patch(rect)

plt.title('Gender Stance Difference Heatmap (Female - Male)\nPositive: Female Higher, Negative: Male Higher')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph2_Difference_Heatmap.png'))
plt.close()


# ==========================================================
# Graph 3: Polar/Radar Charts (Like Screenshot 6)
# Showing geometric shape of opinions
# ==========================================================
print("Drawing Graph 3: Polar Analysis...")

target_years = [2008, 2011, 2014, 2015, 2017, 2019]
labels = ['Believer', 'Neutral', 'Denier']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1] # Close the loop

fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
axs = axs.flatten()

for i, y in enumerate(target_years):
    ax = axs[i]
    
    # Get data
    m_data = df_merged[df_merged['Year'] == y][['Male_Believer', 'Male_Neutral', 'Male_Denier']].values.flatten().tolist()
    f_data = df_merged[df_merged['Year'] == y][['Female_Believer', 'Female_Neutral', 'Female_Denier']].values.flatten().tolist()
    
    # Close the loop
    m_data += m_data[:1]
    f_data += f_data[:1]
    
    # Plot Male
    ax.plot(angles, m_data, color='blue', linewidth=1, label='Male')
    ax.fill(angles, m_data, color='blue', alpha=0.1)
    
    # Plot Female
    ax.plot(angles, f_data, color='red', linewidth=1, label='Female')
    ax.fill(angles, f_data, color='red', alpha=0.1)
    
    # Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(str(y), fontweight='bold')
    
    # Highlight Shift Period (2014, 2015) with red background
    if y in [2014, 2015]:
        ax.set_facecolor('#ffe6e6') # Light red
        ax.set_title(f"{y} (SHIFT PERIOD)", color='red', fontweight='bold')

    if i == 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.suptitle('Polar Analysis of Gender Stance Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph3_Polar_Charts.png'))
plt.close()


# ==========================================================
# Graph 4: Correlation Matrix & Time Series (Like Screenshot 1)
# ==========================================================
print("Drawing Graph 4: Correlation Matrix & Split Time Series...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Believer Correlation (Top Left)
x = df_merged['Male_Believer']
y = df_merged['Female_Believer']
r, p = stats.pearsonr(x, y)
axes[0, 0].scatter(x, y, c=df_merged['Year'], cmap='viridis', s=50, edgecolors='k')
m, b = np.polyfit(x, y, 1)
axes[0, 0].plot(x, m*x + b, 'r--')
axes[0, 0].set_title(f'Believer Stance Correlation\nr = {r:.3f}, p = {p:.3f}')
axes[0, 0].set_xlabel('Male Ratio')
axes[0, 0].set_ylabel('Female Ratio')

# 2. Denier Correlation (Top Right)
x = df_merged['Male_Denier']
y = df_merged['Female_Denier']
r, p = stats.pearsonr(x, y)
axes[0, 1].scatter(x, y, c=df_merged['Year'], cmap='magma', s=50, edgecolors='k')
m, b = np.polyfit(x, y, 1)
axes[0, 1].plot(x, m*x + b, 'r--')
axes[0, 1].set_title(f'Denier Stance Correlation\nr = {r:.3f}, p = {p:.3f}')
axes[0, 1].set_xlabel('Male Ratio')
axes[0, 1].set_ylabel('Female Ratio')

# 3. Neutral Correlation (Bottom Left)
x = df_merged['Male_Neutral']
y = df_merged['Female_Neutral']
r, p = stats.pearsonr(x, y)
axes[1, 0].scatter(x, y, c=df_merged['Year'], cmap='cividis', s=50, edgecolors='k')
m, b = np.polyfit(x, y, 1)
axes[1, 0].plot(x, m*x + b, 'r--')
axes[1, 0].set_title(f'Neutral Stance Correlation\nr = {r:.3f}, p = {p:.3f}')
axes[1, 0].set_xlabel('Male Ratio')
axes[1, 0].set_ylabel('Female Ratio')

# 4. Split Time Series (Bottom Right)
# Split data into 3 parts: Pre-2014, 2014-2015, Post-2015
df_pre = df_merged[df_merged['Year'] < 2014]
df_shift = df_merged[(df_merged['Year'] >= 2014) & (df_merged['Year'] <= 2015)]
df_post = df_merged[df_merged['Year'] > 2015]

ax4 = axes[1, 1]
ax4.plot(df_pre['Year'], df_pre['Stance_Difference_L1'], 'bo-', label='Pre-2014')
ax4.plot(df_shift['Year'], df_shift['Stance_Difference_L1'], 'rs-', linewidth=3, label='2014-2015 Shift')
ax4.plot(df_post['Year'], df_post['Stance_Difference_L1'], 'g^-', label='Post-2015')

# Add overall trend line just to look cool
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
# Graph 5: Dual Axis Analysis (Like Screenshot 2)
# ==========================================================
print("Drawing Graph 5: Dual Axis Analysis...")

fig, ax1 = plt.subplots(figsize=(12, 6))

# Left Axis: L1 Distance (Purple)
ax1.plot(df_merged['Year'], df_merged['Stance_Difference_L1'], 'o-', color='purple', linewidth=3, label='L1 Distance')
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('L1 Distance', color='purple', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='purple')
ax1.grid(True, axis='y', alpha=0.3)

# Right Axis: Believer Ratios (Cyan and Pink)
ax2 = ax1.twinx()
ax2.plot(df_merged['Year'], df_merged['Male_Believer'], 's-', color='cyan', linewidth=2, label='Male Believer Ratio')
ax2.plot(df_merged['Year'], df_merged['Female_Believer'], '^-', color='#ff69b4', linewidth=2, label='Female Believer Ratio') # Hot pink
ax2.set_ylabel('Believer Ratio', color='black', fontsize=12, fontweight='bold')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.title('Dual-Axis Analysis: L1 Distance and Gender Believer Ratios', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Graph5_Dual_Axis.png'))
plt.close()

print("All advanced graphs saved in 'Final_Graphs' folder!")