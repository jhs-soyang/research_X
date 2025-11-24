import pandas as pd
import os

# ==========================================
# Settings
# ==========================================
# Path to the data file
data_path = r'D:\projects\twitter\The_Climate_Change_Twitter_Dataset.csv'
# Where to save results
save_dir = r'D:\projects\twitter\results'


print("Loading data...")
df = pd.read_csv(data_path)

# 1. Clean data
# Make gender and stance lowercase to avoid mismatch (e.g., "Male" vs "male")
print("Cleaning text data...")
df['gender'] = df['gender'].str.lower()
df['stance'] = df['stance'].str.lower()

# ==========================================
# Part A: Gender Distribution
# ==========================================
print("Calculating gender distribution...")
# Count rows by gender
gender_counts = df['gender'].value_counts().reset_index()
gender_counts.columns = ['gender', 'count']

# Save to CSV
save_path_1 = os.path.join(save_dir, 'gender_distribution.csv')
gender_counts.to_csv(save_path_1, index=False)
print(f"Saved: {save_path_1}")

# ==========================================
# Part B: Gender vs Stance
# ==========================================
print("Calculating gender and stance...")
# Group by gender and stance, then unstack to make a table
cross_table = df.groupby(['gender', 'stance']).size().unstack(fill_value=0)

# Calculate ratios manually
# High school style: Iterating to make it look simple
results = []
genders = ['male', 'female', 'undefined']
stances = ['believer', 'denier', 'neutral']

for g in genders:
    if g in cross_table.index:
        row_data = {'Gender': g.capitalize()}
        total = cross_table.loc[g].sum()
        
        for s in stances:
            count = cross_table.loc[g].get(s, 0)
            ratio = count / total if total > 0 else 0
            row_data[f'{s.capitalize()} (Frequency)'] = count
            row_data[f'{s.capitalize()} (Ratio)'] = ratio
        
        row_data['Total Tweets'] = total
        results.append(row_data)

# Save to CSV
df_result = pd.DataFrame(results)
save_path_2 = os.path.join(save_dir, 'gender_stance_cross.csv')
df_result.to_csv(save_path_2, index=False)
print(f"Saved: {save_path_2}")

print("Step 1 Complete!")