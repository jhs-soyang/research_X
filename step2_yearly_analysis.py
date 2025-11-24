import pandas as pd
import numpy as np
import os

# ==========================================
# Settings
# ==========================================
data_path = r'D:\projects\twitter\The_Climate_Change_Twitter_Dataset.csv'

# Output directory for results
output_dir = r'D:\projects\twitter\results\results_GenderStance'

# Create folder if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data for time analysis...")
df = pd.read_csv(data_path)

# Convert time string to datetime object
df['created_at'] = pd.to_datetime(df['created_at'])
# Make a new column for Year
df['Year'] = df['created_at'].dt.year

# Clean gender/stance
df['gender'] = df['gender'].str.lower()
df['stance'] = df['stance'].str.lower()

# Get list of years (2006, 2007, ...)
years = sorted(df['Year'].unique())

all_ratios = []
l1_differences = []

print("Starting yearly loop...")

for y in years:
    print(f"Processing year: {y}")
    
    # Get data for this year only
    year_data = df[df['Year'] == y]
    
    # Skip if too small
    if len(year_data) < 10:
        continue

    # Calculate ratios for Male and Female
    stats = {}
    
    for gender in ['male', 'female']:
        g_data = year_data[year_data['gender'] == gender]
        total = len(g_data)
        
        if total == 0:
            stats[gender] = [0, 0, 0] # believer, denier, neutral
            continue
            
        counts = g_data['stance'].value_counts()
        
        # Calculate ratios
        r_believer = counts.get('believer', 0) / total
        r_denier = counts.get('denier', 0) / total
        r_neutral = counts.get('neutral', 0) / total
        
        stats[gender] = [r_believer, r_denier, r_neutral]
        
        # Save ratio data for later use
        all_ratios.append({
            'Year': y,
            'Gender': gender.capitalize(),
            'Believer_Ratio': r_believer,
            'Denier_Ratio': r_denier,
            'Neutral_Ratio': r_neutral,
            'Total': total
        })

    # Calculate L1 Distance (Difference between Male and Female)
    # Math: sum of absolute differences
    male_vals = np.array(stats['male'])
    female_vals = np.array(stats['female'])
    
    # Simple absolute difference calculation
    diff = np.sum(np.abs(male_vals - female_vals))
    
    l1_differences.append({
        'Year': y,
        'Stance_Difference_L1': diff
    })

# Save the big summary files
ratio_df = pd.DataFrame(all_ratios)
ratio_df.to_csv(os.path.join(output_dir, 'annual_gender_stance_ratios.csv'), index=False)

l1_df = pd.DataFrame(l1_differences)
l1_df.to_csv(os.path.join(output_dir, 'annual_stance_difference_L1.csv'), index=False)

print("Step 2 Complete! Saved time series data.")