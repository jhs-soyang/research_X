import pandas as pd
import os

# ==========================================
# Settings
# ==========================================
data_path = r'D:\projects\twitter\The_Climate_Change_Twitter_Dataset.csv'
output_dir = r'D:\projects\twitter\results'

# Create folder if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data for topic analysis...")
df = pd.read_csv(data_path)

# 1. Clean Text Data
print("Cleaning text data...")
df['gender'] = df['gender'].str.lower()
df['stance'] = df['stance'].str.lower()
df['topic'] = df['topic'].astype(str) # Ensure topic is string

# ==========================================================
# [FIX] Convert 'aggressiveness' from Text to Number
# The error happened because the computer cannot calculate the mean of words.
# We change: "aggressive" -> 1, "not aggressive" -> 0
# ==========================================================
print("Converting aggressiveness to numbers...")

def convert_agg(value):
    # Make sure it is string
    text = str(value).lower()
    if 'not aggressive' in text:
        return 0
    elif 'aggressive' in text:
        return 1
    else:
        return 0 # Default for safety

# Apply this function to the column
df['aggressiveness_score'] = df['aggressiveness'].apply(convert_agg)


# Get unique topics
topics = df['topic'].dropna().unique()

# ==========================================
# Task A: Gender Distribution by Topic & Stance
# (Needed for Bubble Chart)
# ==========================================
print("Analyzing Topic - Stance - Gender...")

results_a = []
stance_types = ['believer', 'denier', 'neutral']
gender_types = ['male', 'female']

for t in topics:
    if t == 'nan': continue # Skip empty topics

    # Filter by topic
    topic_data = df[df['topic'] == t]
    
    for s in stance_types:
        # Filter by stance within topic
        ts_data = topic_data[topic_data['stance'] == s]
        total_count = len(ts_data)
        
        # Count genders
        g_counts = ts_data['gender'].value_counts()
        
        for g in gender_types:
            count = g_counts.get(g, 0)
            # Calculate percentage
            pct = (count / total_count * 100) if total_count > 0 else 0
            
            results_a.append({
                'Topic': t,
                'Stance': s.capitalize(),
                'Gender': g.capitalize(),
                'Count': count,
                'Percentage within Topic-Stance': pct
            })

df_task_a = pd.DataFrame(results_a)
save_path_a = os.path.join(output_dir, 'gender_distribution_by_topic_stance.csv')
df_task_a.to_csv(save_path_a, index=False)
print(f"Saved: {save_path_a}")

# ==========================================
# Task B: Topic Sentiment & Aggressiveness
# (Needed for Scatter Plot)
# ==========================================
print("Analyzing Sentiment & Aggressiveness by Topic...")

results_b = []

for t in topics:
    if t == 'nan': continue

    topic_data = df[df['topic'] == t]
    
    # Calculate stats for Male and Female
    for g in gender_types:
        g_data = topic_data[topic_data['gender'] == g]
        
        if len(g_data) > 0:
            # Now we use 'sentiment' (assuming it is numeric) 
            # and our new 'aggressiveness_score' (which is definitely numeric)
            mean_sent = g_data['sentiment'].mean()
            mean_agg = g_data['aggressiveness_score'].mean() 
            
            results_b.append({
                'Topic': t,
                'Gender': g.capitalize(),
                'Mean_Sentiment': mean_sent,
                'Mean_Aggressiveness': mean_agg,
                'Count': len(g_data)
            })

df_task_b = pd.DataFrame(results_b)
save_path_b = os.path.join(output_dir, 'topic_sentiment_analysis.csv')
df_task_b.to_csv(save_path_b, index=False)
print(f"Saved: {save_path_b}")

print("Step 3 Complete!")