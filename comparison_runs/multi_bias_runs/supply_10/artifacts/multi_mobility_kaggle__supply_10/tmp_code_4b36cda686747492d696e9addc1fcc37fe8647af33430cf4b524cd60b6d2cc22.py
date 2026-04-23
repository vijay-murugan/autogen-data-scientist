import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Ensure output directory exists
output_dir = r'C:\Users\rufus\Projects\autogen-data-scientist\comparison_runs\multi_bias_runs\supply_10\artifacts\multi_mobility_kaggle__supply_10'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset with proper encoding
df = pd.read_csv(r'C:\Users\rufus\Projects\autogen-data-scientist\data\benchmark\kaggle\lakshmi25npathi_bike_sharing_day.csv', encoding='latin-1')

# Map season to labels
season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
df['season_label'] = df['season'].map(season_map)
df['workingday_label'] = df['workingday'].map({0: 'Non-Working', 1: 'Working'})

# Create features for a simple baseline model
feature_cols = ['temp', 'hum', 'windspeed', 'season', 'workingday', 'holiday', 'weekday', 'weathersit']
X = df[feature_cols]
y = df['cnt']

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Generate predictions on full dataset
df['predicted_cnt'] = model.predict(X)
df['actual_cnt'] = df['cnt']
df['absolute_error'] = np.abs(df['actual_cnt'] - df['predicted_cnt'])
df['error'] = df['actual_cnt'] - df['predicted_cnt']  # signed error for bias analysis

print("=" * 80)
print("STEP 3: SLICE-LEVEL MAE ANALYSIS")
print("=" * 80)

# Calculate MAE by slice (season × workingday)
slice_stats = df.groupby(['season_label', 'workingday_label']).agg(
    mae=('absolute_error', 'mean'),
    count=('absolute_error', 'count'),
    mean_error=('error', 'mean'),
    std_error=('error', 'std'),
    median_error=('error', 'median')
).reset_index()

# Create slice identifier
slice_stats['slice'] = slice_stats['season_label'] + ' - ' + slice_stats['workingday_label']

# Sort by MAE descending
slice_stats_sorted = slice_stats.sort_values('mae', ascending=False).reset_index(drop=True)

print("\nMAE by (Season × Workingday) Slice - RANKED:")
print("-" * 80)
for idx, row in slice_stats_sorted.iterrows():
    print(f"Rank {idx+1}: {row['slice']:25} | MAE: {row['mae']:8.2f} | n={row['count']:4} | Mean Error: {row['mean_error']:8.2f} | Std: {row['std_error']:8.2f}")

# Identify top 3 worst slices
top_3_worst = slice_stats_sorted.head(3)
print("\n" + "=" * 80)
print("TOP 3 WORST-PERFORMING SLICES")
print("=" * 80)
for idx, row in top_3_worst.iterrows():
    print(f"\n{idx+1}. {row['slice']}")
    print(f"   MAE: {row['mae']:.2f}")
    print(f"   Sample Size (n): {row['count']}")
    print(f"   Mean Error (Bias): {row['mean_error']:.2f}")
    print(f"   Std Error: {row['std_error']:.2f}")

# Statistical significance check
total_n = len(df)
min_threshold = total_n * 0.05  # 5% of total data
print(f"\nStatistical Significance Check:")
print(f"  Total records: {total_n}")
print(f"  Minimum threshold (5%): {min_threshold:.0f}")
print(f"  All top 3 slices meet threshold: {all(top_3_worst['count'] >= min_threshold)}")

# Save slice statistics to JSON
slice_stats_dict = {
    "all_slices": slice_stats_sorted.to_dict('records'),
    "top_3_worst": top_3_worst.to_dict('records'),
    "total_records": int(total_n),
    "significance_threshold": float(min_threshold)
}

with open(os.path.join(output_dir, 'slice_mae_analysis.json'), 'w') as f:
    json.dump(slice_stats_dict, f, indent=2)

print(f"\nSlice statistics saved to: {os.path.join(output_dir, 'slice_mae_analysis.json')}")
