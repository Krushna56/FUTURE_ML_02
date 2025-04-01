import pandas as pd
import glob
import os

folder_path = r"FUTURE_ML_02\Dataset"  # Update with your actual path

# Detect CSV files
csv_files = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.CSV"))

if not csv_files:
    print("No CSV files found! Check your folder path.")
else:
    print(f"Found {len(csv_files)} CSV files. Merging now...")

# Read and concatenate all CSV files
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")  # Handle bad lines
        dfs.append(df)
        print(f"Loaded {file} with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Merge only if we have valid data
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True, sort=False)
    merged_df.to_csv("merged_stock_data.csv", index=False)
    print("Merging complete! Final dataset shape:", merged_df.shape)
    print(merged_df.head())  # Preview merged data
else:
    print("No valid CSV files were merged.")
