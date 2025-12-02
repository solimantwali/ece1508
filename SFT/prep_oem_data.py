# prepare_oem_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your OEM spreadsheet
df = pd.read_csv("OEM_master.csv")  # <-- change to your actual path
print(df.head())

# 2. Keep only what we need for SFT
needed_cols = ["sentence", "Human"]
df = df[needed_cols].dropna()

# Optional: strip whitespace
df["sentence"] = df["sentence"].str.strip()
df["Human"] = df["Human"].str.strip()

# 3. Train/val split
train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# 4. Save for later use
train_df.to_csv("oem_train.csv", index=False)
val_df.to_csv("oem_val.csv", index=False)
