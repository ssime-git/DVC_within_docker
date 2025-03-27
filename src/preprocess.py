import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to the raw Iris CSV")
parser.add_argument("--output", required=True, help="Path to save the processed Iris CSV")
args = parser.parse_args()

print(f"Reading data from {args.input}...")
df = pd.read_csv(args.input)

# Identify numeric features (excluding the species column)
numeric_cols = df.select_dtypes(include='number').columns.tolist()

print(f"Scaling numeric features: {numeric_cols}")
scaler = StandardScaler()
# Fit and transform the numeric columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Ensure output directory exists
os.makedirs(os.path.dirname(args.output), exist_ok=True)

df.to_csv(args.output, index=False)
print(f"Processed data saved to {args.output}")