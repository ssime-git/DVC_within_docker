import argparse
from sklearn.datasets import load_iris
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="Path to save the raw Iris CSV")
args = parser.parse_args()

print("Fetching Iris dataset...")
iris = load_iris(as_frame=True)
df = iris.frame
# Replace spaces in column names and make target explicit
df.columns = [col.replace('(cm)', '').replace(' ', '_') for col in df.columns]
df.rename(columns={'target': 'species'}, inplace=True)
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Ensure output directory exists
os.makedirs(os.path.dirname(args.output), exist_ok=True)

df.to_csv(args.output, index=False)
print(f"Iris dataset saved to {args.output}")