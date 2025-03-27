# src/train.py

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib       # For saving the model
import os           # For path manipulation (dirname, makedirs)
import dagshub      # For configuring MLflow with DagsHub
import mlflow       # For logging experiments
import sys          # For exiting with error codes
import traceback    # For printing detailed error tracebacks

# --- Argument Parsing ---
# Define expected command-line arguments passed from dvc.yaml
parser = argparse.ArgumentParser(description="Train a Logistic Regression model on Iris data and log to DagsHub/MLflow.")
parser.add_argument("--input", required=True, help="Path to the processed input data CSV file.")
parser.add_argument("--model-output", required=True, help="Path where the trained model (.joblib) should be saved.")
parser.add_argument("--C", type=float, required=True, help="Logistic Regression regularization parameter C.")
parser.add_argument("--solver", type=str, required=True, help="Algorithm to use in the optimization problem (e.g., 'liblinear').")
parser.add_argument("--test-size", type=float, required=True, help="Proportion of the dataset to include in the test split.")
parser.add_argument("--random-state", type=int, required=True, help="Controls the shuffling applied to the data before applying the split.")
args = parser.parse_args()

print(f"--- Training Script Started ---")
print(f"Script Arguments Received:")
print(f"  --input: {args.input}")
print(f"  --model-output: {args.model_output}")
print(f"  --C: {args.C}")
print(f"  --solver: {args.solver}")
print(f"  --test-size: {args.test_size}")
print(f"  --random-state: {args.random_state}")

# --- Main Execution Block with Error Handling ---
try:
    # --- Load Data ---
    print(f"\nLoading data from '{args.input}'...")
    try:
        df = pd.read_csv(args.input)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"!!! ERROR: Input data file not found at '{args.input}'", file=sys.stderr)
        sys.exit(1) # Exit immediately if data isn't found
    except Exception as e:
        print(f"!!! ERROR: Failed to load data from '{args.input}': {e}", file=sys.stderr)
        sys.exit(1) # Exit on other loading errors

    # --- Preprocessing for Training ---
    print("\nPreprocessing data (Label Encoding target)...")
    try:
        le = LabelEncoder()
        target_column = 'species' # Define target column name
        if target_column not in df.columns:
            print(f"!!! ERROR: Target column '{target_column}' not found in the input data.", file=sys.stderr)
            sys.exit(1)
        # Create the encoded column
        df[target_column + '_encoded'] = le.fit_transform(df[target_column])
        # Identify feature columns dynamically
        feature_cols = [col for col in df.columns if col not in [target_column, target_column + '_encoded']]
        if not feature_cols:
            print(f"!!! ERROR: No feature columns identified after excluding target.", file=sys.stderr)
            sys.exit(1)
        print(f"Using features: {feature_cols}")
        # Assign features (X) and target (y)
        X = df[feature_cols]
        y = df[target_column + '_encoded']
        print("Label encoding and feature selection complete.")
    except Exception as e:
        print(f"!!! ERROR: Failed during preprocessing: {e}", file=sys.stderr)
        traceback.print_exc() # Print details
        sys.exit(1)

    # --- Train/Test Split ---
    print(f"\nSplitting data (test_size={args.test_size}, random_state={args.random_state})...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y # Use stratify for classification
        )
        print("Data split successfully.")
        print(f"  Training set size: {X_train.shape[0]}")
        print(f"  Test set size: {X_test.shape[0]}")
    except Exception as e:
        print(f"!!! ERROR: Failed during train/test split: {e}", file=sys.stderr)
        traceback.print_exc() # Print details
        sys.exit(1)

    # --- DagsHub/MLflow Configuration ---
    print("\nConfiguring DagsHub/MLflow...")
    # This relies on environment variables (MLFLOW_*, DAGSHUB_USER_TOKEN) being set correctly
    try:
        dagshub.init(repo_owner="ssime-git", repo_name="DVC_within_docker")
        print("DagsHub initialization complete (MLflow configured).")
    except Exception as e:
        print(f"!!! ERROR: Failed to initialize DagsHub: {e}", file=sys.stderr)
        print("!!! Ensure MLFLOW_* and DAGSHUB_USER_TOKEN environment variables are set correctly.", file=sys.stderr)
        traceback.print_exc() # Print details
        sys.exit(1)

    # --- MLflow Run ---
    print("\nStarting MLflow run...")
    try:
        with mlflow.start_run() as run: # Get the run object
            mlflow_run_id = run.info.run_id
            print(f"MLflow run started (Run ID: {mlflow_run_id}).")
            print(f"  View Run: https://dagshub.com/ssime-git/DVC_within_docker.mlflow/#/experiments/0/runs/{mlflow_run_id}") # Construct the URL

            # --- Log Hyperparameters ---
            print("Logging hyperparameters to MLflow...")
            mlflow.log_param('C', args.C)
            mlflow.log_param('solver', args.solver)
            mlflow.log_param('test_size', args.test_size)
            mlflow.log_param('random_state', args.random_state)
            mlflow.log_param('features', str(feature_cols)) # Log features as string
            print("Hyperparameters logged.")

            # --- Model Training ---
            print(f"Training LogisticRegression model...")
            # Increased max_iter slightly for solvers like 'lbfgs' if used later
            model = LogisticRegression(C=args.C, solver=args.solver, random_state=args.random_state, max_iter=200)
            model.fit(X_train, y_train)
            print("Model training complete.")

            # --- Evaluation ---
            print("Evaluating model on test set...")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {accuracy:.4f}")

            # --- Log Metrics ---
            print("Logging metrics to MLflow...")
            mlflow.log_metric('accuracy', accuracy)
            print("Metrics logged.")

            # --- Save Model Locally ---
            # Removed specific try/except here to let errors propagate
            print(f"Saving trained model locally to '{args.model_output}'...")
            output_dir = os.path.dirname(args.model_output)
            # Ensure output directory exists only if it's not the current directory
            if output_dir:
                 os.makedirs(output_dir, exist_ok=True)
                 print(f"Ensured output directory exists: '{output_dir}'")

            # --- THIS IS THE CRITICAL STEP ---
            joblib.dump(model, args.model_output)
            # ---------------------------------

            print("Model saved locally successfully.")

            # Optional: Log the model as an MLflow artifact
            # print("Logging model as MLflow artifact...")
            # try:
            #     mlflow.sklearn.log_model(model, "model") # Logs to 'model' directory within artifacts
            #     print("Model logged as artifact.")
            # except Exception as artifact_error:
            #     print(f"!!! WARNING: Failed log model artifact: {artifact_error}", file=sys.stderr)

            print("MLflow run completed successfully.")
        # End of 'with mlflow.start_run()'

    except Exception as mlflow_error:
        print(f"!!! ERROR: An error occurred during the MLflow run: {mlflow_error}", file=sys.stderr)
        traceback.print_exc() # Print details
        sys.exit(1)

    print("\n--- Training script finished successfully ---")
    sys.exit(0) # Explicitly exit with success code

# Catch any unexpected errors not caught within the main try block
except Exception as final_error:
    print(f"!!! An unexpected top-level error occurred: {final_error}", file=sys.stderr)
    traceback.print_exc() # Print details
    sys.exit(1) # Exit with non-zero code