import subprocess
import argparse
import sys

# Accept comma-separated list as a string
parser = argparse.ArgumentParser()
parser.add_argument(
    "--models", 
    type=str, 
    default="all", 
    help="Comma-separated list of models or 'all'"
)
parser.add_argument(
    "--dataset", 
    type=str, 
    default="all", 
    help="Comma-separated list of datasets or 'all'"
)
args = parser.parse_args()

# Allowed values
allowed_models = ["openclip", "siglip", "streetclip", "dinov2", "blip2"]
allowed_datasets = ["safe", "lively", "clean", "wealthy", "depressing", "beautiful", "all"]

# Parse model list
if args.models.lower() == "all":
    models = allowed_models
else:
    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in allowed_models:
            raise ValueError(f"Invalid model: {m}")

# Parse dataset list
if args.dataset.lower() == "all":
    selected_datasets = allowed_datasets
else:
    selected_datasets = [d.strip() for d in args.dataset.split(",")]
    for d in selected_datasets:
        if d not in allowed_datasets:
            raise ValueError(f"Invalid dataset: {d}")

for model in models:
    for dataset in selected_datasets:
        print(f"🚀 Starting training for: {model} on {dataset}")
        result = subprocess.run(["python3", "GeoLocal.py", "--model_name", model, "--dataset", dataset])

        if result.returncode is not 0:
            print(f"❌ Training failed for {model} on {dataset}. Exiting early.")
            sys.exit(result.returncode)
        else:
            print(f"✅ Finished training for {model} on {dataset}\n")

print("All models have been processed successfully.")