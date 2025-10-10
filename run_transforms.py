import subprocess
import argparse
import sys

# Accept comma-separated list as a string
parser = argparse.ArgumentParser()
parser.add_argument(
    "--models", 
    type=str, 
    default="streetclip", 
    help="Comma-separated list of models or 'all'"
)
parser.add_argument(
    "--dataset", 
    type=str, 
    default="all", 
    help="Comma-separated list of datasets or 'all'"
)
parser.add_argument(
    "--transform",
    type=str,
    default="all",
    help="Type of transform to apply during training"
)
args = parser.parse_args()

# Allowed values
allowed_models = ["openclip", "siglip", "streetclip", "dinov2", "blip2"]
allowed_datasets = ["safe", "lively", "clean", "wealthy", "depressing", "beautiful", "all"]
allowed_transforms = ["none", "zoomed", "greyscale", "contrast"]

# Parse model list
if args.models.lower() not in allowed_models:
    raise ValueError(f"Invalid model: {args.models}")
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

if args.transform.lower() == "all":
    transforms = allowed_transforms
else:
    transforms = [t.strip() for t in args.transform.split(",")]
    for t in transforms:
        if t not in allowed_transforms:
            raise ValueError(f"Invalid transform: {t}")

for model in models:
    for dataset in selected_datasets:
        for transform in transforms:
            print(f"🚀 Starting training for: {model} on {dataset} with transform {transform}")
            result = subprocess.run(["python3", "GeoLocal.py", "--model_name", model, "--dataset", dataset, "--transform", transform])

        if result.returncode != 0:
            print(f"❌ Training failed for {model} on {dataset}. Exiting early.")
            sys.exit(result.returncode)
        else:
            print(f"✅ Finished training for {model} on {dataset}\n")

print("All models have been processed successfully.")