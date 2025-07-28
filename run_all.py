import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, required=True)
args = parser.parse_args()

version = args.version

models = ["openclip", "siglip", "streetclip", "qwen"]

if version == "test":
    print("Running test training set...")
    for model in models:
        print(f"--- Starting training for {model} ---")
        result = subprocess.run(["python3", "GeoLocalTest.py", "--model_name", model])

        if result.returncode != 0:
            print(f"Training failed for {model}. Exiting.")
            break
        else:
            print(f"--- Finished training for {model} ---\n")

elif version == "train":
    print("Training all models...")
    for model in models:
        print(f"--- Starting training for {model} ---")
        result = subprocess.run(["python3", "GeoLocal.py", "--model_name", model])

        if result.returncode != 0:
            print(f"Training failed for {model}. Exiting.")
            break
        else:
            print(f"--- Finished training for {model} ---\n")
