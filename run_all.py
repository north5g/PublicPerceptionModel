import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, choices=["test", "train"], required=True)
args = parser.parse_args()

models = ["openclip", "siglip", "streetclip", "qwen"]

if args.version == "test":
    print("🧪 Running test training set...\n")
    script = "GeoLocalTest.py"
elif args.version == "train":
    print("🏋️ Training all models...\n")
    script = "GeoLocal.py"
else:
    print("❌ Invalid version. Use --version test or --version train.")
    sys.exit(1)

for model in models:
    print(f"🚀 Starting training for: {model}")
    result = subprocess.run(["python3", script, "--model_name", model])

    if result.returncode != 0:
        print(f"❌ Training failed for {model}. Exiting early.")
        sys.exit(result.returncode)
    else:
        print(f"✅ Finished training for {model}\n")
    print("All models have been processed successfully.")