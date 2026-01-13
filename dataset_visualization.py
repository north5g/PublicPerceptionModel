import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from PlacePulseDataset import PlacePulseDataset
import os

from encoder import load_encoder

parser = argparse.ArgumentParser()
parser.add_argument("--instances", type=int, default=1, help="Number of times dataset is repeated")
parser.add_argument("--noise", type=float, default=0.0, help="Amount of noise to add to normalized scores")
args = parser.parse_args()


study_types = ["safe", "lively", "clean", "wealthy", "depressing", "beautiful", "all"]

path = "./score_distributions/instances_{}/noise_{}".format(args.instances, args.noise)
if not os.path.exists(path):
    os.makedirs(path)

image_size = 224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform_data = (image_size, mean, std)

for type in study_types:
    dataset = PlacePulseDataset(transform_data=transform_data, instances=args.instances, study_type_filter=type, fraction=1.0, noise=args.noise, random_state=42)
    normalized_scores = dataset.df['normalized_score'] 

    plt.figure(figsize=(8, 5))
    sns.histplot(normalized_scores, kde=True, bins=40)
    plt.title(f"Distribution of Normalized PlacePulse Scores - {type.capitalize()} - Noise {args.noise} - Instances {args.instances}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(path, f"normalized_{type}.png"))
    plt.close()

    scores = dataset.df['score'] 

    plt.figure(figsize=(8, 5))
    sns.histplot(scores, kde=True, bins=40)
    plt.title(f"Distribution of PlacePulse Scores - {type.capitalize()} - Noise {args.noise} - Instances {args.instances}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(path, f"{type}.png"))
    plt.close()


