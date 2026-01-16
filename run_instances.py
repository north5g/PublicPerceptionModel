# premade for simplicity
# will include noise levels 3, 5, 7 on instances 2 and 3
# trial run 1 will include noise level 3 on instances 2

import subprocess
import sys

# transformations to run:
# one with no transformation
# one with single transformation (contrast)
# one with multiple transformations (zoomed, greyscale)
transform_combinations = [
    ["none"],
    ["contrast"],
    ["zoomed", "greyscale"]
]

# instances to run:
# one with 2 instances
# one with 3 instances
instances_list = [2, 3]

# noise levels to run:
# 3%, 5%, 7%
noise_levels = [3, 5, 7]

# possible options, stick with streetclip for trials
models = ["streetclip", "siglip", "openclip", "dinov2", "blip2"]

# datasets to run:
# larger the index, more polarizing the dataset 
datasets = ["all", "safe", "lively", "beautiful", "wealthy", "clean", "depressing"]


# run all combinations
combo_a = ["python3", "GeoLocal.py", "--model_name", str(models[0]), "--label", "combo_a", "--dataset", str(datasets[0]), "--instances", str(instances_list[0]), "--noise", str(noise_levels[0]), "--instance_transforms", str("|".join([",".join(transform_combinations[i]) for i in range(instances_list[0])]))]
combo_b = ["python3", "GeoLocal.py", "--model_name", str(models[0]), "--label", "combo_b", "--dataset", str(datasets[0]), "--instances", str(instances_list[1]), "--noise", str(noise_levels[0]), "--instance_transforms", str("|".join([",".join(transform_combinations[i]) for i in range(instances_list[1])]))]
combo_c = ["python3", "GeoLocal.py", "--model_name", str(models[0]), "--label", "combo_c", "--dataset", str(datasets[0]), "--instances", str(instances_list[0]), "--noise", str(noise_levels[1]), "--instance_transforms", str("|".join([",".join(transform_combinations[i]) for i in range(instances_list[0])]))]
combo_d = ["python3", "GeoLocal.py", "--model_name", str(models[0]), "--label", "combo_d", "--dataset", str(datasets[0]), "--instances", str(instances_list[1]), "--noise", str(noise_levels[1]), "--instance_transforms", str("|".join([",".join(transform_combinations[i]) for i in range(instances_list[1])]))]
combo_e = ["python3", "GeoLocal.py", "--model_name", str(models[0]), "--label", "combo_e", "--dataset", str(datasets[0]), "--instances", str(instances_list[0]), "--noise", str(noise_levels[2]), "--instance_transforms", str("|".join([",".join(transform_combinations[i]) for i in range(instances_list[0])]))]
combo_f = ["python3", "GeoLocal.py", "--model_name", str(models[0]), "--label", "combo_f", "--dataset", str(datasets[0]), "--instances", str(instances_list[1]), "--noise", str(noise_levels[2]), "--instance_transforms", str("|".join([",".join(transform_combinations[i]) for i in range(instances_list[1])]))]

# start with combo_a, update with others after successful runs
todo = [combo_a]

for combo in todo:
    print(f"🚀 Starting training on combo")
    result = subprocess.run(combo)
    if result.returncode != 0:
        print(f"❌ Training failed for command: {' '.join(combo)}. Exiting early.")
        sys.exit(result.returncode)
    else:
        print(f"✅ Finished training for command: {' '.join(combo)}\n")

print("All models have been processed successfully.")