import torch
from transformers import TrainingArguments
from PlacePulseDataset import PlacePulseDataset
import RegressionalTrainer
from encoder import load_encoder
from RegressionalTrainer import GradNormCallback, L1Trainer, VisionTextRegressor, get_param_groups
from torchvision import transforms
from transformers import Trainer, EarlyStoppingCallback
import argparse
from transformations import transformation, multi_transformation

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset", type=str, default="all", choices=["all", "safe", "lively", "clean", "wealthy", "depressing", "beautiful"])
parser.add_argument("--transform", type=str, default="none", choices=["none", "zoomed", "greyscale", "contrast", "lowresolution", "flipped"])
parser.add_argument("--instances", type=int, default=1)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selected_model = args.model_name
selected_dataset = args.dataset
transform_type = args.transform
instances = args.instances

# TODO: CHANGE "METRICS_FOR_BEST_MODEL"
# actual training set
if transform_type == "none":
    output_dir="./{}/results_trial2_[{}]".format(selected_model, selected_dataset)
else:
    output_dir="./{}/results_trial2_[{}_{}]".format(selected_model, selected_dataset, transform_type)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    max_grad_norm=1.0,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    num_train_epochs=15,
    lr_scheduler_type="cosine",
    logging_dir="./{}/logs_[{}]".format(selected_model, selected_dataset),
    logging_steps=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="spearman",
    greater_is_better=True,
    fp16=True,
    report_to="none"
)

# TODO: CHANGE METRICS FROM R2 TO SOMETHING ELSE (HUBER?)
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import torchvision.transforms.functional as TF


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = preds.flatten()
    labels = labels.flatten()

    spearman = spearmanr(labels, preds).correlation
    pearson = pearsonr(labels, preds)[0]

    return {
        "mae": mean_absolute_error(labels, preds),
        "r2": r2_score(labels, preds),
        "spearman": float(spearman),
        "pearson": float(pearson)
    }

# 1. Load encoder
encoder, processor, encoder_dim, image_size = load_encoder(selected_model, device=device, use_device_map=False)

mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
std  = getattr(processor, "image_std",  [0.229, 0.224, 0.225])


transform = transformation(transform_type, image_size, mean, std)

dataset = PlacePulseDataset(transform_data = (image_size, mean, std), instances=instances, study_type_filter=selected_dataset, fraction=1.0, random_state=42)
train_dataset, eval_dataset, test_dataset = dataset.split()
train_mean = train_dataset.df['normalized_score'].mean()
train_std = train_dataset.df['normalized_score'].std()

# 2. Init model
model = VisionTextRegressor(
    encoder=encoder,
    num_study_types=6,
    study_embed_dim=128,
    encoder_dim = encoder_dim, 
    image_size=image_size,
    freeze=False,          # set False to fine-tune encoder
    l1_lambda=0.0,
    smooth_l1_beta=0.5,
    label_mean=train_mean,
    label_std=train_std,
    loss_type="huber"
)
model.to(device)


from transformers import default_data_collator

from torch.optim import AdamW
from transformers import get_scheduler

param_groups = get_param_groups(model, encoder_lr=1e-6, head_lr=1e-4, weight_decay=0.02)
optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


trainer = L1Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    l1_lambda=1e-5,         # small L1 on head (optional)
    l1_scope="regressor",
    loss_type="huber",
    callbacks=[GradNormCallback(threshold=5.0), EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=compute_metrics,   # your compute_metrics
    optimizers=(optimizer, None)  # scheduler optional
)

trainer.train()

test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test set results:")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")