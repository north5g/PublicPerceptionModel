import torch
from transformers import TrainingArguments
from PlacePulseDataset import PlacePulseDataset
import RegressionalTrainer
from encoder import load_encoder
from sklearn.model_selection import train_test_split
from RegressionalTrainer import VisionTextRegressor
from torchvision import transforms
from transformers import Trainer, EarlyStoppingCallback
import argparse
from transformations import transformation, multi_transformation


from optimizer import create_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset", type=str, default="all", choices=["all", "safe", "lively", "clean", "wealthy", "depressing", "beautiful"])
parser.add_argument("--instances", type=int, default=1)
parser.add_argument("--noise", type=float, default=0.0, help="Amount of noise to add to normalized scores")
parser.add_argument(
    "--instance_transforms",
    type=str,
    default=None,
    help='Example: "none|contrast|zoomed,greyscale" (one per instance, separated by |)'
)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label = args.label
selected_model = args.model_name
selected_dataset = args.dataset
instances = args.instances

transform_map = {}
if args.instance_transforms is not None:
    if args.instance_transforms == "interactive":
        interactive = True
    else:
        interactive = False
        instance_transform_lists = args.instance_transforms.split("|")

        if len(instance_transform_lists) != instances:
            raise ValueError("instance_transforms count must match --instances")

        for i, part in enumerate(instance_transform_lists):
            transform_map[i] = [t.strip() for t in part.split(",") if t.strip()]

# TODO: CHANGE "METRICS_FOR_BEST_MODEL"
# actual training set

if instances == 1:
    output_dir="./{}/{}_[{}]".format(selected_model, label, selected_dataset)
else:
    output_dir="./{}/{}_[{}_{}]".format(selected_model, label, selected_dataset, instances)

training_args = TrainingArguments(
    output_dir="./{}/results_[{}]".format(selected_model, selected_dataset),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    num_train_epochs=15,
    lr_scheduler_type="cosine",
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
    metric_for_best_model="eval_huber",
    greater_is_better=False,
    fp16=True,
    report_to="none"
)

from sklearn.metrics import mean_squared_error
import numpy as np

def huber_loss_np(y_true, y_pred, delta=1.0):
    # robust Huber loss implemented in numpy
    resid = y_pred - y_true
    abs_r = np.abs(resid)
    is_small = abs_r <= delta
    sq = 0.5 * resid**2
    lin = delta * (abs_r - 0.5 * delta)
    return float(np.mean(np.where(is_small, sq, lin)))

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # flatten in case shape is (N,1)
    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()
    mse = mean_squared_error(labels, preds)
    # delta is tunable; see notes below
    huber = huber_loss_np(labels, preds, delta=1.0)
    return {
        "mse": mse,
        "eval_huber": huber
    }


# pass device into load_encoder so it tries to load directly onto GPU
encoder, processor, encoder_dim, image_size = load_encoder(selected_model, device=device, use_device_map=False)

mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
std  = getattr(processor, "image_std",  [0.229, 0.224, 0.225])

dataset = PlacePulseDataset(transform_data = (image_size, mean, std), instances=instances, study_type_filter=selected_dataset, fraction=1.0, noise=args.noise, transform_map=transform_map, interactive=interactive)
train_dataset, eval_dataset, test_dataset = dataset.split()
train_mean = train_dataset.df['normalized_score'].mean()
train_std = train_dataset.df['normalized_score'].std()

# 2. Init model
model = VisionTextRegressor(
    encoder=encoder,
    num_study_types=6,
    study_embed_dim=128,
    encoder_dim = encoder_dim, 
    processor=processor,
    image_size=image_size,
    freeze=False,          # set False to fine-tune encoder
    l1_lambda=0.0
)
model.to(device)

dataset = PlacePulseDataset(transform = transform)
train_dataset, eval_dataset, test_dataset = dataset.split()
train_dataset_half, _ = train_test_split(train_dataset, test_size=0.5, random_state=42)

from transformers import default_data_collator

optimizer, lr_scheduler = create_optimizer(model, training_args, train_dataset)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_half,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test set results:")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")