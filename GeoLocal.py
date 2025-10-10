import torch
from transformers import TrainingArguments
from PlacePulseDataset import PlacePulseDataset
import RegressionalTrainer
from encoder import load_encoder
from RegressionalTrainer import GradNormCallback, L1Trainer, VisionTextRegressor, get_param_groups
from torchvision import transforms
from transformers import Trainer, EarlyStoppingCallback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset", type=str, default="all", choices=["all", "safe", "lively", "clean", "wealthy", "depressing", "beautiful"])
parser.add_argument("--transform", type=str, default="none", choices=["none", "zoomed", "greyscale", "contrast"])
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selected_model = args.model_name
selected_dataset = args.dataset

# TODO: CHANGE "METRICS_FOR_BEST_MODEL"
# actual training set
training_args = TrainingArguments(
    output_dir="./{}/results_[{}]".format(selected_model, selected_dataset),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    max_grad_norm=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    num_train_epochs=10,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_dir="./{}/logs_[{}]".format(selected_model, selected_dataset),
    logging_steps=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none"
)

# TODO: CHANGE METRICS FROM R2 TO SOMETHING ELSE (HUBER?)
from sklearn.metrics import mean_absolute_error, r2_score
import torchvision.transforms.functional as TF


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    return {
        "mae": mean_absolute_error(labels, preds),
        "r2": r2_score(labels, preds)
    }

# 1. Load encoder
encoder, processor, encoder_dim, image_size = load_encoder(selected_model, device=device, use_device_map=False)

mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
std  = getattr(processor, "image_std",  [0.229, 0.224, 0.225])

def contrast_transform(factor: float):
    return transforms.Lambda(lambda img: TF.adjust_contrast(img, factor))

match args.transform:
    case "none":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # deterministic resize to final_size
            transforms.ToTensor(),                        # 0..1
            transforms.Normalize(mean=mean, std=std)     # encoder expectation
        ])

    case "zoomed":
        # zoom in by a zoom_factor (>1). We'll first resize up, then center-crop to final_size.
        zoom_factor = 1.3333333333  # 1 / 0.75, same visual zoom as your earlier 0.75 scale
        up_size = int(round(image_size * zoom_factor))
        transform = transforms.Compose([
            transforms.Resize((up_size, up_size)),       # enlarge
            transforms.CenterCrop(image_size),           # crop center -> zoomed view
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    case "greyscale":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # keep 3 channels for encoder
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    case "contrast":
        # deterministic contrast multiplier: choose a multiplier e.g. 1.25 (25% stronger)
        contrast_factor = 1.25
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            contrast_transform(contrast_factor),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    case _:
        raise ValueError(f"Unknown transform: {args.transform}")

dataset = PlacePulseDataset(transform = transform, study_type_filter=selected_dataset, fraction=1.0, random_state=42)
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
    freeze=True,          # set False to fine-tune encoder
    l1_lambda=0.0,
    smooth_l1_beta=1.0,
    label_mean=train_mean,
    label_std=train_std,
    loss_type="huber"
)
model.to(device)


from transformers import default_data_collator

from torch.optim import AdamW
from transformers import get_scheduler

param_groups = get_param_groups(model, encoder_lr=5e-5, head_lr=5e-4, weight_decay=0.01)
optimizer = AdamW(param_groups, lr=5e-6, betas=(0.9, 0.999), eps=1e-8)


trainer = L1Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    l1_lambda=1e-5,         # small L1 on head (optional)
    l1_scope="regressor",
    loss_type=None,         # fallback; model.loss_type will be used
    callbacks=[GradNormCallback(threshold=50.0)],
    compute_metrics=None,   # your compute_metrics
    optimizers=(optimizer, None)  # scheduler optional
)

trainer.train()

test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test set results:")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")