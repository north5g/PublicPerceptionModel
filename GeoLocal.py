from transformers import TrainingArguments
from PlacePulseDataset import PlacePulseDataset
from encoder import load_encoder
from RegressionalTrainer import VisionTextRegressor
from torchvision import transforms
from transformers import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

selected_model = args.model_name

# actual training set
training_args = TrainingArguments(
    output_dir="./results_[{}]".format(selected_model),
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    num_train_epochs=15,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_dir="./logs_[{}]".format(selected_model),
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none"
)

from sklearn.metrics import mean_squared_error, r2_score

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    return {
        "mse": mean_squared_error(labels, preds),
        "r2": r2_score(labels, preds)
    }

# 1. Load encoder
encoder, processor, encoder_dim, image_size = load_encoder(selected_model)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# 2. Init model
model = VisionTextRegressor(
    encoder=encoder,
    num_study_types=6,
    image_size=image_size
)
dataset = PlacePulseDataset(transform = transform)
train_dataset, eval_dataset = dataset.split()

from transformers import default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator
)

trainer.train()