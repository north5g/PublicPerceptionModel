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

models = {
    "openclip": 0,
    "siglip": 1,
    "streetclip": 2,
    "qwen": 3
}

selected_model = args.model_name

# test training set
training_args = TrainingArguments(
    output_dir="/tmp/no_save",                  # Temporary directory, not used
    overwrite_output_dir=True,                  # Avoid warnings about existing dirs
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    max_steps=10,
    logging_dir="/tmp/no_logs",                 # Also point to a throwaway location
    logging_steps=999999,                       # Effectively disables logging
    evaluation_strategy="no",                   # Disable evaluation
    save_strategy="no",                         # Disable checkpoint saving
    load_best_model_at_end=False,               # Don't track or restore best model
    report_to="none"                            # Disable reporting to any platform
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
)

trainer.train()