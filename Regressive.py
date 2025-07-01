import torch
import random
import numpy as np
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments
from data_prep_weights import PlacePulseDatasetWeight
from torchvision import transforms
from trl import SFTTrainer, SFTConfig
import torch.nn as nn

# 1. Set random seed for reproducibility
def set_seed(seed=3407):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(3407)

# 2. Define transforms (with normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Study type mapping (should match your dataset)
STUDY_TYPES = [
    'safe', 'lively', 'clean', 'wealthy', 'depressing', 'beautiful'
]
STUDY_TYPE_TO_IDX = {name: i for i, name in enumerate(STUDY_TYPES)}

# 4. Load model and tokenizer
model_name = "Salesforce/blip2-opt-2.7b"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    use_gradient_checkpointing=True,
    device_map="auto"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

FastVisionModel.for_training(model) # Enable for training!

# 5. Optionally wrap model with LoRA/PEFT
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=False,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

train_dataset = PlacePulseDatasetWeight(
    qscores_tsv_path="place-pulse-2.0/qscores.tsv",
    split='train',
    transform=transform,
)
val_dataset = PlacePulseDatasetWeight(
    qscores_tsv_path="place-pulse-2.0/qscores.tsv",
    split='val',
    transform=transform,
)

train_dataset.dataframe = train_dataset.dataframe.iloc[:32]
val_dataset.dataframe = val_dataset.dataframe.iloc[:8]

def data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features]).float()
    # Do NOT set requires_grad_ here!
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)

    batch = {
        "pixel_values": pixel_values,
        "labels": labels,
    }

    if "study_type_idx" in features[0]:
        batch["study_type_idx"] = torch.tensor([f["study_type_idx"] for f in features], dtype=torch.long)
    if "location_id" in features[0]:
        batch["location_id"] = [f["location_id"] for f in features]

    print("pixel_values.requires_grad =", pixel_values.requires_grad)
    print("pixel_values.is_leaf =", pixel_values.is_leaf)
    return batch


train_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    dataloader_num_workers=0,
    logging_first_step=True,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    seed=3407,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    data_collator = data_collator,  # <-- your custom collator
    args = train_args,
)

trainer_stats = trainer.train()