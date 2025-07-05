import torch
import random
import numpy as np
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments, Trainer
from data_prep_weights import PlacePulseDatasetWeight
from torchvision import transforms
from trl import SFTTrainer, SFTConfig

# 1. Set random seed for reproducibility
def set_seed(seed=3407):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(3407)

# 2. Define image transforms
transform = transforms.Compose([
    transforms.Resize((336, 336)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load model and tokenizer
model_name = "unsloth/llava-1.5-7b-hf-bnb-4bit"
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    use_gradient_checkpointing=True,
    device_map="auto"
)
FastVisionModel.for_training(model)

# 4. Add LoRA/PEFT
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

# 5. Load datasets
train_dataset = PlacePulseDatasetWeight(
    qscores_tsv_path="place-pulse-2.0/qscores.tsv",
    split='train',
)
val_dataset = PlacePulseDatasetWeight(
    qscores_tsv_path="place-pulse-2.0/qscores.tsv",
    split='val',
)

# Optional subsample for quick testing
train_dataset.dataframe = train_dataset.dataframe.iloc[:32]
val_dataset.dataframe = val_dataset.dataframe.iloc[:8]

def data_collator(features):
    pixel_values = torch.stack([transform(f["pixel_values"]) for f in features])
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)

    # Tokenize the text field (single word or phrase per sample)
    texts = [f["text"] for f in features]
    text_tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    batch = {
        "images": pixel_values,  # <-- use 'images' instead of 'pixel_values'
        "input_ids": text_tokens["input_ids"],
        "attention_mask": text_tokens["attention_mask"],
        "labels": labels,
    }

    if "study_type_idx" in features[0]:
        batch["study_type_idx"] = torch.tensor([f["study_type_idx"] for f in features], dtype=torch.long)
    if "location_id" in features[0]:
        batch["location_id"] = [f["location_id"] for f in features]

    print("texts:", texts)
    print("input_ids shape:", text_tokens["input_ids"].shape)
    print("input_ids:", text_tokens["input_ids"])

    return batch

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 9. Initialize trainer
trainer = RegressionTrainer(
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",

        # Required for vision inputs
        remove_unused_columns=False,


        # The following are NOT necessary unless you are generating text
        # max_seq_length = 2048,
        # dataset_text_field = "",
    ),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# 10. Train
trainer_stats = trainer.train()


