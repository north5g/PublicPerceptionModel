from data_prep.py import prepare_dataset
from unsloth import FastVisionModel
from transformers import AutoImageProcessor
import torch
from dataset import PlacePulseDataset
from torchvision import transforms
from transformers import TrainingArguments, Trainer

# Example: Compose transforms if needed
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ...add normalization if needed...
])

converted_dataset = prepare_dataset(your_raw_data)  # TODO: Replace with actual raw data loading
train_dataset = PlacePulseDataset(converted_dataset, transform=transform)


# Replace with HuggingFace model name. Majority of file same.
model_name = "geolocal/StreetCLIP"

# Load the tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = True,  # Reduces memory usage. False for 16bit LoRA.
    use_gradient_checkpointing = True,  # True or "unsloth" for gradient checkpointing
    device_map = "auto" # check if necessary
)

image_processor = AutoImageProcessor.from_pretrained(model_name)
transform = image_processor  # If using transformers' processor

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = False, # Do not need language layers for vision tasks
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",  # or "steps" if you have a val set
    remove_unused_columns=False,  # Important for vision models
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Your validation set in pairwise format
    tokenizer=None,
    data_collator=None,
    compute_metrics=compute_pairwise_accuracy,
)

trainer.train()

