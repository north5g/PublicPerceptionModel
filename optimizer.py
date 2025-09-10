import math
import torch
from transformers import get_cosine_schedule_with_warmup

def create_optimizer(model, training_args, train_dataset):
    """
    Create an AdamW optimizer with separate parameter groups for the model's head and base parameters,
    along with a cosine learning rate scheduler with warmup.

    Args:
        model: The model to optimize.
        training_args: Training arguments containing hyperparameters.
        train_dataset: The training dataset to compute the number of steps.

    Returns:
        optimizer: The configured AdamW optimizer.
        lr_scheduler: The learning rate scheduler.
    """
    # --- 0) tune these hyperparams (recommendations) ---
    HEAD_LR = 3e-4            # start here for head; try 1e-3 if stable
    ENCODER_LR = 5e-6         # if you unfreeze encoder blocks; otherwise not used
    WEIGHT_DECAY = 0.01
    HEAD_WEIGHT_DECAY = 0.0   # often better to not WD the head heavily
    BETA = (0.9, 0.999)
    EPS = 1e-8

    # --- 1) compute number of training steps (matches Trainer logic) ---
    train_batch_size = training_args.per_device_train_batch_size
    accum = training_args.gradient_accumulation_steps
    # steps per epoch equals ceil(len(dataset) / (batch_size * accum))
    steps_per_epoch = math.ceil(len(train_dataset) / (train_batch_size * accum))
    num_training_steps = steps_per_epoch * int(training_args.num_train_epochs)
    num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)

    # --- 2) build param groups (head vs rest) ---
    # choose the names that identify your head modules
    head_keywords = ["regressor", "img_proj", "study_proj", "film", "study_embedding", "type_bias", "type_scale"]

    head_params = []
    base_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in head_keywords):
            head_params.append(p)
        else:
            base_params.append(p)

    # If encoder was frozen, base_params may be identical to head_params (or empty) — that's OK.
    optimizer_grouped_parameters = []
    if head_params:
        optimizer_grouped_parameters.append({"params": head_params, "lr": HEAD_LR, "weight_decay": HEAD_WEIGHT_DECAY})
    if base_params:
        optimizer_grouped_parameters.append({"params": base_params, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY})

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=BETA, eps=EPS)

    # --- 3) scheduler (cosine with warmup) ---
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)

    # --- 4) (OPTIONAL) update training_args: be explicit about grad clipping and remove conflicting scheduler settings ---
    # training_args.lr_scheduler_type and warmup_ratio are ignored when you pass your own optimizers,
    # but setting max_grad_norm is still useful.
    training_args.max_grad_norm = 1.0

    return optimizer, lr_scheduler
