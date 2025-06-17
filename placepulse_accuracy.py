from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F


def compute_placepulse_pairwise_accuracy(eval_pred, eval_dataset, labels_dict, device="cpu"):
    """
    Computes pairwise accuracy for PlacePulse-style evaluation.
    For each pair, compares image embeddings to the positive label embedding.
    """
    preds = eval_pred.predictions
    # preds shape: (N, D) where D is embedding dim
    correct = 0
    total = 0

    # Assumes eval_dataset returns pairs in order: left, right, left, right, ...
    for i in range(0, len(eval_dataset), 2):
        left_item = eval_dataset.data[i]
        right_item = eval_dataset.data[i+1]
        study = left_item.get('study', None)
        if study is None or study not in labels_dict:
            continue
        pos_label = labels_dict[study][0]
        # Get positive label embedding from dataset
        pos_emb = eval_dataset.label_embeddings[pos_label].to(device)
        left_emb = torch.tensor(preds[i]).to(device)
        right_emb = torch.tensor(preds[i+1]).to(device)
        # Cosine similarity to positive label
        left_sim = F.cosine_similarity(left_emb, pos_emb, dim=0)
        right_sim = F.cosine_similarity(right_emb, pos_emb, dim=0)
        # Model's predicted winner
        if left_sim > right_sim:
            pred = "left"
        elif right_sim > left_sim:
            pred = "right"
        else:
            pred = "equal"
        # Ground truth
        gt = left_item.get('selected', None)
        if pred == gt:
            correct += 1
        total += 1
    return {"pairwise_accuracy": correct / total if total > 0 else 0}