from sklearn.metrics import accuracy_score

def compute_pairwise_accuracy(eval_pred):
    # eval_pred.predictions: [N,] predicted scores
    # eval_pred.label_ids: [N,] true weights (+1, -1, or 0.5)
    # You need to reconstruct pairs from your eval dataset!
    # This assumes your eval_dataset returns pairs in order: left, right, left, right, ...
    preds = eval_pred.predictions.squeeze()
    labels = eval_pred.label_ids

    correct = 0
    total = 0
    for i in range(0, len(labels), 2):
        left_score = preds[i]
        right_score = preds[i+1]
        left_label = labels[i]
        right_label = labels[i+1]
        # Determine ground truth winner
        if left_label > right_label:
            gt = "left"
        elif right_label > left_label:
            gt = "right"
        else:
            gt = "equal"
        # Model's predicted winner
        if left_score > right_score:
            pred = "left"
        elif right_score > left_score:
            pred = "right"
        else:
            pred = "equal"
        if pred == gt:
            correct += 1
        total += 1
    return {"pairwise_accuracy": correct / total if total > 0 else 0}