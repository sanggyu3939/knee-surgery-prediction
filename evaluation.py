from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1]

    return {
        "AUROC": roc_auc_score(y, prob),
        "AUPRC": average_precision_score(y, prob),
        "Brier": brier_score_loss(y, prob)
    }
