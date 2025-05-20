from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    roc_curve,
)

def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate (EER)

    Args:
        y_true: Ground truth labels (1 for spoof, 0 for bonafide)
        y_score: Predicted scores/probabilities (higher score means more likely to be spoof)

    Returns:
        EER: Equal Error Rate value
        threshold: The threshold at which FAR = FRR
    """
    # Get false positive rate (FPR) and true positive rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # False negative rate (FNR) is 1 - TPR
    fnr = 1 - tpr

    # Find the intersection point where FPR = FNR
    # The EER is the point where the false acceptance rate equals the false rejection rate
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    # Find the threshold where FPR = FNR
    threshold = interp1d(fpr, thresholds)(eer)

    return eer, threshold
