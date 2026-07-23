import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_erosion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

def compute_dice(pred, true):
    pred, true = pred.astype(bool), true.astype(bool)
    u = pred.sum() + true.sum()
    return 1.0 if u == 0 else 2.0 * (pred & true).sum() / u

def compute_hd95(pred, true):
    pred, true = pred.astype(bool), true.astype(bool)
    if pred.sum() == 0 or true.sum() == 0:
        return float('nan')
    ps = pred & ~binary_erosion(pred)
    ts = true & ~binary_erosion(true)
    
    d1 = distance_transform_edt(~true)[ps]
    d2 = distance_transform_edt(~pred)[ts]
    
    if len(d1) == 0 or len(d2) == 0:
        return float('nan')
    return max(np.percentile(d1, 95), np.percentile(d2, 95))

def compute_iou(pred, true):
    pred, true = pred.astype(bool), true.astype(bool)
    intersection = (pred & true).sum()
    union = (pred | true).sum()
    return 1.0 if union == 0 else intersection / union

def compute_precision_seg(pred, true):
    pred, true = pred.astype(bool), true.astype(bool)
    if pred.sum() == 0:
        return 1.0 if true.sum() == 0 else 0.0
    return (pred & true).sum() / pred.sum()

def compute_recall_seg(pred, true):
    pred, true = pred.astype(bool), true.astype(bool)
    if true.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return (pred & true).sum() / true.sum()

def keep_largest_component(mask):
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n+1))
    largest = np.argmax(sizes) + 1
    return labeled == largest

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    valid = ~np.isnan(y_true)
    y_true, y_prob = y_true[valid], y_prob[valid]
    
    if len(y_true) == 0:
        return {}
        
    y_pred = (y_prob >= threshold).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy':    accuracy_score(y_true, y_pred),
        'precision':   precision_score(y_true, y_pred, zero_division=0),
        'recall':      recall_score(y_true, y_pred, zero_division=0),
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'auc':         auc,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'probs': y_prob, 'labels': y_true, 'preds': y_pred
    }

def run_mcnemar_test(y_true, y_pred, majority_class):
    valid = ~np.isnan(y_true)
    y_true, y_pred = y_true[valid].astype(int), y_pred[valid].astype(int)
    
    baseline_preds = np.full_like(y_true, majority_class)
    
    b = int(np.sum((y_pred == y_true) & (baseline_preds != y_true)))
    c = int(np.sum((y_pred != y_true) & (baseline_preds == y_true)))
    
    if b + c == 0:
        return {'applicable': False, 'p_value': float('nan'), 'b': b, 'c': c}
        
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    
    baseline_acc = np.mean(baseline_preds == y_true)
    model_acc = np.mean(y_pred == y_true)
    
    return {
        'applicable': True,
        'p_value': result.pvalue,
        'b': b,
        'c': c,
        'baseline_accuracy': baseline_acc,
        'model_accuracy': model_acc
    }
