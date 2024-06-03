from deepod.metrics import ts_metrics
from deepod.metrics import point_adjustment 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def evaluate_stability(method, X, y, n_splits=10):
    """
    Evaluate the stability of an anomaly detection method using cross-validation.

    Parameters:
    method (object): Anomaly detection method object.
    X (array-like): Input data.
    y (array-like): True labels.
    n_splits (int, optional): Number of splits for cross-validation. Default is 10.

    Returns:
    tuple: A tuple containing the mean and standard deviation of the AUC scores.

    """
    auc_scores = []

    for _ in range(n_splits):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Fit the method on the training data
        method.fit(X_train)

        # Predict the anomalies in the test data
        scores = method.decision_function(X_test)

        # Compute the AUC score
        auc = roc_auc_score(y_test, scores)
        auc_scores.append(auc)

    # Compute the mean and standard deviation of the AUC scores
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    return mean_auc, std_auc

def run_evaluation(y, scores, do_point_adjustment=False):
    """
    y: array-like, true labels
    scores: array-like, predicted scores or probabilities
    point_adjustment: bool, whether to apply point adjustment
    
    Calculate evaluation metrics for binary classification.

    Parameters:
    - y: array-like, true labels
    - scores: array-like, predicted scores or probabilities
    - point_adjustment: bool, whether to apply point adjustment or not

    Returns:
    - result: dict, evaluation metrics
        - AUCROC: float, area under the ROC curve
        - AUCPR: float, area under the precision-recall curve
        - F1: float, F1 score
        - Precision: float, precision
        - Recall: float, recall

    If point_adjustment is True, the following additional metrics will be included in the result:
        - Adjusted AUCROC: float, adjusted area under the ROC curve
        - Adjusted AUCPR: float, adjusted area under the precision-recall curve
        - Adjusted F1: float, adjusted F1 score
        - Adjusted Precision: float, adjusted precision
        - Adjusted Recall: float, adjusted recall
    """
    eval_metrics = ts_metrics(y, scores)
    adj_eval_metrics = ts_metrics(y, point_adjustment(y, scores))
    result = {
        'AUCROC': eval_metrics[0],
        'AUCPR': eval_metrics[1],
        'F1': eval_metrics[2],
        'Precision': eval_metrics[3],
        'Recall': eval_metrics[4]}
    
    if do_point_adjustment:
        result_adjusted = {
            'Adjusted AUCROC': adj_eval_metrics[0],
            'Adjusted AUCPR': adj_eval_metrics[1],
            'Adjusted F1': adj_eval_metrics[2],
            'Adjusted Precision': adj_eval_metrics[3],
            'Adjusted Recall': adj_eval_metrics[4]}
        result = {**result, **result_adjusted}
    return result