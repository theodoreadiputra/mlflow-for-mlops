import mlflow
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from typing import Dict

import matplotlib.pyplot as plt
def set_or_create_experiment(experiment_name)->str:
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id

def get_performance_plot(
        y_true:pd.DataFrame, 
        y_pred:pd.DataFrame, prefix:str
    )->Dict[str, any]:
    roc_figure = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    cm_figure = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    pr_figure = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    return {
        f"{prefix}_roc_curve":roc_figure,
        f"{prefix}_confusion_matrix":cm_figure,
        f"{prefix}_precision_recall_curve":pr_figure,
    }

def get_classification_metrics(
        y_true:pd.DataFrame, 
        y_pred:pd.DataFrame,
        prefix:str 
    )->Dict[str, any]:
    metrics = {
        f"{prefix}_accuracy_score":accuracy_score(y_true, y_pred),
        f"{prefix}_precision_score":precision_score(y_true, y_pred),
        f"{prefix}_recall_score":recall_score(y_true, y_pred),
        f"{prefix}_f1_score":f1_score(y_true, y_pred),
        f"{prefix}_roc_auc_score":roc_auc_score(y_true, y_pred),
    }

    return metrics