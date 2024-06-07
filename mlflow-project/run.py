from package.feature.data_processing import get_feature_dataframe
from package.ml_training.preprocessing_pipeline import get_pipeline
from package.ml_training.retrieval import get_train_test_score_set, get_train_test_set
from package.ml_training.train import train_model

from package.utils.utils import set_or_create_experiment, get_performance_plot, get_classification_metrics
from typing import Dict, List, Optional

from hyperopt import fmin, tpe, Trials, hp
from functools import partial

import mlflow
import pandas as pd

def objective_function(
        params :Dict,
        x_train:pd.DataFrame,
        x_test:pd.DataFrame,
        y_train:pd.DataFrame,
        y_test:pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str]
)-> float:
    pipeline = get_pipeline(numerical_features=numerical_features, categorical_features=categorical_features)
    params.update({"model__max_depth":int(params["model__max_depth"])})
    params.update({"model__n_estimators":int(params["model__n_estimators"])})
    pipeline.set_params(**params)  

    with mlflow.start_run(nested=True) as run:
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics = get_classification_metrics(
            y_true=y_test, y_pred=y_pred, prefix="test"
        )

        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
    return -metrics["test_f1_score"]

if __name__ == '__main__':
    experiment_id = set_or_create_experiment(experiment_name="learning-mlflow")
   
    df = get_feature_dataframe()
    x_train, x_test, y_train, y_test = get_train_test_set(df.drop(["MedHouseVal","id"], axis = 1))

    numerical_features = [f for f in x_train.columns if f not in ["target"]]

    space = {
        "model__n_estimators": hp.quniform("model__n_estimators", 20, 200, 10),
        "model__max_depth": hp.quniform("model__max_depth", 10, 100, 10),
    }
    print(experiment_id)
    with mlflow.start_run(run_name="hyperparameter_optimization", experiment_id=experiment_id) as run:
        best_params = fmin(
            fn=partial(
                objective_function,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                numerical_features=numerical_features,
                categorical_features=[],
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials(),
        )

        pipeline = get_pipeline(numerical_features=numerical_features, categorical_features=[])

        best_params.update({"model__max_depth":int(best_params["model__max_depth"])})
        best_params.update({"model__n_estimators":int(best_params["model__n_estimators"])})
        pipeline.set_params(**best_params) 

        # run_id, pipeline = train_model(pipeline=pipeline, run_name=run_name, model_name=model_name, x=x_train, y = y_train)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics = get_classification_metrics(
            y_true=y_test, y_pred=y_pred, prefix="best_model"
        )

        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
       
        performance_plot = get_performance_plot(y_true=y_test, y_pred=y_pred, prefix="test")
        for plot_name, fig in performance_plot.items():
            mlflow.log_figure(fig, str(plot_name)+".png")

        classification_metrics = get_classification_metrics(y_true=y_test, y_pred=y_pred, prefix="test")
        mlflow.log_metrics(classification_metrics)

        mlflow.autolog()
        mlflow.sklearn.log_model(pipeline, f"{run.info.run_id}-model")
        
        

# if __name__ == '__main__':
#     experiment_name = "house_pricing_classifier"
#     run_name = "training_classifier"
#     model_name = "registered_model"
#     df = get_feature_dataframe()
#     params = Dict

#     x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)

#     features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

#     pipeline = get_pipeline(numerical_features=features, categorical_features=[])

#     experiment_id = set_or_create_experiment(experiment_name=experiment_name)

    # run_id, model = train_model(pipeline=pipeline, run_name=run_name, model_name=model_name, x=x_train[features], y = y_train)

#     y_pred = model.predict(x_test)

#     classification_metrics = get_classification_metrics(y_true=y_test, y_pred=y_pred, prefix="test")

#     performance_plot = get_performance_plot(y_true=y_test, y_pred=y_pred, prefix="test")

#     with mlflow.start_run(run_id=run_id):

#         mlflow.log_metrics(classification_metrics)

#         mlflow.log_params(model[-1].get_params())

#         mlflow.set_tags({"type":"classifier"})

#         mlflow.set_tag("mlflow.note.content", "This is a classifier for the house pricing dataset")

#         for plot_name, fig in performance_plot.items():
#             mlflow.log_figure(fig, str(plot_name)+".png")

