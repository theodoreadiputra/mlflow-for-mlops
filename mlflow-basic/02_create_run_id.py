import mlflow

from mlflow_utils import create_mlflow_experiment

if __name__=="__main__":

    experiment_id = create_mlflow_experiment(
        experiment_name="testing_mlflow",
        artifact_location="testing_mldlow1_artifacts",
        tags={"env": "dev", "version": "1.0.0"}
    )

    with mlflow.start_run(run_name="testing", experiment_id= experiment_id) as run:

        mlflow.log_param("learning_rate",0.01)
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("Status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))

