import mlflow

if __name__ == "__main__":
    mlflow.create_experiment(
        name="testing_mlflow",
        artifact_location="testing_mlflow_artifacts",
        tags={
            "env": "dev",
            "version":"1.0.0"
        },
    )