import mlflow

def create_mlflow_experiment(experiment_name:str, artifact_location: str, tags:dict[str,any]) -> str:
    """Create a new mlflow experiment with the given name and artifact location.

    Args:
        experiment_name (str): The name of the experiment
        artifact_location (str): The artifact location of the experiment to create
        tags (dict[str,any]): The tags of experiment to create.

    Returns:
        str: The id of the created experiment
    """    
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    return experiment_id

def get_mlflow_experiment(experiment_id:str=None, experiment_name:str=None) -> mlflow.entities.Experiment:
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provide")
    return experiment