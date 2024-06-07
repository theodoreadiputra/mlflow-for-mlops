from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_data() -> pd.DataFrame():
    """Downlaod file california housingription dataset

    Returns:
        california housingription dataset
    """

    cur_directory_path = os.path.abspath(os.path.dirname(__file__))
    data = fetch_california_housing(
        data_home=f"{cur_directory_path}/data/", as_frame=True, download_if_missing=True
    )
    return data.frame

def get_feature_dataframe() -> pd.DataFrame:
    df = load_data()
    df["id"] = df.index
    df["target"] = df["MedHouseVal"] >= df["MedHouseVal"].median()
    df["target"] = df["target"].astype(int)

    return df

