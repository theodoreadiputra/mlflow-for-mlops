from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from typing import List

def get_pipeline(
        numerical_features: List[str], 
        categorical_features: List[str]
):
    preprocessing = ColumnTransformer(
        [
            ("numerical", SimpleImputer(strategy="median"), numerical_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("decomposition", PCA(n_components=2), numerical_features+categorical_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", RandomForestClassifier())
        ]
    )

    return pipeline