import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop_duplicates()
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def feature_engineering(df):
    df["Hillshade_Diff"] = df["Hillshade_Noon"] - df["Hillshade_9am"]
    return df

def prepare_features(df):
    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]

    categorical_cols = ["Soil_Type", "Wilderness_Area"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

   # smote = SMOTE(random_state=42)
   # X_res, y_res = smote.fit_resample(X_processed, y)
    X_res, y_res = X_processed, y
    joblib.dump(preprocessor, "preprocessor.pkl")

    return X_res, y_res
