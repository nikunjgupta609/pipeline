import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns

    # Imputation for missing values
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Scaling and encoding
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")

    # Pipeline for numerical and categorical features
    num_pipeline = Pipeline(steps=[("imputer", num_imputer), ("scaler", scaler)])
    cat_pipeline = Pipeline(steps=[("imputer", cat_imputer), ("encoder", encoder)])

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )

    df_processed = preprocessor.fit_transform(df)
    
    return df_processed
