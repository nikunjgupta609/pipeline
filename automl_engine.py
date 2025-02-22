import h2o
from autosklearn.classification import AutoSklearnClassifier
from tpot import TPOTClassifier
from h2o.automl import H2OAutoML
import pandas as pd
import joblib
import mlflow

from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

class AutoMLPipeline:
    def __init__(self, algorithm="h2o"):
        self.algorithm = algorithm
        self.model = None

    def fit(self, X, y):
        # Preprocess data
        X_processed = preprocess_data(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # Initialize MLflow
        mlflow.set_experiment("AutoML_Tracking")
        with mlflow.start_run():
            if self.algorithm == "h2o":
                h2o.init()
                train_df = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
                self.model = H2OAutoML(max_models=10, seed=1)
                self.model.train(y=y_train.name, training_frame=train_df)
            
            elif self.algorithm == "autosklearn":
                self.model = AutoSklearnClassifier(time_left_for_this_task=600)
                self.model.fit(X_train, y_train)

            elif self.algorithm == "tpot":
                self.model = TPOTClassifier(generations=5, population_size=20, verbosity=2)
                self.model.fit(X_train, y_train)

            # Model evaluation
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Log metrics in MLflow
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            print(f"Model Performance: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")

        return self.model

    def predict(self, X):
        X_processed = preprocess_data(X)
        return self.model.predict(X_processed)

    def save_model(self, filename):
        joblib.dump(self.model, filename)
