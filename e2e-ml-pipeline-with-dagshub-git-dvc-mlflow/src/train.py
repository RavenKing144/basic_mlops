import pandas as pd 
import pickle
import yaml
import os
import ssl 

import mlflow
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split, GridSearchCV

from urllib.parse import urlparse


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RavenKing144/basic_mlops.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "RavenKing144"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "905fe156417772df3fbec14f404715b8f5793122"

parameters_setting = yaml.safe_load(open("params.yaml"))["train"]



def hyperparameterstuning(x_train, y_train, param_grid): 
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1, 
        verbose=2
    )
    grid_search.fit(x_train, y_train)
    return grid_search


def train(data_path, model_path, random_state=42, n_estimators=100, max_depth=3):
    data = pd.read_csv(data_path)
    x = data.drop(["Outcome"], axis=1)
    y = data["Outcome"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        mlflow.set_experiment("e2e-pipeline")
    except mlflow.exceptions.MlflowException:
        mlflow.create_experiment("e2e-pipeline")
        mlflow.set_experiment("e2e-pipeline")
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="random_forest", nested = True) as run:
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
        signature = infer_signature(x_train, y_train)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        grid_search = hyperparameterstuning(x_train, y_train, param_grid)
        
        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        
        y_pred = best_model.predict(x_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        print(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        cr = classification_report(y_test, y_pred)
        mlflow.log_text(str(cr), "classification_report.txt")
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        
        mlflow.set_tag("model", "random_forest")
        
        
        pickle.dump(
            best_model, open(model_path, "wb")
        )
        print("done")
        
        
        
        
if __name__ == "__main__":
    train(
        parameters_setting["data"],
        parameters_setting["model"],
        parameters_setting["random_state"],
        parameters_setting["n_estimators"],
        parameters_setting["max_depth"],
    )