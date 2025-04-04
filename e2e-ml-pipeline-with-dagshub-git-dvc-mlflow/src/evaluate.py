import pandas as pd 
import pickle
import os
import yaml
import mlflow
from sklearn.metrics import accuracy_score


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RavenKing144/basic_mlops.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "RavenKing144"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "905fe156417772df3fbec14f404715b8f5793122"


parameters_setting = yaml.safe_load(open("params.yaml"))["train"]



def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    x = data.drop(["Outcome"], axis=1)
    y = data["Outcome"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    model = pickle.load(open(model_path, "rb"))
    
    predictions = model.predict(x)
    
    accuracy = accuracy_score(y, predictions)
    
    mlflow.log_metric("accuracy", accuracy)
    
    print(accuracy)
    
    
if __name__ == "__main__":
    evaluate(parameters_setting["data"], parameters_setting["model"])
