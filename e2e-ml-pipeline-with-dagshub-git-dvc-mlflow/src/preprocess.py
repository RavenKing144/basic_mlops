import pandas as pd
import os
import yaml


params_setings = yaml.safe_load(open("params.yaml"))
preprocess_settings = params_setings["preprocess"]


def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )
    data.to_csv(output_path, index=False)
    print("Preprocess data done.")


if __name__ == "__main__":
    input_path = preprocess_settings["input"]
    output_path = preprocess_settings["output"]
    preprocess_data(input_path, output_path)
