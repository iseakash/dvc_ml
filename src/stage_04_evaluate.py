
from numpy.core.fromnumeric import transpose
from numpy.lib.shape_base import split
from scipy.sparse.construct import random
from utils.all_utils import read_yaml, create_directory, save_reports
import argparse
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mse = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mse, r2

def evaluate(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # save dataset in the local directory
    # create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]['artifacts_dir']
    split_data_dir = config["artifacts"]["split_data_dir"]
    test_data_filename = config["artifacts"]["test"]
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)
    test_data = pd.read_csv(test_data_path)

    test_x = test_data.drop("quality", axis= 1)
    test_y = test_data["quality"]

    model_dir = config["artifacts"]["model_dir"]
    model_filename = config["artifacts"]["model_filename"]
    model_path = os.path.join(artifacts_dir, model_dir, model_filename)
    
    lr = joblib.load(model_path)

    predicted_value = lr.predict(test_x)
    rmse, mse, r2 = evaluate_metrics(test_y, predicted_value)

    scores_dir = config["artifacts"]["reports_dir"]
    scores_filename = config["artifacts"]["scores"]

    scores_dir_path = os.path.join(artifacts_dir, scores_dir)
    create_directory([scores_dir_path])

    scores_filepath = os.path.join(scores_dir_path, scores_filename)

    scores = {
        "rmse": rmse,
        "mse": mse,
        "r2": r2
    }
    save_reports(report = scores, report_path = scores_filepath)

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    evaluate(config_path=parsed_args.config, params_path= parsed_args.params)
