import torch
import pandas as pd
import os

results_path = "./results"
all_results = []

for result_folder in os.listdir(results_path):
    model_name = result_folder.split("/")[-1]
    best = torch.load(results_path + "/" + result_folder + "/best_model.t7")
    all_results.append([model_name, best["best_test_acc"],
                       best["best_test_acc_epoch"]])

df = pd.DataFrame(all_results, columns=[
                  'model', 'best_test_acc', "best_test_acc_epoch"])

print(df.to_string())
