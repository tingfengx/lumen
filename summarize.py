"""
lumen/summarize.py
Author: Tingfeng Xia

Summarization helper module, aiding performance analysis. 
"""

import torch
import pandas as pd
import os


def summarize_all_results(results_path="./results"):
    all_results = []

    for result_folder in os.listdir(results_path):
        model_name = result_folder.split("/")[-1]
        p_ = results_path + "/" + result_folder + "/best_model.t7"
        if os.path.exists(p_):
            best = torch.load(p_)
            all_results.append([model_name, best["best_test_acc"].cpu().numpy(),
                                best["best_test_acc_epoch"]])

    df = pd.DataFrame(all_results, columns=[
        'model', 'best_test_acc', "best_test_acc_epoch"])
    df = df.sort_values('best_test_acc', ascending=False)

    print("====== SUMMARIZE RESULTS (ALL RESULTS) ======")
    print(df.to_string())
    return df


def summarize_each_model(results_path="./results"):
    pass


summarize_all_results()
