"""
Evaluation of non-DL algorithms on perturbed inputs
"""
import sklearn
import torch
from constrained_attacks import datasets
import scikit-learn as sklearn

my_datasets = ["lcld_v2_time", "ctu_13_neris", "url", "malware"]
data_indicator = 2

# load_data
dataset = datasets.load_dataset(my_datasets[data_indicator])
x, y = dataset.get_x_y()
x_adv = torch.load()
splits = dataset.get_splits()
y_test = y[splits["test"]]

models = [sklearn.svm] # non-DL models

model = sklearn.svm()


print("model : ", model, " dataset : ", dataset, " attack : ", " score : ")



