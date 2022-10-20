# This file is meant to generate datasets for data augmentation defenses evaluation

from constrained_attacks import datasets
import numpy as np
import pandas as pd

my_datasets = ["lcld_v2_time", "ctu_13_neris", "url", "malware"]
data_indicator = 2

# load_data
dataset = datasets.load_dataset(my_datasets[data_indicator])
x, y = dataset.get_x_y()
x = np.array(x)
data = np.c_[x,y]

## augmentations methods
# Cut in half
cut_data = data[:len(data)//2]

# GAN-based augmentation
from sdv.tabular import CTGAN

model = CTGAN()
model.fit(data)

GAN_data = model.sample(num_rows=len(data))

print("done")

# Statistical augmentation
from copulas.multivariate import GaussianMultivariate

copula = GaussianMultivariate()
copula.fit(data)
synthetic_data = copula.sample(num_rows=len(data))

print("donea")


# saving new dataset

