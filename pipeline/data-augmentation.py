# This file is meant to generate datasets for data augmentation defenses evaluation

from constrained_attacks import datasets

my_datasets = ["lcld_v2_time", "ctu_13_neris", "url", "malware"]
data_indicator = 2

# load_data
dataset = datasets.load_dataset(my_datasets[data_indicator])
x, y = dataset.get_x_y()

## augmentations methods
# Cut in half


# Statistical augmentation


# GAN-based augmentation


# saving new dataset

