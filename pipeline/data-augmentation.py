# This file is meant to generate datasets for data augmentation defenses evaluation
import pandas as pd
from constrained_attacks import datasets
import numpy as np
import configargparse

from sdv.tabular import CTGAN
from copulas.multivariate import GaussianMultivariate

import sys
sys.path.insert(0,'.')

def cut_in_half(data):
    return data[:len(data)//2]

def CTGAN_augmentation(data):
    # data = np.array(data, dtype=object)
    data = pd.DataFrame(data, columns=["a"]*data.shape[1])
    model = CTGAN()
    model.fit(data)
    return model.sample(num_rows=len(data))

def statistical_augmentation(data):
    copula = GaussianMultivariate()
    copula.fit(data)
    return copula.sample(num_rows=len(data))


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add('--dataset', type=str, required=True, choices=["lcld_v2_time", "url", "ctu_13_neris", "malware"], default='url')
    parser.add('--method', type=str, required=True, choices=["cut_in_half", "CTGAN_augmentation", "statistical_augmentation"], default='cut_in_half')

    args = parser.parse_args()


    # load_data
    dataset = datasets.load_dataset(args.dataset)
    x, y = dataset.get_x_y()
    metadata = dataset.get_metadata()
    x = np.array(x)
    data = np.c_[x, y]
    data = np.array(data, dtype=metadata["type"])
    data_df = pd.DataFrame(data, columns=metadata["feature"])

    # generate new data
    if args.method == "cut_in_half":
        new_data = cut_in_half(data)
    elif args.method == "CTGAN_augmentation":
        new_data = CTGAN_augmentation(data_df)
    elif args.method == "statistical_augmentation":
        new_data = statistical_augmentation(data)
    print("Shape of dataset: ", data.shape, "; shape of new dataset: ", new_data.shape)

    # save new data
    new_path = "augmented_datasets/" + args.dataset + "/" + args.dataset + "_" + args.method + ".csv"
    file = open(new_path, "w+")
    np.savetxt(file, new_data, delimiter=",")

    # data evaluation & visualization





