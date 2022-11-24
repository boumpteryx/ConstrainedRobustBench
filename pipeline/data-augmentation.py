# This file is meant to generate datasets for data augmentation defenses evaluation
import pandas as pd
from constrained_attacks import datasets
import numpy as np
import configargparse
import torch

from sdv.tabular import CTGAN, GaussianCopula
from sdv.constraints import Inequality, create_custom_constraint
from copulas.multivariate import GaussianMultivariate
from imblearn.over_sampling import SMOTE

import sys
sys.path.insert(0,'.')

from constrained_attacks import datasets
from constraints.constraints_checker import ConstraintChecker

def url_constr_check(column_names, data):
    dataset = datasets.load_dataset("url")
    constraints2 = dataset.get_constraints()
    checker = ConstraintChecker(constraints2, tolerance=0.001)
    check = checker.check_constraints(data, data, pt=True)
    print(check)
    return pd.Series(check)

def url_constraints(column_names, data, metadata):
    def apply_if_a_supp_zero_than_b_supp_zero(a,b):
        a = metadata["feature"][a]
        b = metadata["feature"][b]
        return (0 <= data[a]) or (0 <= data[b])
    def g_2():
        int_sum = 0
        for i in range(3, 18):
            int_sum += data[i]
        return  int_sum + 3 * data[19] <= 0
    g1 = Inequality(low_column_name=metadata["feature"][1],high_column_name=metadata["feature"][0], strict_boundaries=False)
    g2 = g_2()
    g3 = apply_if_a_supp_zero_than_b_supp_zero(21,3)
    g4 = apply_if_a_supp_zero_than_b_supp_zero(23,13)
    def g_5():
        return 3 * data[20] + 4 * data[21] + 2 * data[23] <= 0
    g5 = g_5()
    g6 = apply_if_a_supp_zero_than_b_supp_zero(19,25)
    g8 = apply_if_a_supp_zero_than_b_supp_zero(2,25)
    g10 = apply_if_a_supp_zero_than_b_supp_zero(28,25)
    g11 = apply_if_a_supp_zero_than_b_supp_zero(31,26)
    g12 = Inequality(low_column_name=metadata["feature"][38],high_column_name=metadata["feature"][37], strict_boundaries=False)
    def product_then_sum(a,b,c,d):
        return (a * data[b]) <= (data[c] + d)
    g13 = product_then_sum(3,20,0,1)
    g14 = product_then_sum(4,21,0,1)
    g15 = product_then_sum(4,2,0,1)
    g16 = product_then_sum(2,23,0,1)
    return [g1,g2,g3,g4,g5,g6,g8,g10,g11,g12,g13,g14,g15,g16]

def cut_in_half(data):
    return data[:len(data)//2]

def cut_in_tenth(data):
    return data[:len(data) // 10]

def CTGAN_augmentation(data, constraints=None):
    if constraints is not None:
        model = CTGAN(verbose=True,constraints=constraints)
    else:
        model = CTGAN(verbose=True)
    model.fit(data)
    return model.sample(num_rows=len(data)).to_numpy()

def copulas_augmentation(data, constraints=None):
    if constraints is not None:
        model = GaussianCopula(constraints=constraints) # GaussianMultivariate()
    else:
        model = GaussianCopula()
    model.fit(data)
    return model.sample(num_rows=len(data)).to_numpy()

def SMOTE_augmentation(X,y):
    sm = SMOTE(random_state=42)
    X_res,y_res = sm.fit_resample(X, y)
    return np.c_[X_res, y_res]
def save(data, args, old_data = None):
    if old_data is not None:
        data = np.concatenate((old_data,data))
    new_path = "augmented_datasets/" + args.dataset + "/" + args.dataset + "_" + args.method + ".csv"
    file = open(new_path, "w+")
    np.savetxt(file, data, delimiter=",")
    file.close()

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add('--dataset', type=str, required=True, choices=["lcld_v2_time", "url", "ctu_13_neris", "malware"], default='url')
    parser.add('--method', type=str, required=True, choices=["cut_in_half", "cut_in_tenth", "CTGAN_augmentation", "copulas_augmentation", "SMOTE_augmentation", "all"], default='cut_in_half')
    parser.add('--use_constraints', type=int, required=False, default=0)

    args = parser.parse_args()
    print("dataset = ", args.dataset, " method = ", args.method, "using constraints: ", bool(args.use_constraints))

    # load_data
    dataset = datasets.load_dataset(args.dataset)
    x, y = dataset.get_x_y()
    metadata = dataset.get_metadata()
    splits = dataset.get_splits()
    x = np.array(x.iloc[splits["train"]])
    y = np.array(y[splits["train"]])
    data = np.c_[x, y]
    data = np.array(data, dtype=metadata["type"])
    data_df = pd.DataFrame(data, columns=metadata["feature"])
    data_df.astype('object')

    constraints = None
    if args.use_constraints == 1:
        # UrlConstraints = create_custom_constraint(is_valid_fn=url_constraints)
        UrlConstraints = create_custom_constraint(is_valid_fn=url_constr_check)
        constraints = [UrlConstraints(column_names=[])]

    # generate new data
    if args.method == "cut_in_half":
        new_data = cut_in_half(data)
    if args.method == "cut_in_tenth":
        new_data = cut_in_tenth(data)
    elif args.method == "CTGAN_augmentation":
        new_data = CTGAN_augmentation(data_df, constraints=constraints)
    elif args.method == "copulas_augmentation":
        new_data = copulas_augmentation(data, constraints=constraints)
    elif args.method == "SMOTE_augmentation":
        new_data = SMOTE_augmentation(x,y)
    if args.method != "all":
        print("Shape of dataset: ", data.shape, "; shape of new dataset: ", new_data.shape)
        save(new_data, args)

    if args.method == "all":
        args.method = "cut_in_half"
        print(" using = ", args.method, "...")
        save(cut_in_half(data), args)
        args.method = "cut_in_tenth"
        print(" using = ", args.method, "...")
        save(cut_in_tenth(data), args)
        args.method = "copulas_augmentation"
        print(" using = ", args.method, "...")
        save(copulas_augmentation(data_df, constraints=constraints), args, old_data=data)
        args.method = "SMOTE_augmentation"
        print(" using = ", args.method, "...")
        save(SMOTE_augmentation(x,y), args, old_data=data)
        args.method = "CTGAN_augmentation"
        print(" using = ", args.method, "...")
        save(CTGAN_augmentation(data_df, constraints=constraints), args, old_data=data)


    # data evaluation & visualization





