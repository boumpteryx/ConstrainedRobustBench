# This file is meant to generate datasets for data augmentation defenses evaluation
import pandas as pd
import numpy as np
import configargparse

from sdv.tabular import CTGAN, GaussianCopula
from sdv.constraints import create_custom_constraint
from imblearn.over_sampling import SMOTE

import sys

sys.path.insert(0, ".")

from constrained_attacks import datasets
from constraints.constraints_checker import ConstraintChecker


def constr_check(column_names, data, data_name): # do not remove column_names
    dataset = datasets.load_dataset(data_name)
    constraints2 = dataset.get_constraints()
    checker = ConstraintChecker(constraints2, tolerance=0.01)
    check = checker.check_constraints(data.to_numpy(), data.to_numpy())
    return pd.Series(check)


def cut_in_half(data):
    return data[: len(data) // 2]


def cut_in_tenth(data):
    return data[: len(data) // 10]


def CTGAN_augmentation(data, constraints=None):
    if constraints is not None:
        model = CTGAN(verbose=True, constraints=constraints,cuda=True)
    else:
        model = CTGAN(verbose=True,cuda=True)
    model.fit(data)
    return model.sample(num_rows=len(data)).to_numpy()


def copulas_augmentation(data, constraints=None):
    if constraints is not None:
        model = GaussianCopula(constraints=constraints)
    else:
        model = GaussianCopula()
    model.fit(data)
    return model.sample(num_rows=len(data)).to_numpy()


def SMOTE_augmentation(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return np.c_[X_res, y_res]


def save(data, args, old_data=None):
    if old_data is not None:
        data = np.concatenate((old_data, data))
    new_path = (
        "./pipeline/augmented_datasets/"
        + args.dataset
        + "/"
        + args.dataset
        + "_"
        + args.method
        + "_use_constraints_"
        + str(bool(args.use_constraints))
        + ".csv"
    )
    with open(new_path, "w+") as file:
        np.savetxt(file, data, delimiter=",")
    file.close()


def checker_is_valid(data, data_name):
    dataset = datasets.load_dataset(data_name)
    constraints2 = dataset.get_constraints()
    checker = ConstraintChecker(constraints2, tolerance=0.01)
    check = checker.check_constraints(data, data)
    print((len(check) - np.sum(check)) / len(check), "% of invalid data")
    return check


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add(
        "--dataset",
        type=str,
        required=True,
        choices=["lcld_v2_time", "url", "ctu_13_neris", "malware"],
        default="url",
    )
    parser.add(
        "--method",
        type=str,
        required=True,
        choices=[
            "cut_in_half",
            "cut_in_tenth",
            "CTGAN_augmentation",
            "copulas_augmentation",
            "SMOTE_augmentation",
            "all",
        ],
        default="cut_in_half",
    )
    parser.add("--use_constraints", type=int, required=False, default=0)


    args = parser.parse_args()
    print(
        "dataset = ",
        args.dataset,
        " method = ",
        args.method,
        "using constraints: ",
        bool(args.use_constraints),
    )

    # load_data
    dataset = datasets.load_dataset(args.dataset)
    x, y = dataset.get_x_y()
    metadata = dataset.get_metadata()
    splits = dataset.get_splits()
    x = np.array(x.iloc[splits["train"]])
    y = np.array(y[splits["train"]])
    data = np.c_[x, y]  # [:100]
    data = np.array(data, dtype=metadata["type"])

    constraints = None
    if args.use_constraints == 1:
        custom_Constraints = create_custom_constraint(is_valid_fn=constr_check)
        constraints = [custom_Constraints(column_names=[], data_name=args.dataset)]
        is_valid = checker_is_valid(data, args.dataset)
        data = np.array(
            [data[i] for i in range(len(data)) if is_valid[i]], dtype=metadata["type"]
        )
        print("invalid rows removed")

    data_df = pd.DataFrame(data, columns=[str(i) for i in range(len(data[0]))])
    data_df.astype("object")

    # generate new data
    if args.method == "cut_in_half":
        print(" using = ", args.method, "...")
        new_data = cut_in_half(data)
    if args.method == "cut_in_tenth":
        print(" using = ", args.method, "...")
        new_data = cut_in_tenth(data)
    elif args.method == "CTGAN_augmentation":
        print(" using = ", args.method, "...")
        new_data = CTGAN_augmentation(data_df, constraints=constraints)
    elif args.method == "copulas_augmentation":
        print(" using = ", args.method, "...")
        new_data = copulas_augmentation(data_df, constraints=constraints)
    elif args.method == "SMOTE_augmentation":
        print(" using = ", args.method, "...")
        new_data = SMOTE_augmentation(x, y)
    if args.method != "all":
        print(
            "Shape of dataset: ", data.shape, "; shape of new dataset: ", new_data.shape
        )
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
        save(
            copulas_augmentation(data_df, constraints=constraints), args, old_data=data
        )
        checker_is_valid(
            copulas_augmentation(data_df, constraints=constraints), args.dataset
        )  # does it really work? Let's test it out!
        args.method = "SMOTE_augmentation"
        print(" using = ", args.method, "...")
        save(SMOTE_augmentation(x, y), args, old_data=data)
        args.method = "CTGAN_augmentation"
        print(" using = ", args.method, "...")
        save(CTGAN_augmentation(data_df, constraints=constraints), args, old_data=data)
