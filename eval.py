import torch
import os
import sys

import numpy as np

from utils.load_data import load_data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.insert(0, 'autoattack/examples')
from constrained_attacks.constraints.relation_constraint import Constant
from constrained_attacks.constraints.relation_constraint import LessEqualConstraint, Feature

from utils.models import init_model

sys.path.insert(0, 'autoattack')

from resnet import *

import configargparse
import yaml

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=16./255.)
    parser.add_argument('--epsilon_std', type=int, default=0)
    parser.add_argument('--model', type=str, default='./tests/resources/pytorch_models/url_torch.pth')
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='custom')
    parser.add_argument('--model_name', type=str, default='Net',help="Use '#' to use multiple models in an ensemble")
    parser.add_argument('--use_constraints', type=int, default=1)
    parser.add_argument('--all_models', type=int, default=0)
    parser.add_argument('--transfer_from', type=str, default=None)

    parser.add_argument('--api_key', type=str, default="")

    parser.add('--config', type=str,  is_config_file_arg=True, default='config/url.yml')
    # parser.add('--model_name', required=True, help="Name of the model that should be trained")
    parser.add('--dataset', required=True, help="Name of the dataset that will be used", default='url')
    parser.add('--objective', required=True, type=str, default="binary", choices=["regression", "classification",
                                                                                      "binary"],
               help="Set the type of the task")

    parser.add('--use_gpu', action="store_true", help="Set to true if GPU is available")
    parser.add('--gpu_ids', type=int, action="append", help="IDs of the GPUs used when data_parallel is true")
    parser.add('--data_parallel', action="store_true", help="Distribute the training over multiple GPUs")

    parser.add('--optimize_hyperparameters', action="store_true",
               help="Search for the best hyperparameters")
    parser.add('--n_trials', type=int, default=100, help="Number of trials for the hyperparameter optimization")
    parser.add('--direction', type=str, default="minimize", choices=['minimize', 'maximize'],
               help="Direction of optimization.")

    parser.add('--num_splits', type=int, default=5, help="Number of splits done for cross validation")
    parser.add('--shuffle', action="store_true", help="Shuffle data during cross-validation")
    parser.add('--seed', type=int, default=123, help="Seed for KFold initialization.")

    parser.add('--scale', action="store_true", help="Normalize input data.")
    parser.add('--target_encode', action="store_true", help="Encode the targets that they start at 0. (0, 1, 2,...)")
    parser.add('--one_hot_encode', action="store_true", help="OneHotEncode the categorical features")

    # parser.add('--batch_size', type=int, default=128, help="Batch size used for training")
    parser.add('--val_batch_size', type=int, default=128, help="Batch size used for training and testing")
    parser.add('--early_stopping_rounds', type=int, default=20, help="Number of rounds before early stopping applies.")
    parser.add('--epochs', type=int, default=1000, help="Max number of epochs to train.")
    parser.add('--logging_period', type=int, default=100, help="Number of iteration after which validation is printed.")

    parser.add('--num_features', type=int, required=True, help="Set the total number of features.")
    parser.add('--num_classes', type=int, default=1, help="Set the number of classes in a classification task.")
    parser.add('--cat_idx', type=int, action="append", help="Indices of the categorical features")
    parser.add('--cat_dims', type=int, action="append", help="Cardinality of the categorical features (is set "
                                                             "automatically, when the load_data function is used.")

    args = parser.parse_args()

    if args.all_models:
        all_models = ["DeepFM","TabTransformer","Linear","TORCHRLN","VIME", "TabTransformer","LinearModel", "TabTransformer", "Net"] # "DeepFM", "TabTransformer", "LinearModel", "VIME", "Net", "RLN",
        # "TabNet", , "SAINT" , "DANet" , "XGBoost", "CatBoost", "LightGBM", "KNN", "DecisionTree", "RandomForest", "ModelTree",  "DNFNet",  "STG", "NAM",  "MLP",  "NODE", "DeepGBM",
    elif not args.all_models:
        all_models = [args.model_name]
    # load_data
    x_train_original, y_train, dataset_test, scaler_train, encoder_train  = load_data(args, scale=0, one_hot_encode=args.one_hot_encode, split="train-val")
    x_test, y_test, dataset_test, scaler_test, encoder_test = load_data(args, scale=args.scale,
                                                                                one_hot_encode=args.one_hot_encode,
                                                                                split="test")

    if args.epsilon_std>0:
        args.epsilon = args.epsilon_std*x_test.std()

    x_test_original = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    y_test = y_test.to(torch.long)



    for one_model in all_models:
        model, x_train, x_test, scaler = init_model(one_model, args, scaler_train, x_train_original, x_test_original, y_train, y_test)
        min_, max_ = x_train.min(), x_train.max()
        # create save dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # load attack
        from autoattack import AutoAttack

        constraints = dataset_test.get_constraints()
        if not args.use_constraints:
            a = LessEqualConstraint(Constant(float(constraints.lower_bounds[0])-1),Feature(0))
            b = LessEqualConstraint(Constant(float(constraints.lower_bounds[0])-1),Feature(0))
            constraints.relation_constraints = [(a), (b)]
            #constraints = AndConstraint(operands =[(Constant(0) <= Constant(1)), (Constant(0) <= Constant(1))]) #Constraints([],[],[],[], [(Constant(0) <= Constant(1)), (Constant(0) <= Constant(1))], []) # AndConstraint(operands =[(Constant(0) <= Constant(1))]) #
        # constraints = None

        fun_distance_preprocess = lambda x: (scaler_train.inverse_transform(encoder_train.inverse_transform()),
                                             scaler_train, encoder_train)
        adversary = AutoAttack(model=model, arguments=args, constraints=constraints, norm=args.norm, eps=args.epsilon,
                               log_path=args.log_path,
                               version=args.version, verbose=args.verbose,
                               fun_distance_preprocess=lambda x: fun_distance_preprocess)

        if args.version == 'transfer':
            adversary.attacks_to_run = ['transfer']

        if args.version == 'all':
            adversary.attacks_to_run = ['apgd-ce-constrained', 'fab-constrained','moeva2'] if args.use_constraints else ['apgd-ce', 'fab','moeva2']

        # example of custom version
        if args.version == 'custom':
            print(args.use_constraints)
            if args.use_constraints:
                adversary.attacks_to_run = ['apgd-ce-constrained', 'fab-constrained','moeva2'] # 'apgd-t-ce-constrained', 'fab-constrained',
            elif not args.use_constraints:
                adversary.attacks_to_run = ['apgd-ce', 'fab','moeva2']  # 'apgd-t-ce-constrained', 'fab-constrained',
                adversary.attacks_to_run = ['moeva2']

                constraints = [Constant(0) <= Constant(1)]
            adversary.apgd.n_restarts = 2
            adversary.fab.n_restarts = 2

        # run attack and save images

        with torch.no_grad():
            
            x_test_l = x_test[:args.n_ex]
            y_test_l = y_test[:args.n_ex]

            if not args.individual:
                adv_complete, y_adv_complete = adversary.run_standard_evaluation(x_test_l, y_test_l,
                                                                 bs=args.batch_size,
                                                                 return_labels=True,
                                                                 x_unscaled=x_unpreprocessed[:args.n_ex],
                                                                 min_ = min_, max_ =max_             )

                torch.save({'adv_complete': adv_complete}, '{}/{}_{}_dataset_{}_norm_{}_1_{}_eps_{:.5f}_{}_constraints_{}.pth'.format(
                    args.save_dir, 'aa', args.version, args.dataset, args.norm, adv_complete.shape[0],
                    args.epsilon, args.model_name, args.use_constraints))
                torch.save({'y_adv_complete': y_adv_complete}, '{}/{}_{}_dataset_{}_norm_{}_1_{}_eps_{:.5f}_{}_{}_y.pth'.format(
                    args.save_dir, 'aa', args.version, args.dataset, args.norm, y_adv_complete.shape[0],
                    args.epsilon, args.model_name, args.use_constraints))

            else:
                # individual version, each attack is run on all test points
                adv_complete = adversary.run_standard_evaluation_individual(x_test_l,
                                                                            y_test_l, bs=args.batch_size)

                torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}_{}_{}.pth'.format(
                    args.save_dir, 'aa', args.version, args.n_ex, args.epsilon, args.model_name, args.use_constraints))
