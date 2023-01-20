import argparse
import os
import sys

import numpy as np

from constrained_attacks import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0,'.')
from constrained_attacks.constraints.relation_constraint import Constant
from constrained_attacks.constraints.relation_constraint import LessEqualConstraint, Feature

from utils.models import init_model

sys.path.insert(0,'..')

from resnet import *

import configargparse
import yaml

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--model', type=str, default='./tests/resources/pytorch_models/url_torch.pth')
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./autoattack/examples/results')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='custom')
    parser.add_argument('--model_name', type=str, default='Net',help="Use '#' to use multiple models in an ensemble")
    parser.add_argument('--use_constraints', type=int, default=1)
    parser.add_argument('--all_models', type=int, default=0)
    parser.add_argument('--transfer_from', type=str, default=None)

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

    my_datasets = ["lcld_v2_time", "ctu_13_neris", "url", "malware"]
    # feature_number = [28,756,63,24222]
    my_models = ['./tests/resources/pytorch_models/lcld_v2_time_torch.pth',
                 './tests/resources/pytorch_models/ctu_13_neris_test_torch.pth',
                 './tests/resources/pytorch_models/url_test_torch.pth',
                 './tests/resources/pytorch_models/malware_test_torch.pth']
    if args.all_models:
        all_models = ["DeepFM","TabTransformer","Linear","TORCHRLN","VIME", "TabTransformer","LinearModel", "TabTransformer", "Net"] # "DeepFM", "TabTransformer", "LinearModel", "VIME", "Net", "RLN",
        # "TabNet", , "SAINT" , "DANet" , "XGBoost", "CatBoost", "LightGBM", "KNN", "DecisionTree", "RandomForest", "ModelTree",  "DNFNet",  "STG", "NAM",  "MLP",  "NODE", "DeepGBM",
    elif not args.all_models:
        all_models = [args.model_name]
    # load_data
    dataset = datasets.load_dataset(args.dataset)
    x, y = dataset.get_x_y()
    preprocessor = StandardScaler()  # dataset.get_preprocessor()
    splits = dataset.get_splits()
    preprocessor.fit(x.iloc[splits["train"]])
    x_unpreprocessed = torch.FloatTensor(np.array(x)[splits["test"]])
    x = preprocessor.transform(x).astype(np.float32)
    x_test = x[splits["test"]]
    y_test = y[splits["test"]]
    x_train = x[splits["train"]]
    y_train = y[splits["train"]]
    if args.dataset == "ctu_13_neris":
        _, x_test, _, y_test = train_test_split(
            x_test, y_test, test_size=args.n_ex, random_state=42, shuffle=True, stratify=y_test
        )


    mean, std = preprocessor.mean_, preprocessor.scale_
    mean = mean.reshape(1,-1).astype(np.float32)
    std = std.reshape(1,-1).astype(np.float32)
    args.epsilon = np.mean(std) # budget to be adapted

    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    y_test = y_test.to(torch.long)



    for one_model in all_models:
        model = init_model(one_model, args, preprocessor, x_train, x_test, y_train, y_test)

        # create save dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # load attack
        from autoattack import AutoAttack

        constraints = dataset.get_constraints()
        if not args.use_constraints:
            a = LessEqualConstraint(Constant(float(constraints.lower_bounds[0])-1),Feature(0))
            b = LessEqualConstraint(Constant(float(constraints.lower_bounds[0])-1),Feature(0))
            constraints.relation_constraints = [(a), (b)]
            #constraints = AndConstraint(operands =[(Constant(0) <= Constant(1)), (Constant(0) <= Constant(1))]) #Constraints([],[],[],[], [(Constant(0) <= Constant(1)), (Constant(0) <= Constant(1))], []) # AndConstraint(operands =[(Constant(0) <= Constant(1))]) #
        # constraints = None
        adversary = AutoAttack(model=model, arguments=args, constraints=constraints, norm=args.norm, eps=args.epsilon,
                               log_path=args.log_path,
                               version=args.version,
                               fun_distance_preprocess=lambda x: preprocessor.transform(x))

        # l = [x for (x, y) in test_loader]
        # x_test = torch.cat(l, 0)
        # l = [y for (x, y) in test_loader]
        # y_test = torch.cat(l, 0)
        if args.version == 'transfer':
            adversary.attacks_to_run = ['transfer']

        # example of custom version
        if args.version == 'custom':
            print(args.use_constraints)
            if args.use_constraints:
                adversary.attacks_to_run = ['apgd-ce-constrained', 'fab-constrained','moeva2'] # 'apgd-t-ce-constrained', 'fab-constrained',
            elif not args.use_constraints:
                adversary.attacks_to_run = ['apgd-ce', 'fab','moeva2']  # 'apgd-t-ce-constrained', 'fab-constrained',
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
                                                                 x_unscaled=x_unpreprocessed[:args.n_ex])

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


    # load data
    # transform_list = [transforms.ToTensor()]
    # transform_chain = transforms.Compose(transform_list)
    # item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    # test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

