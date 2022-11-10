import argparse
import os
import sys

import numpy as np
import torch
from constrained_attacks import datasets
from sklearn.preprocessing import StandardScaler


# from autoattack.other_utils import add_normalization_layer
sys.path.insert(0,'.')
from pipeline.pytorch import Net
from constraints.relation_constraint import Constant

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
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='custom')
    parser.add_argument('--model_name', type=str, default='Net')
    parser.add_argument('--use_constraints', type=int, default=1)

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
    all_models = ["TabTransformer", "LinearModel", "Net", "DeepFM", "RLN"] # "DeepFM", "TabTransformer", "LinearModel", "VIME", "Net", "RLN",
    # "TabNet", , "SAINT" , "DANet" , "XGBoost", "CatBoost", "LightGBM", "KNN", "DecisionTree", "RandomForest", "ModelTree",  "DNFNet",  "STG", "NAM",  "MLP",  "NODE", "DeepGBM",

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
    mean, std = preprocessor.mean_, preprocessor.scale_
    mean = mean.reshape(1,-1).astype(np.float32)
    std = std.reshape(1,-1).astype(np.float32)
    args.epsilon = np.mean(std) # budget to be adapted

    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    y_test = y_test.to(torch.long)


    class Normalize(nn.Module):
        def __init__(self, meanl, stdl):
            super(Normalize, self).__init__()
            self.register_buffer('meanl', torch.Tensor(meanl))
            self.register_buffer('stdl', torch.Tensor(stdl))

        def forward(self, input):
            if not torch.is_tensor(input):
                input = torch.tensor(input)
            return (input - self.meanl) / self.stdl



    for one_model in all_models:
        args.model = 'trained_models/' + one_model + '/' + args.dataset + '/m_best.pt'

        print("model = ", one_model, " ; dataset = ", args.dataset)

        # load model
        if one_model == "Net":
            args.use_gpus = False
            model = Net(preprocessor, x.shape[1])
            ckpt = torch.load(args.model, map_location=torch.device("cpu")) # "cpu"
            model.load_state_dict(ckpt)
            # model.cuda()
            if torch.cuda.is_available():
                device = torch.device(0) # "cuda"
            else:
                device = torch.device(0)
            model = torch.nn.Sequential(
                Normalize(meanl=mean, stdl=std),
                model
            )
            # model = add_normalization_layer(model=model, mean=mean, std=std)
            model.to(device)
            model.eval()
        else:
            from models import str2model
            # adapt to type of model being run
            import ast
            param_path = 'trained_models/' + one_model + '/' + args.dataset + '/parameters.json'
            parameters = ast.literal_eval(open(param_path).read())
            print("parameters : ", parameters)
            model = str2model(one_model)(parameters, args)
            if one_model == "RLN":
                X_test, Y_test = np.array(x_test), np.array(y_test)
                X_train, Y_train  = np.array(x_train), np.array(y_train)
                model.fit(X_train, Y_train, X_test, Y_test)
            else:
                state_dict = torch.load(args.model, map_location=torch.device('cpu'))
                if one_model == "LinearModel":
                    model = state_dict
                else:
                    from collections import OrderedDict
                    if one_model not in ["DeepFM", "LinearModel", "TabTransformer"] or (one_model == "TabTransformer" and args.dataset == "url"):
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = 'module.' + k[:]  # add `module.`
                            new_state_dict[name] = v
                    else:
                        new_state_dict = state_dict
                    model.model.load_state_dict(new_state_dict)
                    device = torch.device(0)  # "cpu"
                    model.model.to(device)
                    model.model.eval()

        # create save dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # load attack
        from autoattack import AutoAttack

        constraints = dataset.get_constraints()
        # constraints = None
        adversary = AutoAttack(model=model, constraints=constraints, norm=args.norm, eps=args.epsilon,
                               log_path=args.log_path,
                               version=args.version, fun_distance_preprocess=lambda x: preprocessor.transform(x))

        # l = [x for (x, y) in test_loader]
        # x_test = torch.cat(l, 0)
        # l = [y for (x, y) in test_loader]
        # y_test = torch.cat(l, 0)

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
            if not args.individual:
                adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                                                                 bs=args.batch_size,
                                                                 x_unscaled=x_unpreprocessed[:args.n_ex])

                torch.save({'adv_complete': adv_complete}, '{}/{}_{}_dataset_{}_norm_{}_1_{}_eps_{:.5f}.pth'.format(
                    args.save_dir, 'aa', args.version, args.dataset, args.norm, adv_complete.shape[0],
                    args.epsilon))

            else:
                # individual version, each attack is run on all test points
                adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                                                                            y_test[:args.n_ex], bs=args.batch_size)

                torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                    args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))


    # load data
    # transform_list = [transforms.ToTensor()]
    # transform_chain = transforms.Compose(transform_list)
    # item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    # test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

