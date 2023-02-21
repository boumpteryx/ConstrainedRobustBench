import torch
import numpy as np
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),"src/constrained-attacks"))

from utils.load_data import load_data

from constrained_attacks.constraints.relation_constraint import Constant
from constrained_attacks.constraints.relation_constraint import LessEqualConstraint, Feature

from utils.models import init_model
from utils.parser import get_parser

if __name__ == '__main__':
    parser = get_parser("eval")
    args = parser.parse_args()

    all_models = args.model_name.split(";")
    # load_data
    x_train_original, y_train, dataset_test, scaler_train, encoder_train  = load_data(args, scale=args.scale, one_hot_encode=args.one_hot_encode, split="train-val")
    x_test, y_test, dataset_test, scaler_test, encoder_test = load_data(args, scale=args.scale,
                                                                                one_hot_encode=args.one_hot_encode,
                                                                                split="test", reuse_scaler=scaler_train,
                                                                                reuse_encoder=encoder_train)

    x_test_raw, _, _, _, _ = load_data(args, scale=0,
                                                                        one_hot_encode=0,
                                                                        split="test", reuse_scaler=scaler_train,
                                                                        reuse_encoder=encoder_train)

    args.num_dense_features = args.num_features - len(
        args.cat_idx) if args.cat_idx is not None else args.num_features
    args.num_features = x_test.shape[1]

    def softargmax(x, dim=-1):
        # crude: assumes max value is unique
        beta = 100.0
        xx = beta * x
        sm = torch.nn.functional.softmax(xx, dim=dim)
        indices = torch.arange(x.shape[dim])
        y = torch.mul(indices, sm)
        result = torch.sum(y, dim)
        return result


    def fun_preprocess_to_feature(x):
        scaler = scaler_train
        encoder = encoder_train
        nb_features = x.shape[1]

        num_idx = [a for a in range(nb_features) if a not in args.cat_idx]
        x[:, num_idx] = scaler.transform(x[:, num_idx])
        new_x1 = None if args.cat_idx is None else encoder.transform(x[:, args.cat_idx])
        new_x2 = x[:, num_idx]
        x = new_x2 if new_x1 is None else np.concatenate([new_x1, new_x2], axis=1)

        return x


    def fun_preprocess_to_problem(x):
        nb_features = x.shape[1]
        x_cat = x[:, 0:nb_features - args.num_dense_features]
        x_num_unscaled = x[:, nb_features - args.num_dense_features:nb_features]

        if isinstance(x, torch.Tensor):
            ## process as tensor to preserve gradient
            x_num_unscaled = x_num_unscaled * (
                        torch.Tensor(scaler_train.data_max_) - torch.Tensor(scaler_train.data_min_)) + torch.Tensor(
                scaler_train.data_min_)
            x_cat_encoded = torch.split(x_cat, [len(a) for a in encoder_train.categories_], 1)
            # x_cat_argmax = [a.argmax(1) for a in x_cat_encoded]
            x_cat_softmax = [softargmax(a, 1) for a in x_cat_encoded]
            x_cat_unencoded = torch.stack(x_cat_softmax).swapaxes(0, 1)
            x_reversed = torch.zeros((x.shape[0], x_num_unscaled.shape[1] + x_cat_unencoded.shape[1]))
        else:
            ## process as numpy
            x_num_unscaled = scaler_train.inverse_transform(x_num_unscaled)
            x_cat_unencoded = encoder_train.inverse_transform(x_cat)
            x_reversed = np.zeros((x.shape[0], x_num_unscaled.shape[1] + x_cat_unencoded.shape[1]))

        num_index = [a for a in range(x_reversed.shape[1]) if a not in args.cat_idx]
        x_reversed[:, args.cat_idx] = x_cat_unencoded
        x_reversed[:, num_index] = x_num_unscaled

        return x_reversed, scaler_train, encoder_train

    if args.epsilon_std>0:
        args.epsilon = args.epsilon_std*x_test.std()

    x_test_original = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    y_test = y_test.to(torch.long)

    constraints = dataset_test.get_constraints()
    if not args.use_constraints:
        a = LessEqualConstraint(Constant(float(constraints.lower_bounds[0]) - 1), Feature(0))
        b = LessEqualConstraint(Constant(float(constraints.lower_bounds[0]) - 1), Feature(0))
        constraints.relation_constraints = [(a), (b)]

    from constrained_attacks.constraints.constraints_checker import ConstraintChecker

    ## sanity check that the original inputs satisfy the constraints
    checker = ConstraintChecker(constraints, tolerance=args.constraint_tolerance)
    x_test2, _, _ = fun_preprocess_to_problem(x_test[:args.n_ex])
    check_processing = x_test_raw[:args.n_ex] == x_test2
    correct_processing = (x_test_raw[:args.n_ex] - x_test2) < args.constraint_tolerance
    print("Initial inputs features different after scaling & encoding {}%".format(100-correct_processing.mean()*100))
    check_constraints = checker.check_constraints(x_test2, x_test2, pt=True)
    counter = len(check_constraints) - check_constraints.sum()
    print("number of initial inputs not respecting constraints {}/{}".format(counter,len(check_constraints)))

    for one_model in all_models:

        model, x_train, x_test, scaler = init_model(one_model, args, scaler_train, x_train_original, x_test_original, y_train, y_test)
        min_, max_ = x_train.min(), x_train.max()
        # create save dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # load attack
        from autoattack import AutoAttack


        adversary = AutoAttack(model=model, arguments=args, constraints=constraints, norm=args.norm, eps=args.epsilon,
                               log_path=args.log_path,
                               version=args.version, verbose=args.verbose,
                               fun_preprocess_to_problem=fun_preprocess_to_problem, fun_preprocess_to_feature=fun_preprocess_to_feature)

        if args.version == 'transfer':
            adversary.attacks_to_run = ['transfer']

        elif args.version == 'all':
            adversary.attacks_to_run = ['apgd-ce-constrained', 'fab-constrained','moeva2'] if args.use_constraints else ['apgd-ce', 'fab','moeva2']

        # example of custom version
        elif args.version == 'custom':
            print(args.use_constraints)
            if args.use_constraints:
                adversary.attacks_to_run = ['apgd-ce-constrained', 'fab-constrained','moeva2'] # 'apgd-t-ce-constrained', 'fab-constrained',
            elif not args.use_constraints:
                adversary.attacks_to_run = ['apgd-ce','fab','moeva2']  # 'apgd-t-ce-constrained', 'fab-constrained',
            adversary.apgd.n_restarts = 2
            adversary.fab.n_restarts = 2

        else:
            adversary.attacks_to_run = args.version.split("#")
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
                                                                 x_unscaled=None,
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
