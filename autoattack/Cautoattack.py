import math
import time

import numpy as np
import torch
from constrained_attacks.objective_calculator.cache_objective_calculator import ObjectiveCalculator

from autoattack.other_utils import Logger
from autoattack import checks
from constrained_attacks.classifier.classifier import Classifier
from constraints.constraints_checker import ConstraintChecker


class AutoAttack():
    def __init__(self, model, arguments, constraints=None, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cpu', log_path=None, fun_distance_preprocess=None):
        self.model = model
        self.n_ex = arguments.n_ex
        self.use_constraints = arguments.use_constraints
        self.dataset = arguments.dataset
        self.constraints = constraints
        self.norm = norm
        self.save_dir = arguments.save_dir
        self.transfer_from = arguments.transfer_from
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)
        self.fun_distance_preprocess = fun_distance_preprocess

        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")

        if not self.is_tf_model:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, constraints=self.constraints, n_restarts=5, n_iter=100, verbose=False,#self.verbose,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                device=self.device, logger=self.logger)

            from .fab_pt import FABAttack_PT
            self.fab = FABAttack_PT(self.model, self.constraints, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)

            from .square import SquareAttack
            self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=self.verbose, device=self.device, resc_schedule=False)

            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, constraints=self.constraints, n_restarts=1, n_iter=100, verbose=self.verbose,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                logger=self.logger)

            from constrained_attacks.attacks.moeva.moeva import Moeva2
            self.moeva2 = Moeva2(classifier_class = Classifier(self.model), constraints = self.constraints, norm=self.norm, fun_distance_preprocess=self.fun_distance_preprocess, n_jobs=1, verbose=0)

        else:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=self.verbose,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)

            from .fab_tf import FABAttack_TF
            self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=self.verbose, device=self.device)

            from .square import SquareAttack
            self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=self.verbose, device=self.device, resc_schedule=False)

            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model,constraints=self.constraints,  n_restarts=1, n_iter=100, verbose=self.verbose,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)

            from constrained_attacks.attacks.moeva.moeva import Moeva2
            self.moeva2 = Moeva2(classifier_class=Classifier(self.model), constraints=self.constraints)

        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)

    def get_logits(self, x):
        if callable(self.model) and not self.is_tf_model:
            return self.model(x)
        elif not callable(self.model) and not self.is_tf_model and hasattr(self.model, "predict_proba"):
            return torch.tensor(self.model.predict_proba(x))
        elif not self.is_tf_model and hasattr(self.model, "predict_proba"):
            import xgboost
            return torch.tensor(self.model.predict_proba(xgboost.DMatrix(x))) # for Booster
        else:
            return self.model.predict(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def run_standard_evaluation(self, x_orig, y_orig, bs=250, return_labels=False, x_unscaled=None):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))

        # checks on type of defense
        """if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
                y_orig[:bs].to(self.device), bs=bs, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device),
            logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), self.is_tf_model,
            logger=self.logger)
        checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes, logger=self.logger)"""

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            y_adv = torch.empty_like(y_orig)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                if self.is_tf_model:
                    output = torch.tensor(self.get_logits(x))
                else:
                    output = self.get_logits(x).max(dim=1)[1]
                y_adv[start_idx: end_idx] = output
                correct_batch = y.eq(output)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
            from sklearn.metrics import roc_auc_score, matthews_corrcoef
            robust_AUC = roc_auc_score(y_orig, y_adv)
            robust_MCC = matthews_corrcoef(y_orig, y_adv)
            robust_accuracy_dict = {'clean': robust_accuracy}

            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
                self.logger.log('initial AUC: {:.2%}'.format(robust_AUC))
                self.logger.log('initial MCC: {:.2%}'.format(robust_MCC))

            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    if x_unscaled is not None:
                        x_unscaled_usable = x_unscaled[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True

                    elif attack == 'apgd-ce-constrained':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce-constrained'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True

                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True

                    elif attack == 'apgd-dlr-constrained':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr-constrained'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True

                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.is_constrained = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'fab-constrained': # constraints checked within the perturb function
                        # fab
                        self.fab.targeted = False
                        self.fab.is_constrained = True
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)

                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True

                    elif attack == 'apgd-t-constrained':
                        # targeted apgd
                        self.apgd_targeted.loss = 'dlr-targeted-constrained'
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True

                    elif attack == 'apgd-t-ce-constrained':
                        # targeted apgd
                        self.apgd_targeted.loss = 'ce-targeted-constrained'
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True

                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.is_constrained = False
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'fab-t-constrained': # constraints checked within the perturb function
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.is_constrained = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'moeva2':
                        self.moeva2.is_targeted = False
                        adv_curr = self.moeva2.generate(np.array(x_unscaled_usable.numpy()),np.array(y.numpy()),x.shape[0])
                        threshold = {"misclassification":np.array([np.inf, np.inf]),"distance":self.epsilon, "constraints": 0.01}
                        calcul = ObjectiveCalculator(Classifier(self.model),constraints=self.constraints,thresholds=threshold,norm=2,fun_distance_preprocess=self.fun_distance_preprocess)
                        adv_curr = calcul.get_successful_attacks(
                            np.array(x_unscaled_usable.numpy()),
                            np.array(y.numpy()),
                            adv_curr,
                            preferred_metrics="misclassification",
                            order="asc",
                            max_inputs=1,
                            return_index_success=False,
                            recompute=True,
                        )
                        # adv_curr = torch.tensor(adv_curr[:,0,:])
                        adv_curr = torch.tensor(adv_curr)
                        # remove outputs that do not respect constraints
                        checker = ConstraintChecker(self.constraints, tolerance=0.01)
                        check = checker.check_constraints(x, adv_curr, pt=True)
                        for i in range(len(check)):
                            counter = 0
                            if not check[i]:
                                counter += 1
                                adv_curr[i] = x[i]
                        print("number of outputs not respecting constraints = ", counter)


                    elif attack == 'moeva2-t':
                        self.moeva2.is_targeted = True # useless for now
                        adv_curr = self.moeva2.generate(np.array(x_unscaled.numpy()),np.array(y.numpy()),x.shape[0])
                        threshold = {"misclassification": np.array([np.inf, np.inf]), "distance": self.epsilon,
                                     "constraints": 0.01}
                        calcul = ObjectiveCalculator(Classifier(self.model), constraints=self.constraints, thresholds=threshold,
                                                     norm=2, fun_distance_preprocess=self.fun_distance_preprocess)
                        adv_curr = calcul.get_successful_attacks(
                            np.array(x_unscaled_usable.numpy()),
                            np.array(y.numpy()),
                            adv_curr,
                            preferred_metrics="misclassification",
                            order="asc",
                            max_inputs=1,
                            return_index_success=False,
                            recompute=True,
                        )
                        adv_curr = torch.tensor(adv_curr)
                        # remove outputs that do not respect constraints
                        checker = ConstraintChecker(self.constraints, tolerance=0.01)
                        check = checker.check_constraints(x, adv_curr, pt=True)
                        for i in range(len(check)):
                            counter = 0
                            if not check[i]:
                                counter += 1
                                adv_curr[i] = x[i]
                        print("number of outputs not respecting constraints = ", counter)

                    elif attack == "transfer":
                        if self.transfer_from is not None:
                            adv_curr = torch.load('{}/{}_{}_dataset_{}_norm_{}_1_{}_eps_{:.5f}_{}_constraints_{}.pth'.format(
                                self.save_dir, 'aa', "custom", self.dataset, self.norm, self.n_ex,
                                self.epsilon, self.transfer_from, self.use_constraints))["adv_complete"][start_idx:end_idx]
                            print(adv_curr.shape)
                        else:
                            print("missing specification of model to transfer from")

                    else:
                        raise ValueError('Attack not supported')

                    if attack in ['moeva2', 'moeva2-t', 'apgd-t-ce-constrained', 'apgd-t-constrained', 'apgd-ce-constrained', 'apgd-dlr-constrained']:
                        # remove outputs that do not respect constraints
                        checker = ConstraintChecker(self.constraints, tolerance=0.01)
                        check = checker.check_constraints(x, adv_curr, pt=True)
                        for i in range(len(check)):
                            counter = 0
                            if not check[i]:
                                counter += 1
                                adv_curr[i] = x[i]
                        print("number of outputs not respecting constraints = ", counter)

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    # output.to(torch.float)

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                if self.verbose:
                    num_non_robust_batch = torch.sum(false_batch)
                    self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
                        attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_AUC = roc_auc_score(y_orig, y_adv)
                robust_MCC = matthews_corrcoef(y_orig, y_adv)
                robust_accuracy_dict[attack] = robust_accuracy
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))
                    self.logger.log('robust AUC after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_AUC, time.time() - startt))
                    self.logger.log('robust MCC after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_MCC, time.time() - startt))

            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
                self.logger.log('robust AUC: {:.2%}'.format(robust_AUC))
                self.logger.log('robust MCC: {:.2%}'.format(robust_MCC))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv

    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))

        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))

        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False

        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            if return_labels:
                adv[c] = (x_adv, y_adv)
            else:
                adv[c] = x_adv
            if verbose_indiv:
                acc_indiv  = self.clean_accuracy(x_adv, y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))

        return adv

    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))

        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000

        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))

        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20

