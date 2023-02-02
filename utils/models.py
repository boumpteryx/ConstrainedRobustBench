import torch
import numpy as np
from pipeline.pytorch import Net, Linear
from models.basemodel_torch import BaseModelTorch

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

class Denormalize(torch.nn.Module):
    def __init__(self, min=0, max=1,scaler=None):
        super(Denormalize, self).__init__()
        self.minval = min #torch.Tensor(min)
        self.maxval = max #torch.Tensor(max)
        self.scaler = scaler

    def forward(self, input):
        if self.scaler is not None:
            return self.scaler.inverse_transform(input)

        x = input*(self.maxval-self.minval) + self.minval
        return x.to(input.dtype)

class Normalize(torch.nn.Module):
    def __init__(self, meanl, stdl):
        super(Normalize, self).__init__()
        self.register_buffer('meanl', torch.Tensor(meanl))
        self.register_buffer('stdl', torch.Tensor(stdl))

    def forward(self, input):
        if not torch.is_tensor(input):
            input = torch.tensor(input)
        return (input - self.meanl) / self.stdl

class EnsembleModel(BaseModelTorch):
    def __init__(self, models, params, args):
        super().__init__(params, args)
        self.models = models
        self.strategy = "MEAN_PROBA"

    def fit(self, X, y, X_val=None, y_val=None):
        print("Fitting Ensemble with multiple models: {}".format(self.models))
        for i, model in tqdm(enumerate(self.models), total=len(self.models)):
            model.fit(X, y, X_val, y_val)


    def predict_helper(self, X, keep_grad=False):
        preds = [model.predict_helper(X, keep_grad) for model in self.models]

        if keep_grad:
            if self.strategy == "MEAN_PROBA":
                preds = torch.mean(preds,0)

        else:
            if self.strategy == "MEAN_PROBA":
                preds = np.mean(preds,0)
        return preds


def init_model(one_model, args, preprocessor, x_train, x_test, y_train, y_test):
    mean, std = preprocessor.mean_, preprocessor.scale_
    x_train_normalized = x_train
    x_test_normalized = x_test
    scaler = None

    args.model = 'trained_models/' + one_model + '/' + args.dataset + '/m_best.pt'

    print("model = ", one_model, " ; dataset = ", args.dataset)
    print("use_constraint = ", args.use_constraints)

    # load model
    if one_model == "Net":
        args.use_gpus = False
        model = Net(preprocessor, x_train.shape[1])
        ckpt = torch.load(args.model, map_location=torch.device("cpu"))  # "cpu"
        model.load_state_dict(ckpt)
        # model.cuda()
        if torch.cuda.is_available():
            device = torch.device("cpu")  # "cuda"
        else:
            device = torch.device("cpu")
        model = torch.nn.Sequential(
            Normalize(meanl=mean, stdl=std),
            model
        )
        # model = add_normalization_layer(model=model, mean=mean, std=std)
        model.to(device)
        model.eval()
    elif one_model == "Linear":
        args.use_gpus = False
        model = Linear(preprocessor, x_train.shape[1])
        ckpt = torch.load(args.model, map_location=torch.device("cpu"))  # "cpu"
        model.load_state_dict(ckpt)
        # model.cuda()
        if torch.cuda.is_available():
            device = torch.device("cpu")  # "cuda"
        else:
            device = torch.device("cpu")
        model = torch.nn.Sequential(
            Normalize(meanl=mean, stdl=std),
            model
        )
        # model = add_normalization_layer(model=model, mean=mean, std=std)
        model.to(device)
        model.eval()
    elif "#" in one_model:
        models = []
        models_str = one_model.split("#")
        for model_str in models_str:
            model = init_model(model_str, args, preprocessor, x_train, x_test, y_train, y_test)
            models.append(model)

        model = EnsembleModel(models, {}, args)

    else:
        from models import str2model
        # adapt to type of model being run
        import ast
        param_path = 'trained_models/' + one_model + '/' + args.dataset + '/parameters.json'
        parameters = ast.literal_eval(open(param_path).read())
        print("models' parameters : ", parameters)
        model = str2model(one_model)(parameters, args)
        if one_model == "VIME":
            filename_self = 'trained_models/' + one_model + '/' + args.dataset + '/m_best_self.pt'
            state_dict = torch.load(filename_self, map_location=torch.device('cpu'))
            model.model_self.load_state_dict(state_dict)
            filename_semi = 'trained_models/' + one_model + '/' + args.dataset + '/m_best_semi.pt'
            state_dict = torch.load(filename_semi, map_location=torch.device('cpu'))
            model.model_semi.load_state_dict(state_dict)
        if one_model == "RLN":
            from autoattack.utils_tf2 import ModelAdapter
            X_test, Y_test = np.array(x_test), np.array(y_test)
            X_train, Y_train = np.array(x_train), np.array(y_train)
            model.fit(X_train, Y_train, X_test, Y_test)
            model = ModelAdapter(model.model.model, num_classes=2)
        else:
            state_dict = torch.load(args.model, map_location=torch.device('cpu'))
            if one_model == "LinearModelSklearn":
                model = state_dict
            else:
                from collections import OrderedDict
                if one_model not in ["DeepFM", "TabTransformer", "TORCHRLN", "SAINT"]:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = 'module.' + k[:]  # add `module.`
                        new_state_dict[name] = v
                else:
                    new_state_dict = state_dict
                model.model.load_state_dict(new_state_dict)
                device = torch.device('cpu')  # "cpu"
                model.model.to(device)
                model.model.eval()

        if x_test.max()>1 or x_train.max()>1:

            small_eps = 1e-10

            ##TODO: Extend to per feature min and max
            min_ = x_train.min()
            max_ = x_train.max()

            scaler = MinMaxScaler().fit(x_train)
            model.model = torch.nn.Sequential(
                Denormalize(min=min_,max=max_),
                model.model
            )
            x_test_normalized = ((x_test - min_) / (max_ - min_+small_eps))
            x_train_normalized = ((x_train - min_) / (max_ - min_+small_eps))



    return model, x_train_normalized, x_test_normalized, scaler
