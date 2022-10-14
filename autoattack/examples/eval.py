import argparse
import os
import sys

import numpy as np
import torch
from constrained_attacks import datasets
from sklearn.preprocessing import StandardScaler

from autoattack.other_utils import add_normalization_layer
from pipeline.pytorch import Net
from tests.attacks.moeva.url_constraints import get_url_constraints

sys.path.insert(0,'..')

from resnet import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    
    args = parser.parse_args()

    my_datasets = ["lcld_v2_time", "ctu_13_neris", "url", "malware"]
    # feature_number = [28,756,63,24222]
    my_models = ['./tests/resources/pytorch_models/lcld_v2_time_torch.pth',
                 './tests/resources/pytorch_models/ctu_13_neris_test_torch.pth',
                 './tests/resources/pytorch_models/url_test_torch.pth',
                 './tests/resources/pytorch_models/malware_test_torch.pth']
    data_indicator = 2
    args.model = my_models[data_indicator]

    # load_data
    dataset = datasets.load_dataset(my_datasets[data_indicator])
    x, y = dataset.get_x_y()
    preprocessor = StandardScaler()  # dataset.get_preprocessor()
    splits = dataset.get_splits()
    preprocessor.fit(x.iloc[splits["train"]])
    x = preprocessor.transform(x).astype(np.float32)
    x_test = x[splits["test"]]
    y_test = y[splits["test"]]
    mean, std = preprocessor.mean_, preprocessor.scale_
    mean = mean.reshape(1,-1).astype(np.float32)
    std = std.reshape(1,-1).astype(np.float32)
    args.epsilon = 2*np.mean(std)

    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    y_test = y_test.to(torch.long)


    class Normalize(nn.Module):
        def __init__(self, meanl, stdl):
            super(Normalize, self).__init__()
            self.register_buffer('meanl', torch.Tensor(meanl))
            self.register_buffer('stdl', torch.Tensor(stdl))

        def forward(self, input):
            return (input - self.meanl) / self.stdl

    # load model
    model = Net(preprocessor, x.shape[1])
    ckpt = torch.load(args.model, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    #model.cuda()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = torch.nn.Sequential(
        Normalize(meanl=mean, stdl=std),
        model
    )
    # model = add_normalization_layer(model=model, mean=mean, std=std)
    model.to(device)
    model.eval()

    # load data
    # transform_list = [transforms.ToTensor()]
    # transform_chain = transforms.Compose(transform_list)
    # item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    # test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)



    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load attack    
    from autoattack import AutoAttack

    constraints = dataset.get_constraints()
    adversary = AutoAttack(model=model, constraints=constraints, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)



    #l = [x for (x, y) in test_loader]
    #x_test = torch.cat(l, 0)
    #l = [y for (x, y) in test_loader]
    #y_test = torch.cat(l, 0)
    
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['fab-constrained', 'fab']  # 'apgd-t-ce-constrained'
        adversary.apgd.n_restarts = 2
        # adversary.fab.n_restarts = 2

    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size)
            
            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_dataset_{}_norm_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, my_datasets[data_indicator], args.norm, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
                
