import torch
import numpy as np
### Pytorch loss
class BalancedBCELossPytorch(torch.nn.BCEWithLogitsLoss):

    def __init__(self, dataset, **args):
        self.weights = None
        if dataset in ["url", "malware", "ctu_13_neris", "lcld_v2_time"]:
            from constrained_attacks import datasets
            _, y = datasets.load_dataset(dataset).get_x_y()
            y = np.array(y)
            y_class, y_occ = np.unique(y, return_counts=True)
            y_occ = (y_occ.max() -y_occ + y_occ.mean())
            self.weights = dict(zip(y_class, y_occ/y_occ.max()))

        super(BalancedBCELossPytorch, self).__init__(**args)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super(BalancedBCELossPytorch, self).forward(input, target.float())
        if self.weights is None:
            return loss

        positive_loss = target * loss
        negative_loss = (1 - target) * loss

        return positive_loss.sum() * self.weights.get(1) + negative_loss.sum() * self.weights.get(0)

