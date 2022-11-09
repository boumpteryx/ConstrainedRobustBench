import logging
import os

import configutils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from constrained_attacks.datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from tqdm import tqdm

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def compute_binary_metrics(y_true, y_score, threshold=None) -> dict:
    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score[:, 1] >= threshold).astype(int)

    metrics = {
        **classification_report(y_true, y_pred, output_dict=True),
        **{
            "roc_auc_score": roc_auc_score(y_true, y_score[:, 1]),
            "precision_score": precision_score(y_true, y_pred),
            "recall_score": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        },
    }
    return metrics


class Net(nn.Module):
    def __init__(self, preprocessor, feature_number):
        super().__init__()
        self.preprocessor = preprocessor
        self.fc0 = nn.Linear(feature_number, 64)  # first input is # of features
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)  # last input is # of classes

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, x, y, epoch, batch_size):
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 1.0]))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    batch_indexes = np.array_split(np.arange(len(x)), len(x) // batch_size + 1)

    for e in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        print(f"Epoch {e}")
        for i, index in tqdm(enumerate(batch_indexes, 0), total=len(batch_indexes)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = torch.Tensor(x[index]), torch.LongTensor(y[index])
            # print(y[index].sum())
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: " f"{running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")


def predict(net, x, y):
    # since we're not training, we don't need to calculate the gradients for
    # our outputs
    with torch.no_grad():
        inputs, labels = torch.Tensor(x), torch.LongTensor(y)
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        # _, predicted = torch.max(outputs.data, 1)
    return outputs


def run(config: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(config)
    dataset = load_dataset(config["dataset"])
    dataset.drop_date = True
    x, y = dataset.get_x_y()
    preprocessor = StandardScaler()  # dataset.get_preprocessor()
    splits = dataset.get_splits()
    preprocessor.fit(x.iloc[splits["train"]])
    x = preprocessor.transform(x)

    net = Net(preprocessor, x.shape[1])  # adapt this number for each dataset
    train(net, x[splits["train"]], y[splits["train"]], 10, 32)
    path = "./tests/resources/pytorch_models/lcld_v2_time_test_torch.pth"
    torch.save(net.state_dict(), path)
    y_scores = predict(net, x[splits["test"]], y[splits["test"]])
    print(compute_binary_metrics(y[splits["test"]], y_scores))


if __name__ == "__main__":
    run(configutils.get_config())
