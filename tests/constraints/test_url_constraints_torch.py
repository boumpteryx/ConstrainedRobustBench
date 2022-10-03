import numpy as np
import torch

from constraints.constraints_executor import (
    PytorchConstraintsExecutor,
)
from constrained_attacks.constraints.relation_constraint import AndConstraint

from tests.attacks.moeva.url_constraints import get_url_constraints


def test_torch_constraints():
    constraints = get_url_constraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")
    x_clean = torch.tensor(x_clean[:2], dtype=torch.float32)
    executor = PytorchConstraintsExecutor(
        AndConstraint(constraints.relation_constraints)
    )
    c_eval = executor.execute(x_clean)
    assert torch.all(c_eval == 0)
