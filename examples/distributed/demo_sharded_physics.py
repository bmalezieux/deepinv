r"""
Pre-sharded distributed physics
===============================

This example adopts physics and measurements that were already sharded by the
application. It is useful when constructing the full operator stack on every
rank would be expensive or impossible, such as independent radio-interferometry
w-planes.

Run on one process::

    python examples/distributed/demo_sharded_physics.py

or on several processes::

    torchrun --standalone --nproc_per_node=2 examples/distributed/demo_sharded_physics.py
"""

import torch

from deepinv.distributed import DistributedContext, distribute
from deepinv.optim.data_fidelity import L2
from deepinv.physics import LinearPhysics


class ScaledPhysics(LinearPhysics):
    """Small linear operator representing one independently built plane."""

    def __init__(self, scale: float):
        self.scale = scale
        super().__init__(A=self._A, A_adjoint=self._A_adjoint)

    def _A(self, x):
        return self.scale * x

    def _A_adjoint(self, y):
        return self.scale * y


with DistributedContext(device_mode="cpu", seed=0, seed_offset=False) as ctx:
    num_operators = 4

    # The application owns the partition. Reversing the index before assigning
    # it makes this deliberately different from DeepInv's round-robin rule.
    global_indices = [
        index
        for index in range(num_operators)
        if (num_operators - 1 - index) % ctx.inner_world_size == ctx.inner_rank
    ]

    # Only local operators are constructed. Real applications can do this in a
    # CPU DataLoader worker and return the same plain index metadata.
    local_physics = [ScaledPhysics(index + 1.0) for index in global_indices]
    if not local_physics:
        local_physics, supplied_indices = None, None
    else:
        supplied_indices = global_indices

    physics = distribute(
        local_physics,
        ctx,
        type_object="linear_physics",
        from_shard=True,
        num_operators=num_operators,
        global_indices=supplied_indices,
    )

    x_true = torch.full((1, 1, 4, 4), 2.0, device=ctx.device)
    x = torch.zeros_like(x_true, requires_grad=True)
    local_y = [operator.A(x_true) for operator in physics.local_physics]
    if not local_y:
        local_y = None

    data_fidelity = distribute(L2(), ctx)
    gradient = data_fidelity.grad(x, local_y, physics)
    loss = data_fidelity.fn(x, local_y, physics)
    loss.backward()

    # Gathering is optional and restores global operator order.
    gathered_prediction = physics.A(x_true, gather=True)
    expected_gradient = -2.0 * sum(scale**2 for scale in range(1, 5))
    assert len(gathered_prediction) == num_operators
    assert torch.allclose(gradient, torch.full_like(x, expected_gradient))
    assert torch.allclose(x.grad, gradient)

    if ctx.is_global_main:
        print("Global operator order:", list(range(num_operators)))
        print("Reduced gradient value:", gradient[0, 0, 0, 0].item())
