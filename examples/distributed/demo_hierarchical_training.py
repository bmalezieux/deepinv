"""
Hierarchical Distributed Training
---------------------------------

This example combines DeepInv's spatial distribution with PyTorch Distributed
Data Parallel (DDP). Processes in each row cooperate to denoise one image using
tiles, while corresponding processes across rows train on independent images and
synchronize gradients with DDP.

Run four processes with two processes per inverse problem:

.. code-block:: bash

    torchrun --standalone --nproc_per_node=4 \
        examples/distributed/demo_hierarchical_training.py --inner-world-size 2

This creates two data-parallel replicas. The example is deliberately small and
uses synthetic data, so it also runs with Gloo on a CPU-only machine.
"""

# %%
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from deepinv.distributed import DistributedContext, distribute
from deepinv.models import Denoiser


class TinyDenoiser(Denoiser):
    """Small trainable denoiser used to keep the example self-contained."""

    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, sigma=None):
        return self.network(x)


def make_dataset(num_images=8, image_size=64, seed=0):
    """Create deterministic clean/noisy pairs on the CPU."""
    generator = torch.Generator().manual_seed(seed)
    clean = torch.rand(num_images, 1, image_size, image_size, generator=generator)
    noisy = clean + 0.1 * torch.randn(clean.shape, generator=generator)
    return TensorDataset(noisy, clean)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--inner-world-size",
    type=int,
    default=1,
    help="Number of processes that cooperate on each image.",
)
args, _ = parser.parse_known_args()

# %%
# ``inner_world_size`` determines the two-dimensional process topology. With
# four launched processes and a value of two, there are two independent DDP
# replicas and two spatial ranks per image.
with DistributedContext(inner_world_size=args.inner_world_size, seed=0) as ctx:
    dataset = make_dataset()
    sampler = ctx.distributed_data_sampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)

    model = TinyDenoiser().to(ctx.device)

    # First distribute the computation for one image over each inner row.
    model = distribute(
        model,
        ctx,
        patch_size=32,
        overlap=4,
        tiling_dims=(2, 3),
        max_batch_size=1,
    )

    # Then let PyTorch DDP synchronize independent samples across columns.
    model = ctx.distributed_data_parallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):
        # Required for deterministic reshuffling by DistributedSampler.
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for noisy, clean in loader:
            noisy = noisy.to(ctx.device)
            clean = clean.to(ctx.device)

            optimizer.zero_grad()
            denoised = model(noisy, sigma=0.1)
            loss = torch.nn.functional.mse_loss(denoised, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if ctx.is_global_main:
            print(
                f"epoch={epoch} loss={epoch_loss / len(loader):.4f} "
                f"inner_world_size={ctx.inner_world_size} "
                f"dp_world_size={ctx.dp_world_size}"
            )
