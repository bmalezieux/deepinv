r"""
Distributed Training of a Shared-Weight Unrolled PnP Network
--------------------------------------------------------------

This example shows how to train an unrolled reconstruction network in a fully distributed DeepInverse setup:

- distributed physics operators,
- distributed data-fidelity gradient,
- distributed denoiser with tiling,
- trainable algorithmic step sizes and trainable denoiser weights.

We use Urban100HR as image dataset, a pretrained DRUNet as the shared denoiser (same model at all unrolled
iterations), and run only a few epochs for the sake of the demo.

**Usage:**

.. code-block:: bash

    # Single process
    python examples/distributed/demo_unrolled_distributed.py

.. code-block:: bash

    # Multi-process (2 ranks)
    python -m torch.distributed.run --nproc_per_node=2 examples/distributed/demo_unrolled_distributed.py

"""

# %%
from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

import deepinv as dinv
from deepinv.models import DRUNet
from deepinv.loss.metric import PSNR
from deepinv.optim.data_fidelity import L2
from deepinv.distributed import DistributedContext, distribute
from deepinv.physics import Denoising, GaussianNoise, stack
from deepinv.utils import get_data_home
from deepinv.utils.plotting import plot, plot_curves


def _extract_x(batch):
    # Urban100HR returns (image, class_id)
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def _reduce_weighted_scalar(
    ctx: DistributedContext, weighted_sum: float, count: int
) -> float:
    stats = torch.tensor([weighted_sum, float(count)], device=ctx.device)
    if ctx.use_dist:
        stats = ctx.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item() if stats[1] > 0 else 0.0


def build_urban100_loaders(
    data_root: Path,
    crop_size: int,
    train_images: int,
    val_images: int,
    batch_size: int,
):
    transform = Compose([ToTensor(), Resize(crop_size), CenterCrop(crop_size)])
    base = dinv.datasets.Urban100HR(root=data_root, download=True, transform=transform)

    max_images = min(len(base), train_images + val_images)
    indices = list(range(max_images))
    train_dataset = Subset(base, indices[:train_images])
    val_dataset = Subset(base, indices[train_images:max_images])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep identical ordering across ranks for model-parallel physics.
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def create_stacked_denoising_physics(
    device: torch.device, noise_levels=(0.06, 0.08), seed: int = 0
):
    physics_list = []
    for i, sigma in enumerate(noise_levels):
        rng = torch.Generator(device=device).manual_seed(seed + i)
        physics_list.append(Denoising(noise_model=GaussianNoise(sigma=sigma, rng=rng)))
    return stack(*physics_list)


class SharedUnrolledPnP(torch.nn.Module):
    def __init__(
        self,
        denoiser,
        data_fidelity,
        n_iter: int = 4,
        init_stepsize: float = 0.8,
        sigma_denoiser: float = 0.05,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.data_fidelity = data_fidelity
        self.n_iter = n_iter
        self.sigma_denoiser = sigma_denoiser
        self.log_steps = torch.nn.Parameter(
            torch.log(torch.full((n_iter,), init_stepsize))
        )

    def get_stepsizes(self):
        return torch.nn.functional.softplus(self.log_steps)

    def forward(self, y, physics):
        x = physics.A_adjoint(y)
        steps = self.get_stepsizes()
        for k in range(self.n_iter):
            grad = self.data_fidelity.grad(x, y, physics)
            x = x - steps[k] * grad
            x = self.denoiser(x, sigma=self.sigma_denoiser)
        return x


@torch.no_grad()
def evaluate_psnr(
    model: SharedUnrolledPnP,
    denoiser: torch.nn.Module,
    loader: DataLoader,
    physics,
    metric: PSNR,
    ctx: DistributedContext,
) -> float:
    model.eval()
    denoiser.eval()
    local_psnr_sum = 0.0
    local_count = 0
    for batch in loader:
        x = _extract_x(batch).to(ctx.device)
        y = physics(x)
        x_hat = model(y, physics)
        b = x.shape[0]
        local_psnr_sum += metric(x_hat, x).item() * b
        local_count += b
    return _reduce_weighted_scalar(ctx, local_psnr_sum, local_count)


# %%
# ------------------------------------
# Distributed configuration
# ------------------------------------

n_unroll = 4
epochs = 2 if torch.cuda.is_available() else 1
crop_size = 128 if torch.cuda.is_available() else 64
batch_size = 1
train_images = 24 if torch.cuda.is_available() else 8
val_images = 8 if torch.cuda.is_available() else 4
learning_rate = 1e-5
sigma_denoiser = 0.05
patch_size = crop_size // 2
overlap = max(4, patch_size // 8)


# %%
# ---------------------------------------------
# Distributed setup and training
# ---------------------------------------------

with DistributedContext(seed=0) as ctx:
    if ctx.rank == 0:
        print("=" * 78)
        print("Distributed Shared-Weight Unrolled Training Demo")
        print("=" * 78)
        print(f"Processes: {ctx.world_size}")
        print(f"Device: {ctx.device}")
        print(f"Unrolled iterations: {n_unroll}")

    # 1) Data
    data_root = get_data_home() / "Urban100"
    train_loader, val_loader = build_urban100_loaders(
        data_root=data_root,
        crop_size=crop_size,
        train_images=train_images,
        val_images=val_images,
        batch_size=batch_size,
    )

    # 2) Distributed physics and distributed data-fidelity
    stacked_physics = create_stacked_denoising_physics(ctx.device)
    distributed_physics = distribute(
        stacked_physics,
        ctx,
        type_object="linear_physics",
        reduction="mean",
    )
    distributed_data_fidelity = distribute(L2(), ctx)

    # 3) Shared pretrained denoiser + distributed wrapper
    denoiser = DRUNet(pretrained="download").to(ctx.device)
    distributed_denoiser = distribute(
        denoiser,
        ctx,
        type_object="denoiser",
        patch_size=patch_size,
        overlap=overlap,
        max_batch_size=1,
    )

    # 4) Unrolled trainable model (same denoiser at each iteration)
    model = SharedUnrolledPnP(
        denoiser=distributed_denoiser,
        data_fidelity=distributed_data_fidelity,
        n_iter=n_unroll,
        init_stepsize=0.8,
        sigma_denoiser=sigma_denoiser,
    )
    model.log_steps = distribute(model.log_steps, ctx)

    optimizer = torch.optim.Adam(
        [model.log_steps] + list(denoiser.parameters()), lr=learning_rate
    )
    mse_loss = torch.nn.MSELoss()
    psnr_metric = PSNR(reduction="mean")

    # Keep one validation image for qualitative comparison
    demo_x = _extract_x(next(iter(val_loader))).to(ctx.device)
    with torch.no_grad():
        demo_y = distributed_physics(demo_x)
        demo_rec_before = model(demo_y, distributed_physics).detach()

    init_val_psnr = evaluate_psnr(
        model, denoiser, val_loader, distributed_physics, psnr_metric, ctx
    )

    train_psnr_history = []
    val_psnr_history = [init_val_psnr]
    neg_val_psnr_history = [-init_val_psnr]

    if ctx.rank == 0:
        print(f"Initial validation PSNR: {init_val_psnr:.2f} dB")
        print("Starting training...")

    for epoch in range(epochs):
        model.train()
        denoiser.train()
        local_psnr_sum = 0.0
        local_count = 0

        for batch in train_loader:
            x = _extract_x(batch).to(ctx.device)
            y = distributed_physics(x)

            optimizer.zero_grad(set_to_none=True)
            x_hat = model(y, distributed_physics)
            loss = mse_loss(x_hat, x)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([model.log_steps], max_norm=1.0)
            optimizer.step()

            b = x.shape[0]
            local_psnr_sum += psnr_metric(x_hat.detach(), x).item() * b
            local_count += b

        train_psnr = _reduce_weighted_scalar(ctx, local_psnr_sum, local_count)
        val_psnr = evaluate_psnr(
            model, denoiser, val_loader, distributed_physics, psnr_metric, ctx
        )

        train_psnr_history.append(train_psnr)
        val_psnr_history.append(val_psnr)
        neg_val_psnr_history.append(-val_psnr)

        if ctx.rank == 0:
            steps = [f"{s:.4f}" for s in model.get_stepsizes().detach().cpu().tolist()]
            print(
                f"Epoch {epoch + 1}/{epochs} | train PSNR: {train_psnr:.2f} dB | "
                f"val PSNR: {val_psnr:.2f} dB | -val PSNR: {-val_psnr:.2f} | steps: {steps}"
            )

    with torch.no_grad():
        demo_rec_after = model(demo_y, distributed_physics).detach()

    # 5) Visualize (rank 0)
    if ctx.rank == 0:
        final_steps = [
            f"{s:.4f}" for s in model.get_stepsizes().detach().cpu().tolist()
        ]
        print(f"Final trainable step sizes: {final_steps}")
        print(
            "Note: we track `-PSNR` as a decreasing objective (equivalent to increasing PSNR)."
        )

        plot(
            [demo_x, demo_y[0], demo_rec_before, demo_rec_after],
            titles=[
                "Ground truth",
                "One noisy measurement",
                f"Before training ({init_val_psnr:.2f} dB)",
                f"After training ({val_psnr_history[-1]:.2f} dB)",
            ],
            figsize=(16, 4),
            save_fn="distributed_unrolled_result.png",
        )

        plot_curves(
            {
                "train_psnr": [train_psnr_history],
                "val_psnr": [val_psnr_history],
                "neg_val_psnr": [neg_val_psnr_history],
            }
        )

        print("Saved: distributed_unrolled_result.png")
        print("=" * 78)
