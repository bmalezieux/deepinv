"""
Understanding Image Tiling (SmartTilingStrategy)
===============================================

This script visualizes how `SmartTilingStrategy` extracts padded windows,inner crops, 
and how stride/receptive fields relate. The script follows three conceptual stages:

1. Load and optionally rescale a demo image.
2. Instantiate :class:`SmartTilingStrategy` with the chosen patch size and
   receptive-field size (radius), then print the metadata for reference.
3. Generate five companion figures (01–05) that document every step of the
   tiling workflow:
   - 01 – raw image only (baseline reference).
   - 02 – original versus globally padded image.
   - 03 – padded image with the first window/crop overlay.
   - 04 – comparison of two neighbouring patches.
   - 05 – full grid of windows.

Usage:
    python examples/distributed/demo_smart_tiling.py

Outputs are written to ``examples/distributed/image_tiling/`` so you can
inspect the PNGs after the run.
"""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Optional, List
from matplotlib.patches import Rectangle

from deepinv.distributed.strategies import SmartTilingStrategy
from deepinv.utils.demo import load_example

Index = tuple[slice, ...]


# The functions below are utilities that format tensors and metadata so the
# figures closely mirror the behaviour of SmartTilingStrategy. They do not
# modify the algorithm; they only prepare visuals to understand each step.


def _to_display_image(t: torch.Tensor) -> torch.Tensor:
    """Convert high-dimensional tensors into a Matplotlib-friendly image.

    The SmartTilingStrategy can produce batches of 2D or 3D volumes with shape
    `(B, C, H, W)` or `(B, C, D, H, W)`. For visualisation we only display the
    first batch element and, in the 3D case, the middle depth slice. The output
    is clamped to `[0, 1]` and permuted to the `(H, W, C)` convention when
    necessary so that `plt.imshow` renders the tensor without further tweaks.
    """
    x = t
    if x.ndim == 5:
        x = x[0]; mid = x.shape[1] // 2; x = x[:, mid]
    elif x.ndim == 4:
        x = x[0]
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim == 3:
        x = x[0] if x.shape[0] == 1 else x.permute(1, 2, 0)
    return x


def _trim_pad(pads: tuple[int, ...]) -> tuple[int, ...]:
    """Compress padding tuples so they satisfy ``torch.nn.functional.pad``.

    SmartTilingStrategy records padding for every spatial dimension, often
    including trailing zero pairs for dimensions that were already aligned. The
    functional padding API expects those redundant pairs to be stripped; this
    helper pops them off while preserving the intended order of pad values.
    """
    pad_list = list(pads)
    while len(pad_list) >= 2 and pad_list[-1] == 0 and pad_list[-2] == 0:
        pad_list.pop(); pad_list.pop()
    return tuple(pad_list)


def plot_original_only(clean_image: torch.Tensor, out_dir: str) -> None:
    """Save figure 01 containing only the raw input image.

    Parameters
    ----------
    clean_image:
        Tensor representing the full-resolution sample before tiling.
    out_dir:
        Directory where the resulting PNG should be written.
    """
    # Figure 01: baseline reference of the raw image before any padding or tiling.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(_to_display_image(clean_image))
    ax.set_title(f"Original {tuple(clean_image.shape)}", fontsize=20, fontweight="bold")
    ax.axis("off")
    fig.savefig(os.path.join(out_dir, "01_original.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_original_and_padded(clean_image: torch.Tensor, metadata: dict, out_dir: str) -> None:
    """Save figure 02 comparing the unpadded and globally padded images.

    The padded image comes from applying ``metadata['global_padding']`` in the
    same way SmartTilingStrategy prepares data prior to extracting windows.
    Additional overlays highlight the region corresponding to the original
    content so padding margins are obvious to the reader.
    """
    
    # Figure 02: show how SmartTilingStrategy pads the input before extracting patches.
    pads = _trim_pad(metadata["global_padding"])
    padded = F.pad(clean_image, pads, mode=metadata["pad_mode"])

    # Compute display sizes directly from image dimensions so the padded view scales both axes.
    base_display_width = 6.0; base_scale = base_display_width / clean_image.shape[-1]
    orig_w_in, orig_h_in, padded_w_in, padded_h_in = (clean_image.shape[-1] * base_scale, clean_image.shape[-2] * base_scale, padded.shape[-1] * base_scale, padded.shape[-2] * base_scale)

    left_margin_in, right_margin_in, bottom_margin_in, top_margin_in, gap_in = 0.6, 0.4, 0.6, 1.0, 0.8
    content_height_in = max(orig_h_in, padded_h_in)

    fig_width = left_margin_in + orig_w_in + gap_in + padded_w_in + right_margin_in; fig_height = bottom_margin_in + content_height_in + top_margin_in

    fig = plt.figure(figsize=(fig_width, fig_height))

    orig_bottom_in = bottom_margin_in + (content_height_in - orig_h_in) / 2; padded_bottom_in = bottom_margin_in + (content_height_in - padded_h_in) / 2  # vertically center axes

    ax0 = fig.add_axes([left_margin_in / fig_width, orig_bottom_in / fig_height, orig_w_in / fig_width, orig_h_in / fig_height]); ax0.imshow(_to_display_image(clean_image))
    ax0.set_title(f"Original {tuple(clean_image.shape)}", fontsize=20, fontweight="bold"); ax0.axis("off"); ax0.set_aspect("auto")

    ax1 = fig.add_axes([(left_margin_in + orig_w_in + gap_in) / fig_width, padded_bottom_in / fig_height, padded_w_in / fig_width, padded_h_in / fig_height]); ax1.imshow(_to_display_image(padded))

    # Highlight padding area with a translucent overlay and show original extent
    if len(pads) >= 4:
        w_left, w_right, h_top, h_bottom = pads[0], pads[1], pads[2], pads[3]
        orig_h, orig_w = clean_image.shape[-2], clean_image.shape[-1]
        pad_h, pad_w = padded.shape[-2], padded.shape[-1]

        mask = torch.zeros((pad_h, pad_w), device=padded.device, dtype=padded.dtype)  # mark padded regions
        for cond, sl in (
            (h_top, (slice(None, h_top), slice(None))),
            (h_bottom, (slice(pad_h - h_bottom, pad_h), slice(None))),
            (w_left, (slice(None), slice(None, w_left))),
            (w_right, (slice(None), slice(pad_w - w_right, pad_w))),
        ):
            if cond:
                mask[sl] = 1

        ax1.imshow(mask.detach().cpu(), cmap="Reds", alpha=0.25)

        ax1.add_patch(Rectangle((w_left, h_top), orig_w, orig_h, fill=False, edgecolor="yellow", linewidth=2, linestyle="-", alpha=0.9))
        ax1.text(w_left + orig_w * 0.5, h_top - 5, "original region", color="yellow", ha="center", va="bottom", fontsize=15, bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax1.set_title(f"Padded {tuple(padded.shape)} pad_mode={metadata['pad_mode']}", fontsize=20, fontweight="bold"); ax1.axis("off"); ax1.set_aspect("auto")

    fig.savefig(os.path.join(out_dir, "02_original_vs_padded.png"), dpi=120, bbox_inches=None, pad_inches=0); plt.close(fig)


def _draw_patch_overlay(ax, g_sl: Index, c_sl: Index, rf_size, label: str, color_full: str, color_inner: str, linewidth: float = 2.5):
    """Render a window/inner-crop overlay corresponding to a specific patch.

    Parameters mirror the metadata SmartTilingStrategy exposes: ``g_sl`` is the
    window slice in the padded coordinates, ``c_sl`` is the inner crop within
    that window, and ``rf_size`` is the receptive-field size (radius). The
    function only draws annotations; it leaves the underlying image untouched.
    """
    # Draw dashed rectangles for full windows and solid rectangles for inner crops.
    h_sl, w_sl = g_sl[-2], g_sl[-1]
    x0, y0 = w_sl.start, h_sl.start
    w, h = w_sl.stop - w_sl.start, h_sl.stop - h_sl.start
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color_full, linestyle="--", linewidth=linewidth, clip_on=False))
    y_inner0, y_inner1 = h_sl.start + c_sl[-2].start, h_sl.start + c_sl[-2].stop
    x_inner0, x_inner1 = w_sl.start + c_sl[-1].start, w_sl.start + c_sl[-1].stop
    ax.add_patch(Rectangle((x_inner0, y_inner0), x_inner1 - x_inner0, y_inner1 - y_inner0, fill=False, edgecolor=color_inner, linestyle="-", linewidth=linewidth, clip_on=False))
    ax.text(x0 + w * 0.5, y0 + h * 0.5, label, color="white", ha="center", va="center", fontsize=12, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, pad=3, boxstyle="round,pad=0.2"), clip_on=False)


def _annotate_lengths(ax, g_sl: Index, c_sl: Index, rf_size, patch_size, window_shape):
    """Label window width, patch size, and receptive field  for a patch.

    This augments the dashed/solid rectangles emitted by ``_draw_patch_overlay``
    with arrows describing how much padding and overlap surround the inner crop.
    ``window_shape`` provides the receptive-field expansion, while
    ``patch_size`` notes the target tile passed to the downstream model.
    """
    # Add arrows that describe window width, target patch size, and receptive field overlap.
    h_sl, w_sl = g_sl[-2], g_sl[-1]
    x0, y0 = w_sl.start, h_sl.start
    w, h = w_sl.stop - w_sl.start, h_sl.stop - h_sl.start
    rf_w = rf_size if isinstance(rf_size, int) else rf_size[1]
    rf_h = rf_size if isinstance(rf_size, int) else rf_size[0]

    y_top = y0
    ax.annotate("", xy=(x0 + w, y_top), xytext=(x0, y_top), arrowprops=dict(arrowstyle="<->", color="red", linewidth=2.5), annotation_clip=False)  # window width
    ax.text(x0 + w / 2, y_top - 5, f"window {window_shape}", color="red", fontsize=13, fontweight="bold", ha="center", va="bottom", clip_on=False)

    inner_y = h_sl.start + c_sl[-2].start
    inner_start, inner_end = x0 + c_sl[-1].start, x0 + c_sl[-1].stop
    inner_w, inner_h = c_sl[-1].stop - c_sl[-1].start, c_sl[-2].stop - c_sl[-2].start
    if isinstance(patch_size, int):
        target_h = target_w = patch_size
    else:
        target_h, target_w = patch_size[0], patch_size[1] if len(patch_size) > 1 else patch_size[0]

    inner_matches_target = inner_w == target_w and inner_h == target_h

    if inner_matches_target:
        ax.annotate("", xy=(inner_end, inner_y), xytext=(inner_start, inner_y), arrowprops=dict(arrowstyle="<->", color="cyan", linewidth=2.5), annotation_clip=False)
        ax.text((inner_start + inner_end) / 2, inner_y - 6, f"patch size {target_h}x{target_w}", color="cyan", fontsize=13, fontweight="bold", ha="center", va="bottom", clip_on=False, bbox=dict(facecolor="black", alpha=0.6, pad=2, boxstyle="round,pad=0.2"))
    else:
        patch_start = x0 + rf_w
        patch_end = min(patch_start + target_w, x0 + w)
        patch_y = max(y0 , inner_y )
        ax.annotate("", xy=(patch_end, patch_y), xytext=(patch_start, patch_y), arrowprops=dict(arrowstyle="<->", color="gold", linewidth=2.5), annotation_clip=False)
        ax.text((patch_start + patch_end) / 2, patch_y - 6, f"patch size {target_h}x{target_w}", color="gold", fontsize=13, fontweight="bold", ha="center", va="bottom", clip_on=False, bbox=dict(facecolor="black", alpha=0.6, pad=2, boxstyle="round,pad=0.2"))

        ax.annotate("", xy=(inner_end, inner_y+target_h), xytext=(inner_start, inner_y+target_h), arrowprops=dict(arrowstyle="<->", color="cyan", linewidth=2.5), annotation_clip=False)
        ax.text((inner_start + inner_end) / 2, inner_y + target_h, f"inner crop {inner_h}x{inner_w}", color="cyan", fontsize=13, fontweight="bold", ha="center", va="bottom", clip_on=False, bbox=dict(facecolor="black", alpha=0.6, pad=2, boxstyle="round,pad=0.2"))

    rf_y = y0 + rf_h
    ax.annotate("", xy=(x0 + rf_w, rf_y), xytext=(x0, rf_y), arrowprops=dict(arrowstyle="<->", color="magenta", linewidth=2.5, zorder=10), annotation_clip=False, zorder=12)  # left RF padding

    right_rf_start = x0 + c_sl[-1].stop
    ax.annotate("", xy=(right_rf_start + rf_w, rf_y), xytext=(right_rf_start, rf_y), arrowprops=dict(arrowstyle="<->", color="magenta", linewidth=2.5, zorder=10), annotation_clip=False, zorder=12)

    rf_x = x0
    ax.annotate("", xy=(rf_x, y0 + rf_h), xytext=(rf_x, y0), arrowprops=dict(arrowstyle="<->", color="magenta", linewidth=2.5), annotation_clip=False)
    ax.text(rf_x - 8, y0 + rf_h / 2, f"RF size {rf_h}", color="magenta", fontsize=13, fontweight="bold", ha="right", va="center", rotation=90, clip_on=False)

   


def plot_original_vs_padded_with_patches(clean_image: torch.Tensor,padded: torch.Tensor, metadata: dict,
    out_dir: str,patch_indices: list[int],fname: str, modes: Optional[List[str]] = None,
    window_colors: Optional[List[str]] = None, inner_colors: Optional[List[str]] = None,
    show_global_stride_arrow: bool = False,) -> None:
    """Save figures 03/04 highlighting selected patch windows on the padded image.

    Parameters
    ----------
    clean_image, padded:
        The original tensor and its padded counterpart for side-by-side display.
    metadata:
        SmartTilingStrategy metadata containing global and crop slices.
    patch_indices:
        Sequence of patch indices to highlight.
    fname:
        Output filename (e.g., ``03_padded_with_patch1.png``).
    modes / window_colors / inner_colors:
        Optional per-patch configuration controlling whether a given overlay
        draws only the window or both window and inner crop with custom colours.
    show_global_stride_arrow:
        When ``True`` the figure annotates the horizontal stride between the
        first two requested patches, falling back to metadata stride otherwise.
    """
    # Figures 03 and 04: highlight selected patches and optionally expose stride spacing.
    
    global_slices, crop_slices = metadata["global_slices"], metadata["crop_slices"]
    rf_size, patch_size, window_shape = (metadata[k] for k in ("receptive_field_size", "inner_patch_size", "window_shape"))
    
    # Compute display sizes from the image dimensions to grow the padded subplot in both axes.
    base_display_width = 6.0
    base_scale = base_display_width / clean_image.shape[-1]
    orig_w_in = clean_image.shape[-1] * base_scale
    orig_h_in = clean_image.shape[-2] * base_scale
    padded_w_in = padded.shape[-1] * base_scale
    padded_h_in = padded.shape[-2] * base_scale

    left_margin_in = 0.6
    right_margin_in = 0.4
    bottom_margin_in = 0.6
    top_margin_in = 1.0
    gap_in = 0.8
    content_height_in = max(orig_h_in, padded_h_in)

    fig_width = left_margin_in + orig_w_in + gap_in + padded_w_in + right_margin_in
    fig_height = bottom_margin_in + content_height_in + top_margin_in

    fig = plt.figure(figsize=(fig_width, fig_height))

    orig_bottom_in = bottom_margin_in + (content_height_in - orig_h_in) / 2
    padded_bottom_in = bottom_margin_in + (content_height_in - padded_h_in) / 2

    ax0 = fig.add_axes([left_margin_in / fig_width, orig_bottom_in / fig_height, orig_w_in / fig_width, orig_h_in / fig_height])
    ax0.imshow(_to_display_image(clean_image))
    ax0.set_title("Original", fontsize=20, fontweight="bold")
    ax0.axis("off")
    ax0.set_aspect("auto")

    ax1 = fig.add_axes([(left_margin_in + orig_w_in + gap_in) / fig_width, padded_bottom_in / fig_height, padded_w_in / fig_width, padded_h_in / fig_height])
    ax1.imshow(_to_display_image(padded))
    ax1.set_title("Padded with patch overlays\n", fontsize=20, fontweight="bold")
    
    # Overlays for requested patches (support per-patch modes)
    # modes options per patch: 'window-only' | 'full'
    colors = [("red", "cyan"), ("orange", "lime")]
    for i, p_idx in enumerate(patch_indices):
        g_sl = global_slices[p_idx]
        c_sl = crop_slices[p_idx]
        mode = (modes[i] if modes and i < len(modes) else "full")

        # Resolve colors (allow overrides)
        default_full, default_inner = colors[i % len(colors)]
        win_col = window_colors[i] if (window_colors and i < len(window_colors) and window_colors[i]) else default_full
        in_col = inner_colors[i] if (inner_colors and i < len(inner_colors) and inner_colors[i]) else default_inner

        if mode == "window-only":
            h_sl, w_sl = g_sl[-2], g_sl[-1]
            ax1.add_patch(Rectangle((w_sl.start, h_sl.start), w_sl.stop - w_sl.start, h_sl.stop - h_sl.start, fill=False, edgecolor=win_col, linestyle="--", linewidth=2.5))
            continue

        # Full patch overlay with annotations (no stride/target)
        _draw_patch_overlay(ax1, g_sl, c_sl, rf_size, f"patch {p_idx+1}", win_col, in_col)
        _annotate_lengths(ax1, g_sl, c_sl, rf_size, patch_size, window_shape)
    
    # Optional global stride arrow at the very top of the padded image (horizontal)
    if show_global_stride_arrow:
        stride_pixels = None
        arrow_start = 0

        # Prefer computing stride from the first two provided patches so it mirrors the actual layout.
        if len(patch_indices) >= 2:
            sorted_by_x = sorted(patch_indices, key=lambda idx: (global_slices[idx][-1].start or 0))
            left0 = global_slices[sorted_by_x[0]][-1].start or 0
            left1 = global_slices[sorted_by_x[1]][-1].start or 0
            stride_pixels = abs(left1 - left0)
            arrow_start = min(left0, left1)

        # Fall back to metadata stride when we cannot infer it from slices (e.g., single patch selection).
        if stride_pixels in (None, 0):
            s = metadata.get("stride")
            if s is not None:
                stride_pixels = s[-1] if isinstance(s, (tuple, list)) else int(s)
                arrow_start = 0

        if stride_pixels and stride_pixels > 0:
            ytop = min((global_slices[p][-2].start or 0) for p in patch_indices)
            arrow_end = arrow_start + stride_pixels
            ax1.annotate("", xy=(arrow_end, ytop), xytext=(arrow_start, ytop), arrowprops=dict(arrowstyle="<->", color="royalblue", linewidth=2.5), annotation_clip=False, zorder=20)
            ax1.text((arrow_start + arrow_end) / 2, ytop - 6, f"stride {stride_pixels}", color="royalblue", fontsize=13, fontweight="bold", ha="center", va="bottom", clip_on=False, zorder=20)

    ax1.set_aspect("auto")
    ax1.axis("off")

    fig.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def plot_patch_grid_overlay(padded: torch.Tensor, metadata: dict, out_dir: str) -> None:
    """Save figure 05 containing a grid of every padded window and inner crop.

    Each subplot visualises a distinct window produced by SmartTilingStrategy.
    The legend matches the line styles used elsewhere so readers can relate the
    figure to the annotations drawn on the padded overview plots.
    """
    # Figure 05: mosaic of every window to inspect receptive-field overlaps at once.
    global_slices = metadata["global_slices"]
    crop_slices = metadata["crop_slices"]
    rf_size = metadata["receptive_field_size"]
    
    num_patches = len(global_slices)
    grid_shape = metadata["grid_shape"]
    
    # Create subplots with spacing for each patch
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(14, 14))
    if num_patches == 1:
        axes = [[axes]]
    elif grid_shape[0] == 1 or grid_shape[1] == 1:
        axes = axes.reshape(grid_shape)
    else:
        axes = axes.reshape(grid_shape)

    for idx, g_sl in enumerate(global_slices):
        row = idx // grid_shape[1]
        col = idx % grid_shape[1]
        ax = axes[row, col] if isinstance(axes[row], list) else axes[row, col]
        
        h_sl, w_sl = g_sl[-2], g_sl[-1]
        patch_slice = (...,) + (h_sl, w_sl)
        patch_img = padded[patch_slice]
        
        ax.imshow(_to_display_image(patch_img))
        
        c_sl = crop_slices[idx]
        y_inner0, y_inner1 = c_sl[-2].start, c_sl[-2].stop
        x_inner0, x_inner1 = c_sl[-1].start, c_sl[-1].stop
        window_h, window_w = patch_img.shape[-2], patch_img.shape[-1]
        
        # Red dashed rectangle: full window
        ax.add_patch(Rectangle((0, 0), window_w, window_h, fill=False, edgecolor="red", linestyle="--", linewidth=2.5))
        
        # Lime solid rectangle: inner crop
        ax.add_patch(Rectangle((x_inner0, y_inner0), x_inner1 - x_inner0, y_inner1 - y_inner0, fill=False, edgecolor="lime", linestyle="-", linewidth=2.5))
        

        ax.grid(True, alpha=0.2, linestyle=":")
        ax.set_xticks([])
        ax.set_yticks([])

    # Add legend to first subplot
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color="red", linestyle="--", linewidth=2.5, label="Full window"), Line2D([0], [0], color="lime", linestyle="-", linewidth=2.5, label="Inner crop")]
    axes.flat[0].legend(handles=legend_elements, loc="upper left", fontsize=15, framealpha=0.95)

    fig.suptitle("Patch Grid with Receptive Field \n" f"Grid: {metadata['grid_shape']} | Window: {metadata['window_shape']} | RF size: {rf_size}", fontsize=24, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_patch_grid_overlay.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    


# The helpers above prepare the visual elements. The functions below drive the
# main SmartTilingStrategy demonstration from I/O to figure generation.

def load_image(device: torch.device, img_size: int = 1024):
    """Load the butterfly demo image and optionally rescale it for visualisation.

    Parameters
    ----------
    device:
        Torch device on which the tensor should reside.
    img_size:
        Maximum allowed height or width. The image is resized isotropically so
        the largest dimension matches this value, ensuring figures stay a
        manageable size while preserving aspect ratio.
    """
    
    img = load_example("butterfly.png", grayscale=False, device=device)
    # can be another image with different size    
    #img = load_example("CBSD_0010.png", grayscale=False, device=device)

    _, _, h, w = img.shape
    max_dim = max(h, w)
    if max_dim == img_size:
        return img
    scale = img_size / max_dim
    new_h, new_w = int(h * scale), int(w * scale)
    return F.interpolate(img, size=(new_h, new_w), mode="bicubic", align_corners=False)

# Configuration parameters
img_size = 512  # Large image for demonstrating tiling
patch_size = 256  # Size of each patch
receptive_field_size = 64  # Overlap for smooth boundaries
pad_mode = "reflect"  # Padding mode for patch extraction

def main():
    """Run the SmartTilingStrategy demo and persist all generated figures."""
    # Orchestrates the demo: load image, instantiate SmartTilingStrategy, then create figures.
    device = torch.device("cpu")
    clean_image = load_image(device, img_size=img_size)

    smart_tiling = SmartTilingStrategy(
        signal_shape=clean_image.shape,
        tiling_dims=len(clean_image.shape) - 2,
        patch_size=patch_size,
        receptive_field_size=receptive_field_size,
        pad_mode=pad_mode,
    )

    num_patches = smart_tiling.get_num_patches()
    metadata = {**smart_tiling._metadata, "global_slices": smart_tiling._global_slices}
    # Print key tiling metadata (as before)
    print("=" * 70)
    print("SmartTilingStrategy - Visualization")
    print("=" * 70)
    print(f"Device: {device}\nImage shape: {clean_image.shape}\n")
    print("Tiling metadata")
    for label, value in (
        ("grid_shape", metadata["grid_shape"]),
        ("window_shape (patch+2*rf)", metadata["window_shape"]),
        ("inner_patch_size", metadata["inner_patch_size"]),
        ("receptive_field_size", metadata["receptive_field_size"]),
        ("stride", metadata["stride"]),
        ("global_padding", metadata["global_padding"]),
        ("tiling_dims", metadata["tiling_dims"]),
        ("Total number of patches", num_patches),
    ):
        print(f"{label}: {value}")

    out_dir = os.path.join(os.path.dirname(__file__), "image_tiling")
    os.makedirs(out_dir, exist_ok=True)

    padded = F.pad(clean_image, _trim_pad(metadata["global_padding"]), mode=metadata["pad_mode"]) 

    # 1) Original only
    plot_original_only(clean_image, out_dir)

    # 2) Original vs padded
    plot_original_and_padded(clean_image, metadata, out_dir)

    # 3) Original vs padded with first patch annotated (label shows 'patch 1')
    plot_original_vs_padded_with_patches(clean_image, padded, metadata, out_dir, patch_indices=[0], fname="03_padded_with_patch1.png")

    # 4) Two patches: patch 1 window-only (orange), patch 2 full (red/cyan), with global stride arrow
    patch_indices = [idx for idx in [0, 1] if idx < num_patches]
    if patch_indices:
        plot_original_vs_padded_with_patches(clean_image, padded, metadata, out_dir, patch_indices=patch_indices, fname="04_padded_with_patch1_2.png", modes=["window-only", "full"], window_colors=["orange", "red"], inner_colors=[None, "cyan"], show_global_stride_arrow=True)

    # 5) Full patch grid overlay
    plot_patch_grid_overlay(padded, metadata, out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()

