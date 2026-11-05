"""Visualization helpers for masks and overlays."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

from cystic_seg.constants import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def class_colormap(num_classes: int = NUM_CLASSES) -> ListedColormap:
    """Index i is tab10(i); class 0 (background) is black."""
    base = plt.get_cmap("tab10").colors
    colors = [(0.0, 0.0, 0.0, 1.0)]
    for i in range(1, num_classes):
        colors.append(tuple(base[i]) + (1.0,))
    return ListedColormap(colors)


def overlay_colors(num_classes: int = NUM_CLASSES) -> np.ndarray:
    cmap = plt.get_cmap("tab10")
    return (np.array([cmap(i)[:3] for i in range(num_classes)]) * 255).astype(np.uint8)


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalization. Expects CHW float array."""
    img = np.transpose(image, (1, 2, 0))
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def mask_legend_handles(mask: np.ndarray) -> list:
    handles = []
    for class_id in np.unique(mask):
        class_id = int(class_id)
        if class_id == 0:
            continue
        color = plt.get_cmap("tab10")(class_id)[:3]
        handles.append(mpatches.Patch(color=color, label=CLASS_NAMES.get(class_id, f"Class {class_id}")))
    return handles


def overlay_mask_rgb(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Blend class colors onto an RGB uint8 image (no OpenCV)."""
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    if image_rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image {image_rgb.shape} and mask {mask.shape} differ")
    colors = overlay_colors(num_classes)
    painted = image_rgb.copy()
    fg = mask > 0
    if not np.any(fg):
        return painted
    color_img = np.zeros_like(image_rgb)
    for class_id in np.unique(mask):
        class_id = int(class_id)
        if class_id == 0:
            continue
        color_img[mask == class_id] = colors[class_id]
    blended = (
        image_rgb.astype(np.float32) * (1.0 - alpha) + color_img.astype(np.float32) * alpha
    ).astype(np.uint8)
    painted[fg] = blended[fg]
    return painted


def draw_class_labels(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    class_names: dict | None = None,
    min_pixels: int = 80,
) -> np.ndarray:
    """Write class names at region centroids (Pillow, no OpenCV)."""
    from PIL import Image, ImageDraw

    names = class_names or CLASS_NAMES
    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    for class_id, (row, col) in class_centroids(mask).items():
        region = mask == class_id
        if int(region.sum()) < min_pixels:
            continue
        label = names.get(class_id, str(class_id))
        x, y = int(col), int(row)
        try:
            draw.text((x, y), label, fill=(255, 255, 255), anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
        except TypeError:
            draw.text((max(0, x - 20), max(0, y - 6)), label, fill=(255, 255, 255))
    return np.array(img)


def draw_legend(
    image_rgb: np.ndarray,
    class_names: dict,
    num_classes: int | None = None,
) -> np.ndarray:
    """Paint a small color key in the top-left corner."""
    from PIL import Image, ImageDraw

    names = class_names
    n = num_classes or (max(names) + 1)
    colors = overlay_colors(n)
    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    x0, y0 = 12, 12
    row_h = 22
    box = 14
    fg_ids = [i for i in range(1, n) if i in names]
    pad = 8
    width = 220
    height = pad * 2 + row_h * len(fg_ids)
    draw.rectangle([x0 - 4, y0 - 4, x0 + width, y0 + height], fill=(0, 0, 0))
    for k, class_id in enumerate(fg_ids):
        y = y0 + pad + k * row_h
        rgb = tuple(int(c) for c in colors[class_id])
        draw.rectangle([x0, y, x0 + box, y + box], fill=rgb)
        draw.text((x0 + box + 8, y), names[class_id], fill=(255, 255, 255))
    return np.array(img)


def resize_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of an integer mask to (width, height)."""
    from PIL import Image

    width, height = size_wh
    return np.array(Image.fromarray(mask.astype(np.uint8), mode="L").resize((width, height), Image.NEAREST))


def class_centroids(mask: np.ndarray) -> dict[int, tuple[float, float]]:
    """Return {class_id: (row, col)} for foreground regions."""
    centroids = {}
    for class_id in np.unique(mask):
        class_id = int(class_id)
        if class_id == 0:
            continue
        ys, xs = np.where(mask == class_id)
        if ys.size:
            centroids[class_id] = (float(ys.mean()), float(xs.mean()))
    return centroids


def overlay_mask_on_image(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a class mask onto a BGR image and label region centroids."""
    import cv2

    colors = overlay_colors()
    mask_rgb = np.zeros_like(image_bgr)
    for class_id in np.unique(mask):
        class_id = int(class_id)
        if class_id == 0:
            continue
        mask_rgb[mask == class_id] = colors[class_id]
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, mask_rgb, alpha, 0)
    for class_id in np.unique(mask):
        class_id = int(class_id)
        if class_id == 0:
            continue
        label = CLASS_NAMES.get(class_id, str(class_id))
        region = (mask == class_id).astype(np.uint8)
        moments = cv2.moments(region)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.putText(
                overlay,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return overlay


def save_prediction_triplet(
    image_chw: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    num_classes: int = NUM_CLASSES,
) -> None:
    img = denormalize_image(image_chw)
    cmap = class_colormap(num_classes)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    axs[1].imshow(gt, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")
    axs[2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axs[2].set_title("Prediction")
    axs[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
