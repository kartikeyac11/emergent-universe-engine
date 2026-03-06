from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ComponentStats:
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    mean_signal: float
    mean_heat: float
    mean_free_energy: float
    mean_biomass: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def orthogonal_matrix(size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    raw = torch.randn((size, size), device=device, dtype=dtype)
    q, _ = torch.linalg.qr(raw)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def gaussian_sources(coords_x: torch.Tensor, coords_y: torch.Tensor) -> torch.Tensor:
    sources = torch.zeros_like(coords_x)
    stars = [
        (-0.72, -0.35, 0.15, 1.20),
        (0.60, -0.54, 0.22, 0.92),
        (-0.10, 0.64, 0.27, 0.88),
        (0.74, 0.18, 0.18, 0.76),
    ]
    for cx, cy, sigma, amp in stars:
        radius2 = (coords_x - cx) ** 2 + (coords_y - cy) ** 2
        sources = sources + amp * torch.exp(-radius2 / (2 * sigma * sigma))
    return sources.clamp(0.0, 1.8)


def laplace_kernel(device: torch.device, channels: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    return kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)


def laplace(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padded = F.pad(x, (1, 1, 1, 1), mode="circular")
    return F.conv2d(padded, kernel, groups=x.shape[1])


def blur_rgb(rgb: torch.Tensor, radius: int = 5) -> torch.Tensor:
    radius = max(1, radius)
    kernel = torch.ones((3, 1, radius, radius), dtype=rgb.dtype, device=rgb.device)
    kernel = kernel / float(radius * radius)
    padded = F.pad(rgb, (radius // 2, radius // 2, radius // 2, radius // 2), mode="reflect")
    return F.conv2d(padded, kernel, groups=3)


def hsv_to_rgb_torch(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h = h.remainder(1.0)
    i = torch.floor(h * 6.0).to(torch.int64) % 6
    f = h * 6.0 - torch.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)

    masks = [i == idx for idx in range(6)]
    r = torch.where(masks[0], v, r)
    g = torch.where(masks[0], t, g)
    b = torch.where(masks[0], p, b)

    r = torch.where(masks[1], q, r)
    g = torch.where(masks[1], v, g)
    b = torch.where(masks[1], p, b)

    r = torch.where(masks[2], p, r)
    g = torch.where(masks[2], v, g)
    b = torch.where(masks[2], t, b)

    r = torch.where(masks[3], p, r)
    g = torch.where(masks[3], q, g)
    b = torch.where(masks[3], v, b)

    r = torch.where(masks[4], t, r)
    g = torch.where(masks[4], p, g)
    b = torch.where(masks[4], v, b)

    r = torch.where(masks[5], v, r)
    g = torch.where(masks[5], p, g)
    b = torch.where(masks[5], q, b)
    return torch.cat([r, g, b], dim=1)


def normalized_position(size: int, x: float, y: float) -> tuple[float, float]:
    nx = (x / max(1.0, size - 1.0)) * 2.0 - 1.0
    ny = (y / max(1.0, size - 1.0)) * 2.0 - 1.0
    return nx, ny


def gaussian_patch(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    x: float,
    y: float,
    sigma: float,
    amplitude: float,
) -> torch.Tensor:
    radius2 = (grid_x - x) ** 2 + (grid_y - y) ** 2
    return amplitude * torch.exp(-radius2 / max(2.0 * sigma * sigma, 1e-4))


def deposit_gaussian(
    field: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    x_idx: float,
    y_idx: float,
    amplitude: float,
    sigma: float,
) -> torch.Tensor:
    size = field.shape[-1]
    nx, ny = normalized_position(size, x_idx, y_idx)
    patch = gaussian_patch(grid_x, grid_y, nx, ny, sigma, amplitude)
    if field.shape[1] == 1:
        return field + patch.view(1, 1, size, size)
    return field + patch.view(1, 1, size, size).repeat(1, field.shape[1], 1, 1)


def clamp_position(size: int, x: float, y: float) -> tuple[float, float]:
    return (float(np.clip(x, 0.0, size - 1.0)), float(np.clip(y, 0.0, size - 1.0)))


def detect_components(
    mask: np.ndarray,
    signal: np.ndarray,
    heat: np.ndarray,
    free_energy: np.ndarray,
    biomass: np.ndarray,
    min_area: int = 18,
) -> list[ComponentStats]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    components: list[ComponentStats] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = deque([(x, y)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            x_min = x_max = x
            y_min = y_max = y

            while queue:
                cx, cy = queue.popleft()
                pixels.append((cx, cy))
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

            if len(pixels) < min_area:
                continue

            xs = np.array([pixel[0] for pixel in pixels], dtype=np.float32)
            ys = np.array([pixel[1] for pixel in pixels], dtype=np.float32)
            components.append(
                ComponentStats(
                    centroid=(float(xs.mean()), float(ys.mean())),
                    bbox=(x_min, y_min, x_max, y_max),
                    area=len(pixels),
                    mean_signal=float(np.mean([signal[py, px] for px, py in pixels])),
                    mean_heat=float(np.mean([heat[py, px] for px, py in pixels])),
                    mean_free_energy=float(np.mean([free_energy[py, px] for px, py in pixels])),
                    mean_biomass=float(np.mean([biomass[py, px] for px, py in pixels])),
                )
            )

    components.sort(key=lambda component: component.area, reverse=True)
    return components


def sparkline_points(values: list[float], left: int, top: int, width: int, height: int) -> list[tuple[int, int]]:
    if len(values) < 2:
        return []
    data = np.array(values[-180:], dtype=np.float32)
    v_min = float(data.min())
    v_max = float(data.max())
    span = max(v_max - v_min, 1e-6)
    xs = np.linspace(left, left + width, num=data.shape[0], dtype=np.float32)
    ys = top + height - ((data - v_min) / span) * height
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def shannon_entropy(values: torch.Tensor) -> float:
    probs = values / (values.sum() + 1e-6)
    return float(-(probs * torch.log(probs + 1e-8)).sum().item() / math.log(probs.numel()))

