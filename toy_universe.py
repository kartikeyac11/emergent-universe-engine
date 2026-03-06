from __future__ import annotations

import argparse
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pygame
import torch
import torch.nn.functional as F


@dataclass
class Config:
    grid_size: int = 192
    width: int = 1600
    height: int = 960
    steps: int = 360
    fps: int = 30
    seed: int = 7
    headless: bool = False
    save_gif: bool = True
    save_poster: bool = True
    sample_every: int = 4
    organism_refresh: int = 6
    output_dir: str = "artifacts"
    device: str = "auto"


@dataclass
class Organism:
    area: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    mean_signal: float
    mean_heat: float


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


def gaussian_sources(coords_x: torch.Tensor, coords_y: torch.Tensor) -> torch.Tensor:
    sources = torch.zeros_like(coords_x)
    stars = [
        (-0.68, -0.34, 0.16, 1.2),
        (0.58, -0.52, 0.21, 0.95),
        (-0.08, 0.62, 0.26, 0.85),
        (0.72, 0.18, 0.18, 0.72),
    ]
    for cx, cy, sigma, amp in stars:
        radius2 = (coords_x - cx) ** 2 + (coords_y - cy) ** 2
        sources = sources + amp * torch.exp(-radius2 / (2 * sigma * sigma))
    return sources.clamp(0.0, 1.8)


def laplace_kernel(device: torch.device, channels: int) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    return kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)


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

    m0 = i == 0
    m1 = i == 1
    m2 = i == 2
    m3 = i == 3
    m4 = i == 4
    m5 = i == 5

    r = torch.where(m0, v, r)
    g = torch.where(m0, t, g)
    b = torch.where(m0, p, b)

    r = torch.where(m1, q, r)
    g = torch.where(m1, v, g)
    b = torch.where(m1, p, b)

    r = torch.where(m2, p, r)
    g = torch.where(m2, v, g)
    b = torch.where(m2, t, b)

    r = torch.where(m3, p, r)
    g = torch.where(m3, q, g)
    b = torch.where(m3, v, b)

    r = torch.where(m4, t, r)
    g = torch.where(m4, p, g)
    b = torch.where(m4, v, b)

    r = torch.where(m5, v, r)
    g = torch.where(m5, p, g)
    b = torch.where(m5, q, b)

    return torch.cat([r, g, b], dim=1)


def sparkline_points(values: list[float], rect: pygame.Rect) -> list[tuple[int, int]]:
    if len(values) < 2:
        return []
    data = np.array(values[-160:], dtype=np.float32)
    v_min = float(data.min())
    v_max = float(data.max())
    span = max(v_max - v_min, 1e-6)
    xs = np.linspace(rect.left, rect.right, num=data.shape[0], dtype=np.float32)
    ys = rect.bottom - ((data - v_min) / span) * rect.height
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def detect_components(
    mask: np.ndarray,
    signal: np.ndarray,
    heat: np.ndarray,
    min_area: int = 22,
) -> list[Organism]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    organisms: list[Organism] = []

    for y in range(h):
        for x in range(w):
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
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

            if len(pixels) < min_area:
                continue

            xs = np.array([p[0] for p in pixels], dtype=np.float32)
            ys = np.array([p[1] for p in pixels], dtype=np.float32)
            signal_mean = float(np.mean([signal[py, px] for px, py in pixels]))
            heat_mean = float(np.mean([heat[py, px] for px, py in pixels]))
            organisms.append(
                Organism(
                    area=len(pixels),
                    centroid=(float(xs.mean()), float(ys.mean())),
                    bbox=(x_min, y_min, x_max, y_max),
                    mean_signal=signal_mean,
                    mean_heat=heat_mean,
                )
            )

    organisms.sort(key=lambda org: org.area, reverse=True)
    return organisms


def build_social_edges(organisms: list[Organism], world_size: int) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    limit = min(10, len(organisms))
    diag = math.sqrt(2.0) * world_size
    for i in range(limit):
        for j in range(i + 1, limit):
            ax, ay = organisms[i].centroid
            bx, by = organisms[j].centroid
            dist = math.hypot(ax - bx, ay - by) / diag
            signal_match = 1.0 - min(abs(organisms[i].mean_signal - organisms[j].mean_signal), 1.0)
            heat_penalty = 1.0 - min((organisms[i].mean_heat + organisms[j].mean_heat) * 0.4, 0.8)
            strength = (1.0 - dist * 1.6) * (0.45 + 0.55 * signal_match) * heat_penalty
            if strength > 0.2:
                edges.append((i, j, float(strength)))
    return edges


class ToyUniverse:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.size = cfg.grid_size
        self.dtype = torch.float32
        self.time = 0

        line = torch.linspace(-1.0, 1.0, self.size, device=self.device, dtype=self.dtype)
        grid_y, grid_x = torch.meshgrid(line, line, indexing="ij")
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.radius = torch.sqrt(grid_x**2 + grid_y**2)

        self.source_map = gaussian_sources(self.grid_x, self.grid_y).view(1, 1, self.size, self.size)
        self.edge_cooling = (0.12 + 0.28 * self.radius.clamp(0.0, 1.0)).view(1, 1, self.size, self.size)

        self.psi = torch.randn((1, 4, self.size, self.size), device=self.device, dtype=self.dtype) * 0.35
        self.chem = torch.zeros((1, 3, self.size, self.size), device=self.device, dtype=self.dtype)
        self.chem[:, 0] = 1.0
        self.chem[:, 1] = 0.12 * torch.exp(-((self.radius / 0.55) ** 2))
        self.chem[:, 2] = 0.18 * self.source_map[:, 0]

        self.free_energy = 0.25 + 0.55 * self.source_map
        self.heat = 0.06 * torch.ones((1, 1, self.size, self.size), device=self.device, dtype=self.dtype)
        self.signal = torch.zeros((1, 1, self.size, self.size), device=self.device, dtype=self.dtype)
        self.biomass = 0.12 * torch.rand((1, 1, self.size, self.size), device=self.device, dtype=self.dtype)
        self.trail = torch.zeros((1, 3, self.size, self.size), device=self.device, dtype=self.dtype)

        self.channel_mix = self._make_rotation_matrix()
        self.phase_bias = torch.tensor([0.0, 0.17, 0.33, 0.49], device=self.device, dtype=self.dtype)
        self.lap4 = laplace_kernel(self.device, 4)
        self.lap3 = laplace_kernel(self.device, 3)
        self.lap1 = laplace_kernel(self.device, 1)
        self.organisms: list[Organism] = []
        self.social_edges: list[tuple[int, int, float]] = []
        self.last_metrics: dict[str, float] = {}
        self.history: dict[str, list[float]] = {
            "entropy": [],
            "free_energy": [],
            "heat": [],
            "biomass": [],
            "coherence": [],
            "organisms": [],
            "network_links": [],
        }

    def _make_rotation_matrix(self) -> torch.Tensor:
        raw = torch.randn((4, 4), device=self.device, dtype=self.dtype)
        q, _ = torch.linalg.qr(raw)
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q

    def laplace(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.conv2d(padded, kernel, groups=x.shape[1])

    def step(self) -> None:
        dt = 0.22

        lap_psi = self.laplace(self.psi, self.lap4)
        mixed = torch.einsum("ij,bjhw->bihw", self.channel_mix, self.psi)
        shifted_x = torch.roll(self.psi, shifts=1, dims=3) - torch.roll(self.psi, shifts=-1, dims=3)
        shifted_y = torch.roll(self.psi, shifts=1, dims=2) - torch.roll(self.psi, shifts=-1, dims=2)
        swirl = torch.stack(
            [
                shifted_y[:, 1] - shifted_x[:, 2],
                shifted_x[:, 0] + shifted_y[:, 3],
                shifted_y[:, 2] - shifted_x[:, 0],
                shifted_x[:, 1] - shifted_y[:, 2],
            ],
            dim=1,
        )
        psi_drive = 0.12 * torch.sin(mixed * 1.7 + self.phase_bias.view(1, -1, 1, 1) + self.time * 0.025)
        self.psi = self.psi + dt * (0.18 * lap_psi + 0.17 * mixed + 0.06 * swirl + psi_drive - 0.13 * self.psi)
        self.psi = self.psi / (self.psi.norm(dim=1, keepdim=True) + 1e-4)

        phase = torch.atan2(self.psi[:, 1:2], self.psi[:, 0:1])
        coherence = 0.5 + 0.5 * torch.tanh(self.psi[:, :2].norm(dim=1, keepdim=True) * 1.7 - 0.9)

        lap_chem = self.laplace(self.chem, self.lap3)
        a = self.chem[:, 0:1]
        b = self.chem[:, 1:2]
        nutrient = self.chem[:, 2:3]

        free_norm = self.free_energy / (self.free_energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
        heat_norm = self.heat / (self.heat.amax(dim=(2, 3), keepdim=True) + 1e-6)

        feed = 0.022 + 0.018 * self.source_map + 0.012 * coherence
        kill = 0.048 + 0.015 * heat_norm + 0.006 * (1.0 - coherence)
        reaction = a * b * b
        substrate_push = 0.04 * torch.sigmoid(torch.sin(phase * 2.0) + coherence * 1.1)

        a = a + dt * (0.18 * lap_chem[:, 0:1] - reaction + feed * (1.0 - a))
        b = b + dt * (0.09 * lap_chem[:, 1:2] + reaction - (kill + feed) * b + substrate_push * nutrient)
        nutrient = nutrient + dt * (
            0.15 * lap_chem[:, 2:3]
            + 0.12 * self.source_map
            - 0.08 * nutrient
            - 0.04 * reaction
            + 0.015 * torch.relu(1.0 - heat_norm)
        )
        self.chem = torch.cat([a, b.clamp(0.0, 1.2), nutrient.clamp(0.0, 1.2)], dim=1)

        lap_free = self.laplace(self.free_energy, self.lap1)
        lap_heat = self.laplace(self.heat, self.lap1)
        lap_signal = self.laplace(self.signal, self.lap1)
        lap_biomass = self.laplace(self.biomass, self.lap1)

        life_potential = torch.sigmoid(
            4.4 * (self.chem[:, 1:2] - 0.24)
            + 2.8 * (free_norm - 0.34)
            - 3.1 * heat_norm
            + 1.7 * coherence
            + 0.8 * torch.sin(phase * 3.0)
        )

        self.biomass = (
            self.biomass
            + dt * (0.11 * lap_biomass + 0.3 * (life_potential - self.biomass))
            + 0.018 * self.chem[:, 1:2] * torch.relu(self.free_energy - 0.15)
        ).clamp(0.0, 1.0)

        activity = torch.relu(self.biomass - 0.35)
        energy_consumption = 0.03 * reaction + 0.05 * activity + 0.03 * torch.relu(nutrient - 0.4) * activity
        self.free_energy = (
            self.free_energy
            + dt * (0.19 * lap_free + 0.17 * self.source_map - 0.09 * self.free_energy - energy_consumption)
        ).clamp(0.0, 1.75)

        self.heat = (
            self.heat
            + dt * (
                0.14 * lap_heat
                + 0.85 * energy_consumption
                + 0.03 * reaction
                - self.edge_cooling * self.heat
            )
        ).clamp(0.0, 1.65)

        self.signal = (
            self.signal
            + dt
            * (
                0.16 * lap_signal
                + 0.28 * activity * torch.tanh(self.psi[:, 2:3] + 0.4 * self.psi[:, 3:4])
                - 0.18 * self.signal
            )
        ).clamp(-1.0, 1.0)

        render_state = self.render_world_tensor()
        self.trail = 0.86 * self.trail + 0.14 * render_state

        self.time += 1
        if self.time % self.cfg.organism_refresh == 0:
            self.refresh_organisms()
        self.capture_metrics()

    def refresh_organisms(self) -> None:
        mask = self.current_mask().detach().cpu().numpy()
        signal = self.signal[0, 0].detach().cpu().numpy()
        heat = self.heat[0, 0].detach().cpu().numpy()
        self.organisms = detect_components(mask, signal, heat)
        self.social_edges = build_social_edges(self.organisms, self.size)

    def current_mask(self) -> torch.Tensor:
        return (
            (self.biomass > 0.52)
            & (self.free_energy > 0.18)
            & (self.heat < 0.82)
            & (self.chem[:, 1:2] > 0.14)
        )[0, 0]

    def capture_metrics(self) -> None:
        energy = (self.free_energy + self.heat).flatten()
        probs = energy / (energy.sum() + 1e-6)
        entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item() / math.log(probs.numel()))
        coherence = float(self.psi[:, :2].norm(dim=1).mean().item())
        biomass = float(self.biomass.mean().item())
        free_energy = float(self.free_energy.mean().item())
        heat = float(self.heat.mean().item())
        metrics = {
            "entropy": entropy,
            "free_energy": free_energy,
            "heat": heat,
            "biomass": biomass,
            "coherence": coherence,
            "organisms": float(len(self.organisms)),
            "network_links": float(len(self.social_edges)),
        }
        self.last_metrics = metrics
        for name, value in metrics.items():
            self.history[name].append(value)

    def render_world_tensor(self) -> torch.Tensor:
        phase = torch.atan2(self.psi[:, 1:2], self.psi[:, 0:1])
        phase_norm = (phase / (2.0 * math.pi) + 0.5).remainder(1.0)
        free_norm = self.free_energy / (self.free_energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
        heat_norm = self.heat / (self.heat.amax(dim=(2, 3), keepdim=True) + 1e-6)
        biomass = self.biomass
        chemical = self.chem[:, 1:2]
        nutrient = self.chem[:, 2:3]
        coherence = 0.5 + 0.5 * torch.tanh(self.psi[:, :2].norm(dim=1, keepdim=True) * 1.7 - 0.9)
        hue = (phase_norm + 0.12 * chemical + 0.05 * self.signal + 0.08 * nutrient).remainder(1.0)
        sat = (0.42 + 0.58 * torch.sigmoid(chemical * 3.2 + biomass * 2.4)).clamp(0.0, 1.0)
        val = (
            0.05
            + 0.52 * torch.sqrt(free_norm + 1e-6)
            + 0.28 * coherence
            + 0.24 * biomass
            + 0.14 * nutrient
        ).clamp(0.0, 1.0)
        rgb = hsv_to_rgb_torch(hue, sat, val)

        warm = torch.cat(
            [
                (0.8 * heat_norm + 0.7 * self.source_map).clamp(0.0, 1.0),
                (0.42 * heat_norm + 0.33 * self.source_map).clamp(0.0, 1.0),
                (0.18 * heat_norm).clamp(0.0, 1.0),
            ],
            dim=1,
        )
        cool = torch.cat(
            [
                (0.08 * nutrient).clamp(0.0, 1.0),
                (0.22 * nutrient + 0.18 * torch.relu(self.signal)).clamp(0.0, 1.0),
                (0.35 * nutrient + 0.45 * torch.relu(self.signal)).clamp(0.0, 1.0),
            ],
            dim=1,
        )

        rgb = rgb * (1.0 - 0.28 * heat_norm) + warm * 0.25 + cool * 0.2
        bloom = blur_rgb(torch.relu(rgb - 0.55), radius=7)
        vignette = (1.1 - 0.32 * self.radius.clamp(0.0, 1.0)).view(1, 1, self.size, self.size)
        rgb = (rgb + 0.9 * bloom) * vignette
        return rgb.clamp(0.0, 1.0)

    def dashboard_snapshot(self) -> dict[str, object]:
        with torch.no_grad():
            world = self.trail[0].permute(1, 2, 0).detach().cpu().numpy()
            mask = self.current_mask().detach().cpu().numpy()
            return {
                "world": np.clip(world, 0.0, 1.0),
                "mask": mask,
                "organisms": self.organisms,
                "edges": self.social_edges,
                "metrics": self.last_metrics.copy(),
                "history": {k: v[:] for k, v in self.history.items()},
                "step": self.time,
                "device": str(self.device),
                "grid_size": self.size,
            }


class DashboardRenderer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("Emergent Toy Universe")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 20)
        self.small_font = pygame.font.SysFont("consolas", 16)
        self.large_font = pygame.font.SysFont("segoeuisemibold", 34)
        self.world_rect = pygame.Rect(34, 34, 950, 890)
        self.info_rect = pygame.Rect(1020, 34, 546, 260)
        self.chart_rect = pygame.Rect(1020, 318, 546, 292)
        self.graph_rect = pygame.Rect(1020, 636, 546, 288)

    def shutdown(self) -> None:
        pygame.quit()

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render(self, snapshot: dict[str, object]) -> np.ndarray:
        self.screen.fill((8, 10, 18))
        self._draw_background()
        self._draw_world(snapshot)
        self._draw_info(snapshot)
        self._draw_charts(snapshot)
        self._draw_network(snapshot)
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)
        return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)

    def _draw_background(self) -> None:
        gradient = pygame.Surface((self.cfg.width, self.cfg.height), pygame.SRCALPHA)
        for y in range(self.cfg.height):
            blend = y / max(1, self.cfg.height - 1)
            color = (
                int(8 + 16 * blend),
                int(10 + 10 * blend),
                int(18 + 26 * blend),
                255,
            )
            pygame.draw.line(gradient, color, (0, y), (self.cfg.width, y))
        self.screen.blit(gradient, (0, 0))

        halo = pygame.Surface((self.cfg.width, self.cfg.height), pygame.SRCALPHA)
        pygame.draw.circle(halo, (40, 82, 164, 34), (240, 180), 320)
        pygame.draw.circle(halo, (255, 106, 62, 22), (1420, 140), 240)
        pygame.draw.circle(halo, (44, 168, 180, 20), (1200, 760), 280)
        self.screen.blit(halo, (0, 0))

    def _draw_panel(self, rect: pygame.Rect, title: str) -> None:
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, (14, 20, 34, 228), panel.get_rect(), border_radius=22)
        pygame.draw.rect(panel, (58, 78, 112, 132), panel.get_rect(), width=1, border_radius=22)
        self.screen.blit(panel, rect.topleft)
        title_surf = self.large_font.render(title, True, (230, 238, 255))
        self.screen.blit(title_surf, (rect.left + 18, rect.top + 14))

    def _draw_world(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.world_rect, "Toy Universe")
        world = snapshot["world"]
        mask = snapshot["mask"]
        organisms: list[Organism] = snapshot["organisms"]  # type: ignore[assignment]
        edges: list[tuple[int, int, float]] = snapshot["edges"]  # type: ignore[assignment]
        assert isinstance(world, np.ndarray)
        assert isinstance(mask, np.ndarray)

        world_rgb = np.clip(world * 255.0, 0, 255).astype(np.uint8)
        world_surface = pygame.surfarray.make_surface(world_rgb.swapaxes(0, 1))
        world_surface = pygame.transform.smoothscale(world_surface, (self.world_rect.width - 28, self.world_rect.height - 70))
        display_rect = world_surface.get_rect(topleft=(self.world_rect.left + 14, self.world_rect.top + 56))
        self.screen.blit(world_surface, display_rect)

        overlay = pygame.Surface(display_rect.size, pygame.SRCALPHA)
        scale_x = display_rect.width / mask.shape[1]
        scale_y = display_rect.height / mask.shape[0]

        outline = mask & (
            (~np.roll(mask, 1, axis=0))
            | (~np.roll(mask, -1, axis=0))
            | (~np.roll(mask, 1, axis=1))
            | (~np.roll(mask, -1, axis=1))
        )
        ys, xs = np.nonzero(outline)
        for x, y in zip(xs, ys):
            overlay.fill(
                (255, 247, 222, 120),
                pygame.Rect(int(x * scale_x), int(y * scale_y), max(1, int(scale_x)), max(1, int(scale_y))),
            )

        for i, organism in enumerate(organisms[:10]):
            x_min, y_min, x_max, y_max = organism.bbox
            rect = pygame.Rect(
                int(x_min * scale_x),
                int(y_min * scale_y),
                max(4, int((x_max - x_min + 1) * scale_x)),
                max(4, int((y_max - y_min + 1) * scale_y)),
            )
            color = (111, 240, 255, 110) if i < 4 else (240, 220, 255, 85)
            pygame.draw.rect(overlay, color, rect, width=1, border_radius=8)
            cx, cy = organism.centroid
            pygame.draw.circle(overlay, (255, 248, 214, 180), (int(cx * scale_x), int(cy * scale_y)), 3)

        for src_idx, dst_idx, strength in edges:
            src = organisms[src_idx]
            dst = organisms[dst_idx]
            ax, ay = src.centroid
            bx, by = dst.centroid
            color = (
                int(70 + 140 * strength),
                int(180 + 60 * strength),
                int(255),
                int(40 + 120 * strength),
            )
            pygame.draw.line(
                overlay,
                color,
                (int(ax * scale_x), int(ay * scale_y)),
                (int(bx * scale_x), int(by * scale_y)),
                width=max(1, int(1 + strength * 2)),
            )

        self.screen.blit(overlay, display_rect.topleft)
        footer = self.small_font.render(
            "Hue: substrate phase  |  Brightness: free energy  |  Warm glow: heat  |  White outlines: persistent structures",
            True,
            (175, 190, 220),
        )
        self.screen.blit(footer, (self.world_rect.left + 18, self.world_rect.bottom - 28))

    def _draw_info(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.info_rect, "State")
        metrics: dict[str, float] = snapshot["metrics"]  # type: ignore[assignment]
        organisms: list[Organism] = snapshot["organisms"]  # type: ignore[assignment]
        edges: list[tuple[int, int, float]] = snapshot["edges"]  # type: ignore[assignment]
        step = int(snapshot["step"])
        device = str(snapshot["device"])
        grid_size = int(snapshot["grid_size"])

        rows = [
            f"step                {step:>5d}",
            f"device              {device}",
            f"grid                {grid_size} x {grid_size}",
            f"entropy             {metrics['entropy']:.3f}",
            f"free energy         {metrics['free_energy']:.3f}",
            f"heat                {metrics['heat']:.3f}",
            f"biomass             {metrics['biomass']:.3f}",
            f"coherence           {metrics['coherence']:.3f}",
            f"organisms           {len(organisms):>5d}",
            f"social links        {len(edges):>5d}",
        ]
        for idx, row in enumerate(rows):
            text = self.small_font.render(row, True, (210, 224, 255))
            self.screen.blit(text, (self.info_rect.left + 24, self.info_rect.top + 64 + idx * 22))

        pygame.draw.line(
            self.screen,
            (52, 78, 112),
            (self.info_rect.left + 300, self.info_rect.top + 64),
            (self.info_rect.left + 300, self.info_rect.bottom - 24),
            width=1,
        )

        labels = [
            ("Largest organism", organisms[0].area if organisms else 0),
            ("Second largest", organisms[1].area if len(organisms) > 1 else 0),
            ("Third largest", organisms[2].area if len(organisms) > 2 else 0),
        ]
        for idx, (label, value) in enumerate(labels):
            text = self.font.render(label, True, (239, 240, 255))
            value_surf = self.font.render(str(value), True, (115, 232, 255))
            y = self.info_rect.top + 72 + idx * 56
            self.screen.blit(text, (self.info_rect.left + 332, y))
            self.screen.blit(value_surf, (self.info_rect.left + 332, y + 24))

        tag = self.small_font.render("Emergence needs gradients, feedback, and local rules.", True, (164, 184, 220))
        self.screen.blit(tag, (self.info_rect.left + 24, self.info_rect.bottom - 28))

    def _draw_charts(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.chart_rect, "Signals")
        history: dict[str, list[float]] = snapshot["history"]  # type: ignore[assignment]

        plot_area = pygame.Rect(self.chart_rect.left + 18, self.chart_rect.top + 60, self.chart_rect.width - 36, self.chart_rect.height - 78)
        pygame.draw.rect(self.screen, (11, 16, 28), plot_area, border_radius=18)
        pygame.draw.rect(self.screen, (48, 64, 90), plot_area, width=1, border_radius=18)

        colors = {
            "entropy": (255, 190, 92),
            "free_energy": (110, 232, 255),
            "heat": (255, 104, 74),
            "biomass": (148, 246, 174),
            "coherence": (203, 165, 255),
        }
        for idx, (name, color) in enumerate(colors.items()):
            pts = sparkline_points(history[name], plot_area)
            if len(pts) > 1:
                pygame.draw.lines(self.screen, color, False, pts, width=2)
            label = self.small_font.render(name.replace("_", " "), True, color)
            self.screen.blit(label, (plot_area.left + 14 + idx * 96, plot_area.top + 12))

    def _draw_network(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.graph_rect, "Organism Network")
        organisms: list[Organism] = snapshot["organisms"]  # type: ignore[assignment]
        edges: list[tuple[int, int, float]] = snapshot["edges"]  # type: ignore[assignment]
        if not organisms:
            label = self.font.render("No persistent structures yet.", True, (176, 188, 214))
            self.screen.blit(label, (self.graph_rect.left + 24, self.graph_rect.top + 92))
            return

        graph_area = pygame.Rect(self.graph_rect.left + 18, self.graph_rect.top + 58, self.graph_rect.width - 36, self.graph_rect.height - 76)
        pygame.draw.rect(self.screen, (10, 15, 26), graph_area, border_radius=18)
        pygame.draw.rect(self.screen, (46, 61, 86), graph_area, width=1, border_radius=18)

        top = organisms[:10]
        xs = [org.centroid[0] for org in top]
        ys = [org.centroid[1] for org in top]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)

        node_positions: list[tuple[int, int]] = []
        for org in top:
            nx = graph_area.left + 34 + int(((org.centroid[0] - min_x) / span_x) * (graph_area.width - 68))
            ny = graph_area.top + 28 + int(((org.centroid[1] - min_y) / span_y) * (graph_area.height - 56))
            node_positions.append((nx, ny))

        for src, dst, strength in edges:
            if src >= len(top) or dst >= len(top):
                continue
            color = (int(60 + 120 * strength), int(150 + 80 * strength), 255)
            pygame.draw.line(self.screen, color, node_positions[src], node_positions[dst], width=max(1, int(1 + strength * 3)))

        for idx, org in enumerate(top):
            pos = node_positions[idx]
            radius = max(6, min(18, int(4 + math.sqrt(org.area) * 0.6)))
            glow = pygame.Surface((radius * 6, radius * 6), pygame.SRCALPHA)
            pygame.draw.circle(glow, (80, 226, 255, 28), (radius * 3, radius * 3), radius * 2)
            self.screen.blit(glow, (pos[0] - radius * 3, pos[1] - radius * 3))
            color = (244, 248, 255) if idx < 3 else (122, 223, 255)
            pygame.draw.circle(self.screen, color, pos, radius)
            label = self.small_font.render(str(org.area), True, (12, 14, 20))
            self.screen.blit(label, label.get_rect(center=pos))


def save_gif(frames: list[np.ndarray], output_path: Path, duration_ms: int = 90) -> None:
    images = [Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_poster(
    output_path: Path,
    history: dict[str, list[float]],
    sample_frames: list[np.ndarray],
    final_frame: np.ndarray,
    snapshot: dict[str, object],
) -> None:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), dpi=140)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 1.2, 1.2], height_ratios=[1.1, 1.0], wspace=0.16, hspace=0.18)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(final_frame)
    ax_main.set_title("Emergent Toy Universe", fontsize=20, loc="left")
    ax_main.axis("off")

    ax_metrics = fig.add_subplot(gs[0, 1:])
    colors = {
        "entropy": "#ffbf5c",
        "free_energy": "#6ee8ff",
        "heat": "#ff684a",
        "biomass": "#94f6ae",
        "coherence": "#cba5ff",
    }
    for key, color in colors.items():
        ax_metrics.plot(history[key], label=key.replace("_", " "), lw=2.2, color=color)
    ax_metrics.set_title("System Metrics")
    ax_metrics.set_xlabel("Step")
    ax_metrics.grid(alpha=0.18)
    ax_metrics.legend(frameon=False, ncols=3, fontsize=9)

    ax_phase = fig.add_subplot(gs[1, 1])
    x = np.array(history["entropy"], dtype=np.float32)
    y = np.array(history["biomass"], dtype=np.float32)
    c = np.linspace(0.0, 1.0, num=x.shape[0])
    scatter = ax_phase.scatter(x, y, c=c, cmap="magma", s=18, alpha=0.8)
    ax_phase.set_title("Phase Portrait")
    ax_phase.set_xlabel("Entropy")
    ax_phase.set_ylabel("Biomass")
    fig.colorbar(scatter, ax=ax_phase, fraction=0.046, pad=0.04, label="time")

    ax_strip = fig.add_subplot(gs[1, 2])
    if sample_frames:
        strip = np.concatenate(sample_frames, axis=0)
        ax_strip.imshow(strip)
    ax_strip.set_title("Evolution Strip")
    ax_strip.axis("off")

    metrics: dict[str, float] = snapshot["metrics"]  # type: ignore[assignment]
    fig.text(
        0.67,
        0.08,
        "\n".join(
            [
                f"device: {snapshot['device']}",
                f"grid: {snapshot['grid_size']} x {snapshot['grid_size']}",
                f"step: {snapshot['step']}",
                f"entropy: {metrics['entropy']:.3f}",
                f"free energy: {metrics['free_energy']:.3f}",
                f"heat: {metrics['heat']:.3f}",
                f"biomass: {metrics['biomass']:.3f}",
                f"organisms: {int(metrics['organisms'])}",
                f"social links: {int(metrics['network_links'])}",
            ]
        ),
        fontsize=11,
        color="#d7e4ff",
        family="monospace",
        ha="left",
        va="bottom",
    )

    fig.patch.set_facecolor("#080a12")
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run(cfg: Config) -> dict[str, Path]:
    set_seed(cfg.seed)
    device = choose_device(cfg.device)

    if cfg.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = ToyUniverse(cfg, device)
    renderer = DashboardRenderer(cfg)
    frames_for_gif: list[np.ndarray] = []
    sample_frames: list[np.ndarray] = []
    final_frame: np.ndarray | None = None
    running = True

    while running and sim.time < cfg.steps:
        running = renderer.process_events()
        sim.step()
        snapshot = sim.dashboard_snapshot()
        frame = renderer.render(snapshot)
        final_frame = frame.copy()
        if sim.time % cfg.sample_every == 0:
            gif_frame = np.array(Image.fromarray(frame).resize((1200, 720), resample=Image.Resampling.LANCZOS))
            frames_for_gif.append(gif_frame)
        if sim.time in {cfg.steps // 4, cfg.steps // 2, (3 * cfg.steps) // 4}:
            sample_frames.append(np.array(Image.fromarray(frame).resize((480, 288), resample=Image.Resampling.LANCZOS)))

    snapshot = sim.dashboard_snapshot()
    artifacts: dict[str, Path] = {}

    if cfg.save_gif and frames_for_gif:
        gif_path = output_dir / "toy_universe_timelapse.gif"
        save_gif(frames_for_gif, gif_path)
        artifacts["gif"] = gif_path

    if cfg.save_poster and final_frame is not None:
        poster_path = output_dir / "toy_universe_showcase.png"
        save_poster(poster_path, sim.history, sample_frames, final_frame, snapshot)
        artifacts["poster"] = poster_path

    renderer.shutdown()
    return artifacts


def parse_args(argv: Iterable[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="GPU-accelerated toy universe with emergent structures.")
    parser.add_argument("--grid-size", type=int, default=192, help="Universe lattice width and height.")
    parser.add_argument("--width", type=int, default=1600, help="Window width.")
    parser.add_argument("--height", type=int, default=960, help="Window height.")
    parser.add_argument("--steps", type=int, default=360, help="Number of simulation steps.")
    parser.add_argument("--fps", type=int, default=30, help="Render frame rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--headless", action="store_true", help="Render without opening a visible window.")
    parser.add_argument("--device", default="auto", help='Torch device, for example "auto", "cuda", or "cpu".')
    parser.add_argument("--output-dir", default="artifacts", help="Where to write visualization files.")
    parser.add_argument("--sample-every", type=int, default=4, help="Store every Nth frame in the GIF.")
    parser.add_argument("--organism-refresh", type=int, default=6, help="Refresh connected components every N steps.")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF export.")
    parser.add_argument("--no-poster", action="store_true", help="Skip poster export.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return Config(
        grid_size=args.grid_size,
        width=args.width,
        height=args.height,
        steps=args.steps,
        fps=args.fps,
        seed=args.seed,
        headless=args.headless,
        save_gif=not args.no_gif,
        save_poster=not args.no_poster,
        sample_every=args.sample_every,
        organism_refresh=args.organism_refresh,
        output_dir=args.output_dir,
        device=args.device,
    )


def main(argv: Iterable[str] | None = None) -> None:
    cfg = parse_args(argv)
    artifacts = run(cfg)
    for name, path in artifacts.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
