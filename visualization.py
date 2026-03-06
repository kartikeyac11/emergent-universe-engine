from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pygame
import torch

from state import UniverseState
from utils import blur_rgb, hsv_to_rgb_torch, sparkline_points


def render_world_tensor(state: UniverseState) -> torch.Tensor:
    phase_norm = (state.phase / (2.0 * math.pi) + 0.5).remainder(1.0)
    free_norm = state.free_energy / (state.free_energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
    heat_norm = state.heat / (state.heat.amax(dim=(2, 3), keepdim=True) + 1e-6)
    biomass = state.biomass
    nutrient = state.chem[:, 2:3]
    chemical = state.chem[:, 1:2]

    hue = (phase_norm + 0.11 * chemical + 0.04 * state.signal + 0.08 * nutrient + 0.03 * state.matter_type).remainder(1.0)
    sat = (0.46 + 0.54 * torch.sigmoid(chemical * 3.0 + biomass * 2.2 + state.membrane * 1.8)).clamp(0.0, 1.0)
    val = (
        0.06
        + 0.46 * torch.sqrt(free_norm + 1e-6)
        + 0.23 * state.coherence
        + 0.20 * biomass
        + 0.14 * nutrient
        + 0.10 * state.membrane
    ).clamp(0.0, 1.0)
    rgb = hsv_to_rgb_torch(hue, sat, val)

    warm = torch.cat(
        [
            (0.86 * heat_norm + 0.72 * state.source_map).clamp(0.0, 1.0),
            (0.44 * heat_norm + 0.34 * state.source_map).clamp(0.0, 1.0),
            (0.18 * heat_norm).clamp(0.0, 1.0),
        ],
        dim=1,
    )
    cool = torch.cat(
        [
            (0.08 * nutrient + 0.08 * state.membrane).clamp(0.0, 1.0),
            (0.20 * nutrient + 0.20 * torch.relu(state.signal)).clamp(0.0, 1.0),
            (0.34 * nutrient + 0.44 * torch.relu(state.signal)).clamp(0.0, 1.0),
        ],
        dim=1,
    )
    rgb = rgb * (1.0 - 0.26 * heat_norm) + warm * 0.24 + cool * 0.22
    bloom = blur_rgb(torch.relu(rgb - 0.56), radius=7)
    vignette = (1.08 - 0.30 * state.radius.clamp(0.0, 1.0)).view(1, 1, state.cfg.grid_size, state.cfg.grid_size)
    rgb = (rgb + 0.85 * bloom) * vignette
    return rgb.clamp(0.0, 1.0)


def snapshot_from_state(state: UniverseState) -> dict[str, object]:
    world = render_world_tensor(state)
    state.trail = 0.86 * state.trail + 0.14 * world
    mask = (
        (state.biomass > 0.54)
        & (state.free_energy > 0.16)
        & (state.heat < 0.90)
        & (state.chem[:, 1:2] > 0.14)
    )[0, 0]

    metrics = {name: values[-1] for name, values in state.history.items() if values}
    organisms = [organism for organism in state.organisms.values() if organism.alive]
    organisms.sort(key=lambda organism: organism.fitness, reverse=True)

    return {
        "world": state.trail[0].permute(1, 2, 0).detach().cpu().numpy(),
        "mask": mask.detach().cpu().numpy(),
        "clusters": state.clusters,
        "cluster_map": state.cluster_map.copy(),
        "organisms": organisms,
        "social_edges": list(state.social_edges.values()),
        "science_report": dict(state.science_report),
        "history": {key: values[:] for key, values in state.history.items()},
        "metrics": metrics,
        "step": state.step,
        "device": str(state.device),
        "grid_size": state.cfg.grid_size,
    }


class DashboardRenderer:
    def __init__(self, width: int, height: int, fps: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Emergent Universe Engine")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont("segoeui", 20)
        self.small_font = pygame.font.SysFont("consolas", 16)
        self.large_font = pygame.font.SysFont("segoeuisemibold", 34)
        margin = 28
        gutter = 24
        world_width = int(width * 0.58)
        self.world_rect = pygame.Rect(margin, margin, world_width, height - 2 * margin)

        right_x = self.world_rect.right + gutter
        right_width = width - right_x - margin
        top_height = max(210, int(height * 0.23))
        chart_height = max(200, int(height * 0.23))
        network_height = max(180, int(height * 0.21))
        science_height = max(150, height - (margin * 2 + top_height + chart_height + network_height + gutter * 3))

        self.top_rect = pygame.Rect(right_x, margin, right_width, top_height)
        self.chart_rect = pygame.Rect(right_x, self.top_rect.bottom + gutter, right_width, chart_height)
        self.network_rect = pygame.Rect(right_x, self.chart_rect.bottom + gutter, right_width, network_height)
        self.science_rect = pygame.Rect(right_x, self.network_rect.bottom + gutter, right_width, science_height)

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def shutdown(self) -> None:
        pygame.quit()

    def render(self, snapshot: dict[str, object]) -> np.ndarray:
        self.screen.fill((8, 10, 18))
        self._draw_background()
        self._draw_world(snapshot)
        self._draw_state(snapshot)
        self._draw_charts(snapshot)
        self._draw_network(snapshot)
        self._draw_science(snapshot)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)

    def _draw_background(self) -> None:
        gradient = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for y in range(self.height):
            blend = y / max(1, self.height - 1)
            color = (int(8 + 16 * blend), int(10 + 12 * blend), int(18 + 24 * blend), 255)
            pygame.draw.line(gradient, color, (0, y), (self.width, y))
        self.screen.blit(gradient, (0, 0))

        glow = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(glow, (36, 84, 164, 34), (220, 180), 300)
        pygame.draw.circle(glow, (255, 112, 68, 20), (1460, 150), 260)
        pygame.draw.circle(glow, (44, 168, 180, 18), (1280, 720), 300)
        self.screen.blit(glow, (0, 0))

    def _draw_panel(self, rect: pygame.Rect, title: str) -> None:
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, (14, 20, 34, 228), panel.get_rect(), border_radius=22)
        pygame.draw.rect(panel, (58, 78, 112, 132), panel.get_rect(), width=1, border_radius=22)
        self.screen.blit(panel, rect.topleft)
        title_surface = self.large_font.render(title, True, (232, 238, 255))
        self.screen.blit(title_surface, (rect.left + 18, rect.top + 14))

    def _draw_world(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.world_rect, "Emergent Universe")
        world = np.clip(snapshot["world"] * 255.0, 0, 255).astype(np.uint8)
        mask = snapshot["mask"]
        clusters = snapshot["clusters"]
        cluster_map = snapshot["cluster_map"]
        organisms = snapshot["organisms"]
        assert isinstance(mask, np.ndarray)

        world_surface = pygame.surfarray.make_surface(world.swapaxes(0, 1))
        world_surface = pygame.transform.smoothscale(world_surface, (self.world_rect.width - 28, self.world_rect.height - 70))
        display_rect = world_surface.get_rect(topleft=(self.world_rect.left + 14, self.world_rect.top + 56))
        self.screen.blit(world_surface, display_rect)

        overlay = pygame.Surface(display_rect.size, pygame.SRCALPHA)
        scale_x = display_rect.width / mask.shape[1]
        scale_y = display_rect.height / mask.shape[0]
        outline = mask & ((~np.roll(mask, 1, axis=0)) | (~np.roll(mask, 1, axis=1)) | (~np.roll(mask, -1, axis=0)) | (~np.roll(mask, -1, axis=1)))
        ys, xs = np.nonzero(outline)
        for x, y in zip(xs, ys):
            overlay.fill((255, 246, 224, 112), pygame.Rect(int(x * scale_x), int(y * scale_y), max(1, int(scale_x)), max(1, int(scale_y))))

        top_ids = {organism.organism_id for organism in organisms[:10]}
        for index, cluster in enumerate(clusters[:18]):
            x_min, y_min, x_max, y_max = cluster.bbox
            rect = pygame.Rect(
                int(x_min * scale_x),
                int(y_min * scale_y),
                max(4, int((x_max - x_min + 1) * scale_x)),
                max(4, int((y_max - y_min + 1) * scale_y)),
            )
            organism_id = cluster_map.get(index)
            color = (111, 240, 255, 110) if organism_id in top_ids else (236, 216, 255, 76)
            pygame.draw.rect(overlay, color, rect, width=1, border_radius=8)
            if organism_id is not None:
                label = self.small_font.render(str(organism_id), True, (245, 248, 255))
                overlay.blit(label, (rect.left + 2, rect.top + 1))

        self.screen.blit(overlay, display_rect.topleft)
        footer = self.small_font.render(
            "phase + chemistry + biomass + heat -> color | outlines = persistent organisms | labels = tracked ids",
            True,
            (176, 190, 220),
        )
        self.screen.blit(footer, (self.world_rect.left + 18, self.world_rect.bottom - 28))

    def _draw_state(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.top_rect, "State")
        metrics = snapshot["metrics"]
        organisms = snapshot["organisms"]
        science = snapshot["science_report"]

        rows = [
            f"step              {int(snapshot['step']):>5d}",
            f"device            {snapshot['device']}",
            f"grid              {snapshot['grid_size']} x {snapshot['grid_size']}",
            f"entropy           {metrics.get('entropy', 0.0):.3f}",
            f"free energy       {metrics.get('free_energy', 0.0):.3f}",
            f"heat              {metrics.get('heat', 0.0):.3f}",
            f"biomass           {metrics.get('biomass', 0.0):.3f}",
            f"membrane          {metrics.get('membrane', 0.0):.3f}",
            f"population        {int(metrics.get('population', 0.0)):>5d}",
            f"links             {int(metrics.get('links', 0.0)):>5d}",
            f"diversity         {metrics.get('diversity', 0.0):.3f}",
            f"science score     {metrics.get('science', 0.0):.3f}",
        ]
        for idx, row in enumerate(rows):
            text = self.small_font.render(row, True, (212, 224, 255))
            self.screen.blit(text, (self.top_rect.left + 22, self.top_rect.top + 62 + idx * 18))

        pygame.draw.line(
            self.screen,
            (54, 78, 112),
            (self.top_rect.left + 276, self.top_rect.top + 62),
            (self.top_rect.left + 276, self.top_rect.bottom - 24),
            width=1,
        )

        label = self.font.render("Top Organisms", True, (238, 242, 255))
        self.screen.blit(label, (self.top_rect.left + 300, self.top_rect.top + 62))
        for idx, organism in enumerate(organisms[:5]):
            line = f"#{organism.organism_id:02d} age {organism.age:03d} reserve {organism.energy_reserve:.2f} dream {organism.dream_counter:02d}"
            text = self.small_font.render(line, True, (118, 232, 255) if idx < 2 else (197, 205, 232))
            self.screen.blit(text, (self.top_rect.left + 300, self.top_rect.top + 96 + idx * 24))

        note = self.small_font.render(f"latest discovery source: {len(organisms[0].discoveries) if organisms else 0} notes", True, (170, 188, 220))
        self.screen.blit(note, (self.top_rect.left + 300, self.top_rect.bottom - 30))

    def _draw_charts(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.chart_rect, "Signals")
        history = snapshot["history"]
        plot = pygame.Rect(self.chart_rect.left + 18, self.chart_rect.top + 56, self.chart_rect.width - 36, self.chart_rect.height - 74)
        pygame.draw.rect(self.screen, (10, 16, 28), plot, border_radius=18)
        pygame.draw.rect(self.screen, (48, 64, 90), plot, width=1, border_radius=18)

        colors = {
            "entropy": (255, 192, 96),
            "free_energy": (110, 232, 255),
            "heat": (255, 106, 72),
            "population": (148, 246, 174),
            "science": (203, 165, 255),
        }
        for idx, (name, color) in enumerate(colors.items()):
            points = sparkline_points(history.get(name, []), plot.left + 10, plot.top + 26, plot.width - 20, plot.height - 36)
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, width=2)
            label = self.small_font.render(name.replace("_", " "), True, color)
            self.screen.blit(label, (plot.left + 12 + idx * 96, plot.top + 8))

    def _draw_network(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.network_rect, "Social Graph")
        organisms = snapshot["organisms"][:10]
        edges = snapshot["social_edges"]
        if not organisms:
            text = self.font.render("No organisms yet.", True, (174, 188, 214))
            self.screen.blit(text, (self.network_rect.left + 24, self.network_rect.top + 90))
            return

        area = pygame.Rect(self.network_rect.left + 16, self.network_rect.top + 58, self.network_rect.width - 32, self.network_rect.height - 72)
        pygame.draw.rect(self.screen, (10, 15, 26), area, border_radius=18)
        pygame.draw.rect(self.screen, (48, 64, 90), area, width=1, border_radius=18)

        xs = [organism.centroid[0] for organism in organisms]
        ys = [organism.centroid[1] for organism in organisms]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        positions: dict[int, tuple[int, int]] = {}

        for organism in organisms:
            px = area.left + 28 + int(((organism.centroid[0] - min_x) / span_x) * (area.width - 56))
            py = area.top + 22 + int(((organism.centroid[1] - min_y) / span_y) * (area.height - 44))
            positions[organism.organism_id] = (px, py)

        for edge in edges:
            if edge.organism_a not in positions or edge.organism_b not in positions:
                continue
            width = max(1, int(1 + edge.weight * 3))
            color = (
                max(0, min(255, int(70 + 110 * edge.trust))),
                max(0, min(255, int(140 + 90 * edge.weight))),
                255,
            )
            pygame.draw.line(self.screen, color, positions[edge.organism_a], positions[edge.organism_b], width=width)

        for index, organism in enumerate(organisms):
            pos = positions[organism.organism_id]
            radius = max(7, min(18, int(4 + math.sqrt(max(organism.area, 1)) * 0.6)))
            color = (244, 248, 255) if index < 3 else (122, 223, 255)
            pygame.draw.circle(self.screen, color, pos, radius)
            text = self.small_font.render(str(organism.organism_id), True, (10, 12, 18))
            self.screen.blit(text, text.get_rect(center=pos))

    def _draw_science(self, snapshot: dict[str, object]) -> None:
        self._draw_panel(self.science_rect, "Science Engine")
        science = snapshot["science_report"]
        organisms = snapshot["organisms"]
        lines = [
            f"heat law    {science.get('heat_law', 'collecting...')}",
            f"energy law  {science.get('energy_law', 'collecting...')}",
            f"score       {science.get('score', 0.0):.3f}",
            "",
            "recent discoveries:",
        ]
        for organism in organisms[:4]:
            if organism.discoveries:
                lines.append(f"#{organism.organism_id:02d} {organism.discoveries[-1]}")

        for idx, line in enumerate(lines):
            text = self.small_font.render(line, True, (214, 224, 255) if idx < 3 else (174, 192, 224))
            self.screen.blit(text, (self.science_rect.left + 22, self.science_rect.top + 60 + idx * 22))


def save_gif(frames: list[np.ndarray], output_path: Path, duration_ms: int = 90) -> None:
    images = [Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE) for frame in frames]
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=False)


def save_poster(output_path: Path, snapshot: dict[str, object], sample_frames: list[np.ndarray], final_frame: np.ndarray) -> None:
    history = snapshot["history"]
    science = snapshot["science_report"]
    metrics = snapshot["metrics"]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), dpi=140)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 1.25, 1.25], height_ratios=[1.1, 1.0], wspace=0.16, hspace=0.18)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(final_frame)
    ax_main.set_title("Emergent Universe Engine", fontsize=20, loc="left")
    ax_main.axis("off")

    ax_metrics = fig.add_subplot(gs[0, 1:])
    for key, color in {
        "entropy": "#ffbf5c",
        "free_energy": "#6ee8ff",
        "heat": "#ff684a",
        "population": "#94f6ae",
        "science": "#cba5ff",
    }.items():
        ax_metrics.plot(history.get(key, []), lw=2.1, label=key.replace("_", " "), color=color)
    ax_metrics.set_title("System Metrics")
    ax_metrics.grid(alpha=0.18)
    ax_metrics.legend(frameon=False, ncols=3, fontsize=9)

    ax_strip = fig.add_subplot(gs[1, 1])
    if sample_frames:
        strip = np.concatenate(sample_frames, axis=0)
        ax_strip.imshow(strip)
    ax_strip.set_title("Evolution Strip")
    ax_strip.axis("off")

    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    text = "\n".join(
        [
            f"step: {snapshot['step']}",
            f"device: {snapshot['device']}",
            f"population: {int(metrics.get('population', 0))}",
            f"diversity: {metrics.get('diversity', 0.0):.3f}",
            f"science score: {metrics.get('science', 0.0):.3f}",
            "",
            "heat law:",
            str(science.get("heat_law", "collecting...")),
            "",
            "energy law:",
            str(science.get("energy_law", "collecting...")),
        ]
    )
    ax_text.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace", color="#d7e4ff")

    fig.patch.set_facecolor("#080a12")
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
