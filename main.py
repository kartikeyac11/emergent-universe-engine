from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from cellular_life import CellularLifeEngine
from config import SimulationConfig
from evolution import EvolutionEngine
from neural_brain import NeuralBrainEngine
from organism import OrganismManager
from physics_engine import PhysicsEngine
from science_engine import ScienceEngine
from social_network import SocialNetworkEngine
from state import initial_state
from thermodynamics import ThermodynamicsEngine
from utils import choose_device, set_seed
from visualization import DashboardRenderer, save_gif, save_poster, snapshot_from_state


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(description="Emergent Universe Engine")
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--width", type=int, default=1720)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", default="artifacts_full")
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--organism-refresh", type=int, default=4)
    parser.add_argument("--science-period", type=int, default=24)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return SimulationConfig(
        grid_size=args.grid_size,
        width=args.width,
        height=args.height,
        steps=args.steps,
        fps=args.fps,
        seed=args.seed,
        headless=args.headless,
        output_dir=args.output_dir,
        sample_every=args.sample_every,
        organism_refresh=args.organism_refresh,
        science_period=args.science_period,
        device=args.device,
    )


def run(cfg: SimulationConfig) -> dict[str, Path]:
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)

    if cfg.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = initial_state(cfg, device)
    physics = PhysicsEngine(state)
    thermodynamics = ThermodynamicsEngine()
    cellular = CellularLifeEngine()
    organism_manager = OrganismManager(cfg, rng)
    neural = NeuralBrainEngine(cfg, rng)
    evolution = EvolutionEngine(cfg, rng, organism_manager)
    social = SocialNetworkEngine()
    science = ScienceEngine(cfg, rng)
    renderer = DashboardRenderer(cfg.width, cfg.height, cfg.fps)

    frames_for_gif: list[np.ndarray] = []
    sample_frames: list[np.ndarray] = []
    final_frame: np.ndarray | None = None
    running = True

    while running and state.step < cfg.steps:
        running = renderer.process_events()
        physics.step(state)
        thermodynamics.step(state)
        cellular.step(state)
        if state.step % cfg.organism_refresh == 0:
            organism_manager.refresh(state)
        state.reset_drives()
        neural.step(state)
        evolution.step(state)
        social.step(state)
        science.step(state)
        state.capture_metrics()

        snapshot = snapshot_from_state(state)
        frame = renderer.render(snapshot)
        final_frame = frame.copy()
        if state.step % cfg.sample_every == 0:
            gif_frame = np.array(Image.fromarray(frame).resize((1280, 728), resample=Image.Resampling.LANCZOS))
            frames_for_gif.append(gif_frame)
        if state.step in {cfg.steps // 4, cfg.steps // 2, (3 * cfg.steps) // 4}:
            sample_frames.append(np.array(Image.fromarray(frame).resize((480, 274), resample=Image.Resampling.LANCZOS)))
        state.step += 1

    snapshot = snapshot_from_state(state)
    artifacts: dict[str, Path] = {}
    gif_path = output_dir / "emergent_universe_full.gif"
    poster_path = output_dir / "emergent_universe_full.png"
    if frames_for_gif:
        save_gif(frames_for_gif, gif_path)
        artifacts["gif"] = gif_path
    if final_frame is not None:
        save_poster(poster_path, snapshot, sample_frames, final_frame)
        artifacts["poster"] = poster_path

    renderer.shutdown()
    return artifacts


def main() -> None:
    cfg = parse_args()
    artifacts = run(cfg)
    for name, path in artifacts.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()

