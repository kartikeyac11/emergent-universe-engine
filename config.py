from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    grid_size: int = 192
    internal_dim: int = 4
    chemical_dim: int = 3
    width: int = 1720
    height: int = 980
    steps: int = 360
    fps: int = 30
    seed: int = 7
    headless: bool = False
    output_dir: str = "artifacts_full"
    sample_every: int = 4
    organism_refresh: int = 4
    science_period: int = 24
    max_organisms: int = 32
    max_memory: int = 32
    device: str = "auto"
    hidden_size: int = 12
    action_size: int = 6
    observation_size: int = 11

