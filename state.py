from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from config import SimulationConfig
from utils import ComponentStats, gaussian_sources, laplace_kernel, shannon_entropy


@dataclass
class Genome:
    metabolism: float
    heat_tolerance: float
    motility: float
    signaling: float
    cooperation: float
    reproduction_threshold: float
    mutation_scale: float
    hue_shift: float
    brain_scale: float
    dream_gain: float
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    world_model: np.ndarray


@dataclass
class OrganismRecord:
    organism_id: int
    parent_id: int | None
    birth_step: int
    last_seen_step: int
    age: int
    centroid: tuple[float, float]
    velocity: tuple[float, float]
    area: int
    energy_reserve: float
    fitness: float
    genome: Genome
    memory: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = field(default_factory=list)
    last_observation: np.ndarray | None = None
    last_action: np.ndarray | None = None
    dream_counter: int = 0
    discoveries: list[str] = field(default_factory=list)
    alive: bool = True


@dataclass
class SocialEdge:
    organism_a: int
    organism_b: int
    weight: float
    trust: float
    conflict: float
    last_step: int


@dataclass
class PendingBirth:
    organism_id: int
    parent_id: int
    centroid: tuple[float, float]
    genome: Genome
    expiry_step: int


@dataclass
class UniverseState:
    cfg: SimulationConfig
    device: torch.device
    step: int
    grid_x: torch.Tensor
    grid_y: torch.Tensor
    radius: torch.Tensor
    source_map: torch.Tensor
    edge_cooling: torch.Tensor
    psi: torch.Tensor
    chem: torch.Tensor
    free_energy: torch.Tensor
    heat: torch.Tensor
    signal: torch.Tensor
    biomass: torch.Tensor
    membrane: torch.Tensor
    trail: torch.Tensor
    phase: torch.Tensor
    coherence: torch.Tensor
    matter_type: torch.Tensor
    energy_demand: torch.Tensor
    signal_drive: torch.Tensor
    repair_drive: torch.Tensor
    lap1: torch.Tensor
    lap3: torch.Tensor
    lap4: torch.Tensor
    organisms: dict[int, OrganismRecord] = field(default_factory=dict)
    clusters: list[ComponentStats] = field(default_factory=list)
    cluster_map: dict[int, int] = field(default_factory=dict)
    social_edges: dict[tuple[int, int], SocialEdge] = field(default_factory=dict)
    pending_births: list[PendingBirth] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)
    science_report: dict[str, object] = field(default_factory=dict)
    next_organism_id: int = 1
    diversity: float = 0.0
    prev_heat: torch.Tensor | None = None
    prev_free_energy: torch.Tensor | None = None

    def reset_drives(self) -> None:
        self.energy_demand.zero_()
        self.signal_drive.zero_()
        self.repair_drive.zero_()

    def capture_metrics(self) -> None:
        free_mean = float(self.free_energy.mean().item())
        heat_mean = float(self.heat.mean().item())
        biomass_mean = float(self.biomass.mean().item())
        coherence_mean = float(self.coherence.mean().item())
        membrane_mean = float(self.membrane.mean().item())
        entropy = shannon_entropy((self.free_energy + self.heat).flatten())
        science_score = float(self.science_report.get("score", 0.0))
        metrics = {
            "entropy": entropy,
            "free_energy": free_mean,
            "heat": heat_mean,
            "biomass": biomass_mean,
            "coherence": coherence_mean,
            "membrane": membrane_mean,
            "population": float(sum(1 for organism in self.organisms.values() if organism.alive)),
            "links": float(len(self.social_edges)),
            "science": science_score,
            "diversity": self.diversity,
        }
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)


def random_genome(cfg: SimulationConfig, rng: np.random.Generator) -> Genome:
    hidden = cfg.hidden_size
    obs = cfg.observation_size
    actions = cfg.action_size
    return Genome(
        metabolism=float(rng.uniform(0.7, 1.35)),
        heat_tolerance=float(rng.uniform(0.25, 0.85)),
        motility=float(rng.uniform(0.2, 1.0)),
        signaling=float(rng.uniform(0.2, 1.2)),
        cooperation=float(rng.uniform(0.1, 1.0)),
        reproduction_threshold=float(rng.uniform(0.55, 1.15)),
        mutation_scale=float(rng.uniform(0.04, 0.14)),
        hue_shift=float(rng.uniform(-0.18, 0.18)),
        brain_scale=float(rng.uniform(0.8, 1.3)),
        dream_gain=float(rng.uniform(0.05, 0.30)),
        w1=rng.normal(0.0, 0.55, size=(obs, hidden)).astype(np.float32),
        b1=rng.normal(0.0, 0.22, size=(hidden,)).astype(np.float32),
        w2=rng.normal(0.0, 0.40, size=(hidden, actions)).astype(np.float32),
        b2=rng.normal(0.0, 0.18, size=(actions,)).astype(np.float32),
        world_model=rng.normal(0.0, 0.20, size=(obs + actions, obs)).astype(np.float32),
    )


def mutate_genome(parent: Genome, cfg: SimulationConfig, rng: np.random.Generator) -> Genome:
    scale = parent.mutation_scale

    def scalar(value: float, lower: float, upper: float) -> float:
        return float(np.clip(value + rng.normal(0.0, scale), lower, upper))

    return Genome(
        metabolism=scalar(parent.metabolism, 0.5, 1.6),
        heat_tolerance=scalar(parent.heat_tolerance, 0.05, 1.0),
        motility=scalar(parent.motility, 0.05, 1.3),
        signaling=scalar(parent.signaling, 0.05, 1.5),
        cooperation=scalar(parent.cooperation, 0.0, 1.2),
        reproduction_threshold=scalar(parent.reproduction_threshold, 0.4, 1.4),
        mutation_scale=scalar(parent.mutation_scale, 0.02, 0.18),
        hue_shift=scalar(parent.hue_shift, -0.3, 0.3),
        brain_scale=scalar(parent.brain_scale, 0.5, 1.6),
        dream_gain=scalar(parent.dream_gain, 0.0, 0.45),
        w1=(parent.w1 + rng.normal(0.0, scale, size=parent.w1.shape)).astype(np.float32),
        b1=(parent.b1 + rng.normal(0.0, scale, size=parent.b1.shape)).astype(np.float32),
        w2=(parent.w2 + rng.normal(0.0, scale, size=parent.w2.shape)).astype(np.float32),
        b2=(parent.b2 + rng.normal(0.0, scale, size=parent.b2.shape)).astype(np.float32),
        world_model=(parent.world_model + rng.normal(0.0, scale, size=parent.world_model.shape)).astype(np.float32),
    )


def blend_genomes(a: Genome, b: Genome, cfg: SimulationConfig, rng: np.random.Generator) -> Genome:
    mixed = Genome(
        metabolism=float((a.metabolism + b.metabolism) * 0.5),
        heat_tolerance=float((a.heat_tolerance + b.heat_tolerance) * 0.5),
        motility=float((a.motility + b.motility) * 0.5),
        signaling=float((a.signaling + b.signaling) * 0.5),
        cooperation=float((a.cooperation + b.cooperation) * 0.5),
        reproduction_threshold=float((a.reproduction_threshold + b.reproduction_threshold) * 0.5),
        mutation_scale=float((a.mutation_scale + b.mutation_scale) * 0.5),
        hue_shift=float((a.hue_shift + b.hue_shift) * 0.5),
        brain_scale=float((a.brain_scale + b.brain_scale) * 0.5),
        dream_gain=float((a.dream_gain + b.dream_gain) * 0.5),
        w1=((a.w1 + b.w1) * 0.5).astype(np.float32),
        b1=((a.b1 + b.b1) * 0.5).astype(np.float32),
        w2=((a.w2 + b.w2) * 0.5).astype(np.float32),
        b2=((a.b2 + b.b2) * 0.5).astype(np.float32),
        world_model=((a.world_model + b.world_model) * 0.5).astype(np.float32),
    )
    return mutate_genome(mixed, cfg, rng)


def genome_distance(a: Genome, b: Genome) -> float:
    scalars = [
        abs(a.metabolism - b.metabolism),
        abs(a.heat_tolerance - b.heat_tolerance),
        abs(a.motility - b.motility),
        abs(a.signaling - b.signaling),
        abs(a.cooperation - b.cooperation),
        abs(a.reproduction_threshold - b.reproduction_threshold),
        abs(a.hue_shift - b.hue_shift),
    ]
    return float(np.mean(scalars))


def initial_state(cfg: SimulationConfig, device: torch.device) -> UniverseState:
    dtype = torch.float32
    size = cfg.grid_size
    line = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(line, line, indexing="ij")
    radius = torch.sqrt(grid_x**2 + grid_y**2)

    source_map = gaussian_sources(grid_x, grid_y).view(1, 1, size, size)
    edge_cooling = (0.10 + 0.28 * radius.clamp(0.0, 1.0)).view(1, 1, size, size)
    psi = torch.randn((1, cfg.internal_dim, size, size), device=device, dtype=dtype) * 0.35
    chem = torch.zeros((1, cfg.chemical_dim, size, size), device=device, dtype=dtype)
    chem[:, 0] = 1.0
    chem[:, 1] = 0.12 * torch.exp(-((radius / 0.55) ** 2))
    chem[:, 2] = 0.18 * source_map[:, 0]
    free_energy = 0.24 + 0.54 * source_map
    heat = 0.05 * torch.ones((1, 1, size, size), device=device, dtype=dtype)
    signal = torch.zeros((1, 1, size, size), device=device, dtype=dtype)
    biomass = 0.10 * torch.rand((1, 1, size, size), device=device, dtype=dtype)
    membrane = torch.zeros((1, 1, size, size), device=device, dtype=dtype)
    trail = torch.zeros((1, 3, size, size), device=device, dtype=dtype)

    phase = torch.atan2(psi[:, 1:2], psi[:, 0:1])
    coherence = 0.5 + 0.5 * torch.tanh(psi[:, :2].norm(dim=1, keepdim=True) * 1.7 - 0.9)
    matter_type = torch.argmax(torch.abs(psi), dim=1, keepdim=True).to(dtype)

    return UniverseState(
        cfg=cfg,
        device=device,
        step=0,
        grid_x=grid_x,
        grid_y=grid_y,
        radius=radius,
        source_map=source_map,
        edge_cooling=edge_cooling,
        psi=psi,
        chem=chem,
        free_energy=free_energy,
        heat=heat,
        signal=signal,
        biomass=biomass,
        membrane=membrane,
        trail=trail,
        phase=phase,
        coherence=coherence,
        matter_type=matter_type,
        energy_demand=torch.zeros((1, 1, size, size), device=device, dtype=dtype),
        signal_drive=torch.zeros((1, 1, size, size), device=device, dtype=dtype),
        repair_drive=torch.zeros((1, 1, size, size), device=device, dtype=dtype),
        lap1=laplace_kernel(device, 1, dtype=dtype),
        lap3=laplace_kernel(device, cfg.chemical_dim, dtype=dtype),
        lap4=laplace_kernel(device, cfg.internal_dim, dtype=dtype),
        history={},
        science_report={"heat_law": "collecting...", "energy_law": "collecting...", "score": 0.0},
    )

