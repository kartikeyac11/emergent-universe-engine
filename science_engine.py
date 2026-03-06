from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import SimulationConfig
from state import UniverseState
from utils import laplace


@dataclass
class CandidateLaw:
    mask: np.ndarray
    coeffs: np.ndarray
    bias: float


class ScienceEngine:
    def __init__(self, cfg: SimulationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.heat_features: list[np.ndarray] = []
        self.heat_targets: list[float] = []
        self.energy_features: list[np.ndarray] = []
        self.energy_targets: list[float] = []
        self.feature_names = [
            "center",
            "laplace",
            "source",
            "biomass",
            "signal",
            "chemical_b",
            "bias_hint",
        ]

    def step(self, state: UniverseState) -> None:
        self._collect_samples(state)
        if state.step == 0 or state.step % self.cfg.science_period != 0:
            return
        if len(self.heat_targets) < 64 or len(self.energy_targets) < 64:
            return

        heat_candidate, heat_score = self._evolve(np.array(self.heat_features), np.array(self.heat_targets))
        energy_candidate, energy_score = self._evolve(np.array(self.energy_features), np.array(self.energy_targets))
        score = float(max(0.0, 1.0 - 0.5 * (heat_score + energy_score)))

        heat_law = self._render_law(heat_candidate, target_name="heat_next")
        energy_law = self._render_law(energy_candidate, target_name="free_next")
        state.science_report = {
            "heat_law": heat_law,
            "energy_law": energy_law,
            "score": score,
        }

        scientist = self._lead_scientist(state)
        if scientist is not None:
            scientist.discoveries.append(heat_law)
            scientist.discoveries = scientist.discoveries[-4:]

    def _collect_samples(self, state: UniverseState) -> None:
        if state.prev_heat is None or state.prev_free_energy is None:
            return
        prev_heat = state.prev_heat
        prev_energy = state.prev_free_energy
        heat_lap = laplace(prev_heat, state.lap1)
        energy_lap = laplace(prev_energy, state.lap1)

        size = state.cfg.grid_size
        ys = self.rng.integers(0, size, size=48)
        xs = self.rng.integers(0, size, size=48)
        for x, y in zip(xs, ys):
            heat_features = np.array(
                [
                    float(prev_heat[0, 0, y, x].item()),
                    float(heat_lap[0, 0, y, x].item()),
                    float(state.source_map[0, 0, y, x].item()),
                    float(state.biomass[0, 0, y, x].item()),
                    float(state.signal[0, 0, y, x].item()),
                    float(state.chem[0, 1, y, x].item()),
                    1.0,
                ],
                dtype=np.float32,
            )
            energy_features = np.array(
                [
                    float(prev_energy[0, 0, y, x].item()),
                    float(energy_lap[0, 0, y, x].item()),
                    float(state.source_map[0, 0, y, x].item()),
                    float(state.biomass[0, 0, y, x].item()),
                    float(state.signal[0, 0, y, x].item()),
                    float(state.chem[0, 1, y, x].item()),
                    1.0,
                ],
                dtype=np.float32,
            )
            self.heat_features.append(heat_features)
            self.heat_targets.append(float(state.heat[0, 0, y, x].item()))
            self.energy_features.append(energy_features)
            self.energy_targets.append(float(state.free_energy[0, 0, y, x].item()))

        self.heat_features = self.heat_features[-640:]
        self.heat_targets = self.heat_targets[-640:]
        self.energy_features = self.energy_features[-640:]
        self.energy_targets = self.energy_targets[-640:]

    def _evolve(self, features: np.ndarray, targets: np.ndarray) -> tuple[CandidateLaw, float]:
        population = [self._random_candidate(features.shape[1]) for _ in range(24)]
        best = population[0]
        best_score = 1e9

        for _ in range(14):
            scored = []
            for candidate in population:
                error = self._score(candidate, features, targets)
                scored.append((error, candidate))
                if error < best_score:
                    best_score = error
                    best = candidate
            scored.sort(key=lambda item: item[0])
            survivors = [candidate for _, candidate in scored[:8]]
            population = survivors[:]
            while len(population) < 24:
                parent = survivors[self.rng.integers(0, len(survivors))]
                population.append(self._mutate(parent))

        return best, float(best_score)

    def _random_candidate(self, size: int) -> CandidateLaw:
        mask = self.rng.integers(0, 2, size=size).astype(np.float32)
        coeffs = self.rng.normal(0.0, 0.6, size=size).astype(np.float32)
        return CandidateLaw(mask=mask, coeffs=coeffs, bias=float(self.rng.normal(0.0, 0.2)))

    def _mutate(self, candidate: CandidateLaw) -> CandidateLaw:
        mask = candidate.mask.copy()
        coeffs = candidate.coeffs.copy()
        flip = self.rng.integers(0, len(mask))
        mask[flip] = 1.0 - mask[flip]
        coeffs = coeffs + self.rng.normal(0.0, 0.12, size=coeffs.shape).astype(np.float32)
        bias = float(candidate.bias + self.rng.normal(0.0, 0.08))
        return CandidateLaw(mask=mask, coeffs=coeffs, bias=bias)

    def _score(self, candidate: CandidateLaw, features: np.ndarray, targets: np.ndarray) -> float:
        prediction = (features * candidate.mask) @ candidate.coeffs + candidate.bias
        mse = float(np.mean((prediction - targets) ** 2))
        penalty = 0.012 * float(candidate.mask.sum())
        return mse + penalty

    def _render_law(self, candidate: CandidateLaw, target_name: str) -> str:
        pieces = []
        for name, coeff, active in zip(self.feature_names, candidate.coeffs, candidate.mask):
            if active < 0.5 or abs(coeff) < 0.04:
                continue
            pieces.append(f"{coeff:+.2f}*{name}")
        body = " ".join(pieces) if pieces else "+0.00"
        return f"{target_name} = {candidate.bias:+.2f} {body}"

    def _lead_scientist(self, state: UniverseState):
        alive = [organism for organism in state.organisms.values() if organism.alive]
        if not alive:
            return None
        alive.sort(key=lambda organism: organism.fitness + 0.03 * organism.age, reverse=True)
        return alive[0]

