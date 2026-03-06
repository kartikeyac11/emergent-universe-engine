from __future__ import annotations

import torch

from state import UniverseState
from utils import laplace


class ThermodynamicsEngine:
    def step(self, state: UniverseState) -> None:
        state.prev_heat = state.heat.detach().clone()
        state.prev_free_energy = state.free_energy.detach().clone()

        lap_free = laplace(state.free_energy, state.lap1)
        lap_heat = laplace(state.heat, state.lap1)

        source = 0.16 * state.source_map
        cooling = state.edge_cooling * state.heat
        metabolic_load = state.energy_demand
        repair = 0.14 * state.repair_drive

        state.free_energy = (
            state.free_energy
            + 0.19 * lap_free
            + source
            - 0.08 * state.free_energy
            - metabolic_load
        ).clamp(0.0, 1.8)

        state.heat = (
            state.heat
            + 0.14 * lap_heat
            + 0.80 * metabolic_load
            - cooling
            - repair
        ).clamp(0.0, 1.8)

        state.heat = state.heat + 0.01 * torch.relu(state.free_energy - 1.2)
        state.free_energy = (state.free_energy - 0.02 * torch.relu(state.heat - 1.0)).clamp(0.0, 1.8)

