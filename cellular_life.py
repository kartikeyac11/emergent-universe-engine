from __future__ import annotations

import torch

from state import UniverseState
from utils import laplace


class CellularLifeEngine:
    def step(self, state: UniverseState) -> None:
        lap_chem = laplace(state.chem, state.lap3)
        lap_biomass = laplace(state.biomass, state.lap1)
        lap_signal = laplace(state.signal, state.lap1)

        a = state.chem[:, 0:1]
        b = state.chem[:, 1:2]
        nutrient = state.chem[:, 2:3]

        free_norm = state.free_energy / (state.free_energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
        heat_norm = state.heat / (state.heat.amax(dim=(2, 3), keepdim=True) + 1e-6)
        phase_wave = torch.sin(state.phase * 2.4)

        feed = 0.024 + 0.020 * state.source_map + 0.016 * state.coherence
        kill = 0.048 + 0.020 * heat_norm + 0.010 * (1.0 - state.coherence)
        reaction = a * b * b
        substrate_push = 0.05 * torch.sigmoid(phase_wave + state.coherence * 1.1)

        a = a + 0.22 * lap_chem[:, 0:1] - reaction + feed * (1.0 - a)
        b = b + 0.11 * lap_chem[:, 1:2] + reaction - (kill + feed) * b + substrate_push * nutrient
        nutrient = (
            nutrient
            + 0.14 * lap_chem[:, 2:3]
            + 0.12 * state.source_map
            - 0.08 * nutrient
            - 0.05 * reaction
        )
        state.chem = torch.cat([a, b.clamp(0.0, 1.5), nutrient.clamp(0.0, 1.4)], dim=1)

        life_potential = torch.sigmoid(
            4.3 * (state.chem[:, 1:2] - 0.22)
            + 2.4 * (free_norm - 0.30)
            - 3.1 * heat_norm
            + 1.8 * state.coherence
            + 0.8 * phase_wave
        )

        biomass_growth = 0.14 * state.chem[:, 1:2] * torch.relu(state.free_energy - 0.18)
        state.biomass = (
            state.biomass
            + 0.12 * lap_biomass
            + 0.34 * (life_potential - state.biomass)
            + biomass_growth
        ).clamp(0.0, 1.0)

        membrane_drive = torch.sigmoid(5.0 * (state.biomass - 0.45)) * torch.sigmoid(4.0 * (state.chem[:, 1:2] - 0.18))
        state.membrane = (0.82 * state.membrane + 0.18 * membrane_drive).clamp(0.0, 1.0)

        state.signal = (
            state.signal
            + 0.16 * lap_signal
            + 0.22 * torch.relu(state.biomass - 0.34) * torch.tanh(state.psi[:, 2:3] + 0.3 * state.psi[:, 3:4])
            + 0.36 * state.signal_drive
            - 0.20 * state.signal
        ).clamp(-1.0, 1.0)

        state.energy_demand.add_(0.02 * reaction + 0.03 * torch.relu(state.biomass - 0.30))

