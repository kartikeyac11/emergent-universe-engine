from __future__ import annotations

import math

import numpy as np

from config import SimulationConfig
from organism import OrganismManager
from state import UniverseState, blend_genomes, mutate_genome
from utils import clamp_position, deposit_gaussian


class EvolutionEngine:
    def __init__(self, cfg: SimulationConfig, rng: np.random.Generator, organism_manager: OrganismManager):
        self.cfg = cfg
        self.rng = rng
        self.organism_manager = organism_manager

    def step(self, state: UniverseState) -> None:
        alive = [organism for organism in state.organisms.values() if organism.alive]
        if not alive:
            return

        for organism in alive:
            if organism.age < 10:
                continue
            threshold = organism.genome.reproduction_threshold
            if organism.energy_reserve < threshold:
                continue
            if len([record for record in state.organisms.values() if record.alive]) >= self.cfg.max_organisms:
                break

            partner = self._select_partner(state, organism.organism_id)
            if partner is not None:
                child_genome = blend_genomes(organism.genome, partner.genome, self.cfg, self.rng)
            else:
                child_genome = mutate_genome(organism.genome, self.cfg, self.rng)

            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            distance = 8.0 + 10.0 * child_genome.motility
            target = (
                organism.centroid[0] + math.cos(angle) * distance,
                organism.centroid[1] + math.sin(angle) * distance,
            )
            target = clamp_position(state.cfg.grid_size, *target)
            reserve = max(0.18, 0.46 * organism.energy_reserve)
            child_id = self.organism_manager.create_child(state, target, organism.organism_id, child_genome, reserve)
            organism.energy_reserve *= 0.52

            state.biomass = deposit_gaussian(state.biomass, state.grid_x, state.grid_y, target[0], target[1], 0.34, 0.06).clamp(0.0, 1.0)
            state.chem[:, 1:2] = deposit_gaussian(
                state.chem[:, 1:2], state.grid_x, state.grid_y, target[0], target[1], 0.28, 0.07
            ).clamp(0.0, 1.5)
            state.signal = deposit_gaussian(
                state.signal, state.grid_x, state.grid_y, target[0], target[1], 0.42 * child_genome.signaling, 0.08
            ).clamp(-1.0, 1.0)
            state.free_energy = deposit_gaussian(
                state.free_energy, state.grid_x, state.grid_y, target[0], target[1], 0.10, 0.08
            ).clamp(0.0, 1.8)
            state.organisms[child_id].discoveries.append("born")

        for organism in alive:
            if organism.age > 180 and organism.energy_reserve < 0.10:
                self.organism_manager.kill(state, organism.organism_id)

    def _select_partner(self, state: UniverseState, organism_id: int):
        candidates = []
        for key, edge in state.social_edges.items():
            if edge.weight < 0.35:
                continue
            if organism_id not in key:
                continue
            other_id = key[1] if key[0] == organism_id else key[0]
            other = state.organisms.get(other_id)
            if other is None or not other.alive or other.energy_reserve < 0.25:
                continue
            candidates.append((edge.weight + edge.trust - edge.conflict, other))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

