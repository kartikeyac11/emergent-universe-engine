from __future__ import annotations

import math

import numpy as np

from config import SimulationConfig
from state import OrganismRecord, UniverseState, genome_distance, random_genome
from utils import clamp_position, detect_components


class OrganismManager:
    def __init__(self, cfg: SimulationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def current_mask(self, state: UniverseState) -> np.ndarray:
        mask = (
            (state.biomass > 0.54)
            & (state.free_energy > 0.16)
            & (state.heat < 0.90)
            & (state.chem[:, 1:2] > 0.14)
        )[0, 0]
        return mask.detach().cpu().numpy()

    def refresh(self, state: UniverseState) -> None:
        mask = self.current_mask(state)
        signal = state.signal[0, 0].detach().cpu().numpy()
        heat = state.heat[0, 0].detach().cpu().numpy()
        free_energy = state.free_energy[0, 0].detach().cpu().numpy()
        biomass = state.biomass[0, 0].detach().cpu().numpy()
        clusters = detect_components(mask, signal, heat, free_energy, biomass)
        state.clusters = clusters
        state.cluster_map = {}

        for organism in state.organisms.values():
            if organism.alive:
                organism.age += 1
                organism.energy_reserve = max(0.0, organism.energy_reserve - 0.003)

        alive_ids = [organism_id for organism_id, organism in state.organisms.items() if organism.alive]
        candidates: list[tuple[float, int, int]] = []
        for index, cluster in enumerate(clusters):
            for organism_id in alive_ids:
                organism = state.organisms[organism_id]
                distance = math.hypot(cluster.centroid[0] - organism.centroid[0], cluster.centroid[1] - organism.centroid[1])
                limit = 18.0 + math.sqrt(max(cluster.area, organism.area))
                if distance <= limit:
                    candidates.append((distance, organism_id, index))

        assigned_clusters: set[int] = set()
        assigned_organisms: set[int] = set()
        for _, organism_id, index in sorted(candidates, key=lambda item: item[0]):
            if index in assigned_clusters or organism_id in assigned_organisms:
                continue
            cluster = clusters[index]
            organism = state.organisms[organism_id]
            vx = cluster.centroid[0] - organism.centroid[0]
            vy = cluster.centroid[1] - organism.centroid[1]
            organism.centroid = cluster.centroid
            organism.velocity = (0.65 * organism.velocity[0] + 0.35 * vx, 0.65 * organism.velocity[1] + 0.35 * vy)
            organism.area = cluster.area
            organism.last_seen_step = state.step
            organism.energy_reserve = max(0.0, 0.82 * organism.energy_reserve + 0.36 * cluster.mean_free_energy + 0.24 * cluster.mean_biomass)
            organism.fitness = organism.energy_reserve + 0.004 * organism.age
            assigned_clusters.add(index)
            assigned_organisms.add(organism_id)
            state.cluster_map[index] = organism_id

        for index, cluster in enumerate(clusters):
            if index in assigned_clusters:
                continue
            if len([organism for organism in state.organisms.values() if organism.alive]) >= self.cfg.max_organisms:
                break
            genome = random_genome(self.cfg, self.rng)
            organism_id = state.next_organism_id
            state.next_organism_id += 1
            state.organisms[organism_id] = OrganismRecord(
                organism_id=organism_id,
                parent_id=None,
                birth_step=state.step,
                last_seen_step=state.step,
                age=0,
                centroid=cluster.centroid,
                velocity=(0.0, 0.0),
                area=cluster.area,
                energy_reserve=cluster.mean_free_energy + cluster.mean_biomass,
                fitness=cluster.mean_free_energy + cluster.mean_biomass,
                genome=genome,
            )
            state.cluster_map[index] = organism_id

        stale_ids: list[int] = []
        for organism_id, organism in state.organisms.items():
            if not organism.alive:
                continue
            stale = state.step - organism.last_seen_step
            too_hot = organism.energy_reserve < 0.015
            if stale > 16 or too_hot:
                stale_ids.append(organism_id)

        for organism_id in stale_ids:
            self.kill(state, organism_id)

        self._update_diversity(state)

    def create_child(
        self,
        state: UniverseState,
        centroid: tuple[float, float],
        parent_id: int,
        genome,
        reserve: float,
    ) -> int:
        organism_id = state.next_organism_id
        state.next_organism_id += 1
        state.organisms[organism_id] = OrganismRecord(
            organism_id=organism_id,
            parent_id=parent_id,
            birth_step=state.step,
            last_seen_step=state.step,
            age=0,
            centroid=clamp_position(state.cfg.grid_size, *centroid),
            velocity=(0.0, 0.0),
            area=0,
            energy_reserve=reserve,
            fitness=reserve,
            genome=genome,
        )
        return organism_id

    def kill(self, state: UniverseState, organism_id: int) -> None:
        organism = state.organisms.get(organism_id)
        if organism is None or not organism.alive:
            return
        organism.alive = False
        x, y = organism.centroid
        ix, iy = int(x), int(y)
        window = 5
        x0 = max(0, ix - window)
        x1 = min(state.cfg.grid_size, ix + window)
        y0 = max(0, iy - window)
        y1 = min(state.cfg.grid_size, iy + window)
        state.biomass[:, :, y0:y1, x0:x1] *= 0.7
        state.signal[:, :, y0:y1, x0:x1] *= 0.5

    def _update_diversity(self, state: UniverseState) -> None:
        genomes = [organism.genome for organism in state.organisms.values() if organism.alive]
        if len(genomes) < 2:
            state.diversity = 0.0
            return
        distances: list[float] = []
        for i, first in enumerate(genomes):
            for second in genomes[i + 1 :]:
                distances.append(genome_distance(first, second))
        state.diversity = float(np.mean(distances)) if distances else 0.0

