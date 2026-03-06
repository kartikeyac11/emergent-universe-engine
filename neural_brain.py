from __future__ import annotations

import math

import numpy as np

from config import SimulationConfig
from state import OrganismRecord, UniverseState
from utils import clamp_position, deposit_gaussian


class NeuralBrainEngine:
    def __init__(self, cfg: SimulationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def step(self, state: UniverseState) -> None:
        alive = [organism for organism in state.organisms.values() if organism.alive]
        for organism in alive:
            observation = self._observe(state, organism)
            if organism.last_observation is not None and organism.last_action is not None:
                reward = self._reward(observation, organism)
                next_obs = observation.astype(np.float32)
                organism.memory.append((organism.last_observation, organism.last_action, reward, next_obs))
                if len(organism.memory) > self.cfg.max_memory:
                    organism.memory.pop(0)
                self._update_world_model(organism)

            action = self._forward(organism, observation)
            self._apply_action(state, organism, action, observation)
            organism.last_observation = observation.astype(np.float32)
            organism.last_action = action.astype(np.float32)

            if observation[1] < organism.genome.heat_tolerance and organism.energy_reserve > 0.25:
                self._dream(organism)

    def _observe(self, state: UniverseState, organism: OrganismRecord) -> np.ndarray:
        x, y = clamp_position(state.cfg.grid_size, *organism.centroid)
        ix = int(round(x))
        iy = int(round(y))
        degree = sum(1 for key in state.social_edges if organism.organism_id in key)
        obs = np.array(
            [
                float(state.free_energy[0, 0, iy, ix].item()),
                float(state.heat[0, 0, iy, ix].item()),
                float(state.signal[0, 0, iy, ix].item()),
                float(state.biomass[0, 0, iy, ix].item()),
                float(state.chem[0, 0, iy, ix].item()),
                float(state.chem[0, 1, iy, ix].item()),
                float(state.chem[0, 2, iy, ix].item()),
                float(organism.energy_reserve),
                float(min(organism.age / 200.0, 1.0)),
                float(min(degree / 8.0, 1.0)),
                float(self.rng.uniform(-1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return obs

    def _forward(self, organism: OrganismRecord, observation: np.ndarray) -> np.ndarray:
        hidden = np.tanh(observation @ organism.genome.w1 + organism.genome.b1)
        actions = np.tanh(hidden @ organism.genome.w2 + organism.genome.b2)
        return (actions * organism.genome.brain_scale).astype(np.float32)

    def _apply_action(self, state: UniverseState, organism: OrganismRecord, action: np.ndarray, observation: np.ndarray) -> None:
        dx = float(action[0] * organism.genome.motility * 2.8)
        dy = float(action[1] * organism.genome.motility * 2.8)
        signal_emit = float(action[2] * organism.genome.signaling)
        harvest = float(max(action[3], 0.0) * organism.genome.metabolism)
        repair = float(max(action[4], 0.0))
        reproduce_bias = float(max(action[5], 0.0))

        target = clamp_position(state.cfg.grid_size, organism.centroid[0] + dx, organism.centroid[1] + dy)
        organism.centroid = target
        organism.velocity = (0.7 * organism.velocity[0] + 0.3 * dx, 0.7 * organism.velocity[1] + 0.3 * dy)

        state.signal_drive = deposit_gaussian(
            state.signal_drive, state.grid_x, state.grid_y, target[0], target[1], 0.18 * signal_emit, 0.05
        ).clamp(-1.0, 1.0)
        state.repair_drive = deposit_gaussian(
            state.repair_drive, state.grid_x, state.grid_y, target[0], target[1], 0.12 * repair, 0.05
        ).clamp(0.0, 1.0)
        state.energy_demand = deposit_gaussian(
            state.energy_demand,
            state.grid_x,
            state.grid_y,
            target[0],
            target[1],
            0.02 + 0.04 * harvest + 0.02 * abs(signal_emit),
            0.05,
        ).clamp(0.0, 1.5)
        state.biomass = deposit_gaussian(
            state.biomass, state.grid_x, state.grid_y, target[0], target[1], 0.04 * (harvest + repair), 0.05
        ).clamp(0.0, 1.0)
        state.membrane = deposit_gaussian(
            state.membrane, state.grid_x, state.grid_y, target[0], target[1], 0.03 * repair, 0.05
        ).clamp(0.0, 1.0)

        energy_gain = max(0.0, observation[0] * (0.12 + 0.10 * harvest) - observation[1] * 0.06)
        energy_cost = 0.02 + 0.018 * abs(dx) + 0.018 * abs(dy) + 0.03 * abs(signal_emit) + 0.04 * repair
        organism.energy_reserve = max(0.0, organism.energy_reserve + energy_gain - energy_cost)
        organism.fitness = organism.energy_reserve + 0.005 * organism.age + 0.08 * reproduce_bias

    def _reward(self, observation: np.ndarray, organism: OrganismRecord) -> float:
        safe_obs = np.nan_to_num(observation, nan=0.0, posinf=2.0, neginf=-2.0)
        reserve = float(np.nan_to_num(organism.energy_reserve, nan=0.0, posinf=2.0, neginf=0.0))
        reward = safe_obs[0] * 0.8 - safe_obs[1] * 0.7 + safe_obs[3] * 0.4 + reserve * 0.5
        return float(reward)

    def _update_world_model(self, organism: OrganismRecord) -> None:
        obs, action, _, next_obs = organism.memory[-1]
        model_input = np.clip(np.concatenate([obs, action]).astype(np.float32), -2.0, 2.0)
        predicted = np.clip(model_input @ organism.genome.world_model, -2.0, 2.0)
        error = np.clip(next_obs - predicted, -1.5, 1.5)
        updated = organism.genome.world_model + organism.genome.dream_gain * np.outer(model_input, error)
        organism.genome.world_model = np.clip(np.nan_to_num(updated, nan=0.0, posinf=4.0, neginf=-4.0), -4.0, 4.0).astype(np.float32)

    def _dream(self, organism: OrganismRecord) -> None:
        if not organism.memory:
            return
        organism.dream_counter += 1
        obs, _, _, _ = organism.memory[-1]
        base_action = self._forward(organism, obs)
        candidates = [base_action]
        for _ in range(3):
            candidates.append(base_action + self.rng.normal(0.0, organism.genome.dream_gain, size=base_action.shape).astype(np.float32))

        best_action = base_action
        best_score = -1e9
        for candidate in candidates:
            model_input = np.clip(np.concatenate([obs, candidate]).astype(np.float32), -2.0, 2.0)
            predicted_obs = np.clip(np.nan_to_num(model_input @ organism.genome.world_model, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0)
            score = self._reward(predicted_obs, organism) - 0.03 * float(np.linalg.norm(candidate))
            if score > best_score:
                best_score = score
                best_action = candidate

        organism.genome.b2 = (
            0.96 * organism.genome.b2 + 0.04 * best_action + organism.genome.dream_gain * 0.01
        ).clip(-2.0, 2.0).astype(np.float32)
        organism.fitness += 0.01 * math.tanh(best_score)
