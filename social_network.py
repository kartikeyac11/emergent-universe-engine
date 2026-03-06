from __future__ import annotations

import math

from state import SocialEdge, UniverseState


class SocialNetworkEngine:
    def step(self, state: UniverseState) -> None:
        alive = [organism for organism in state.organisms.values() if organism.alive]
        for key in list(state.social_edges.keys()):
            edge = state.social_edges[key]
            edge.weight *= 0.985
            edge.trust *= 0.992
            edge.conflict *= 0.992
            if edge.weight < 0.03 and edge.trust < 0.03 and edge.conflict < 0.03:
                del state.social_edges[key]

        for i, first in enumerate(alive):
            for second in alive[i + 1 :]:
                distance = math.hypot(first.centroid[0] - second.centroid[0], first.centroid[1] - second.centroid[1])
                if distance > 34.0:
                    continue
                key = tuple(sorted((first.organism_id, second.organism_id)))
                edge = state.social_edges.get(
                    key,
                    SocialEdge(first.organism_id, second.organism_id, weight=0.0, trust=0.0, conflict=0.0, last_step=state.step),
                )

                signal_gap = abs(first.genome.signaling - second.genome.signaling)
                cooperation = max(0.0, 1.0 - signal_gap) * 0.03
                competition = max(0.0, 0.03 - 0.0008 * distance)
                if first.energy_reserve > second.energy_reserve + 0.20 and first.genome.cooperation > 0.55:
                    transfer = min(0.03, first.energy_reserve * 0.05)
                    first.energy_reserve -= transfer
                    second.energy_reserve += transfer
                    cooperation += 0.05
                elif distance < 12.0:
                    competition += 0.02

                edge.weight = min(1.5, edge.weight + cooperation - 0.5 * competition + 0.02)
                edge.trust = min(1.0, edge.trust + cooperation)
                edge.conflict = min(1.0, edge.conflict + competition)
                edge.last_step = state.step
                state.social_edges[key] = edge

                if edge.trust > 0.35 and first.discoveries and not second.discoveries:
                    second.discoveries.append(first.discoveries[-1])
                if edge.trust > 0.45 and second.discoveries and not first.discoveries:
                    first.discoveries.append(second.discoveries[-1])

