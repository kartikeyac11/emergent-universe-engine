from __future__ import annotations

import torch

from state import UniverseState
from utils import laplace, orthogonal_matrix


class PhysicsEngine:
    def __init__(self, state: UniverseState):
        d = state.cfg.internal_dim
        device = state.device
        self.d = d
        self.horizontal_gate = orthogonal_matrix(2 * d, device)
        self.vertical_gate = orthogonal_matrix(2 * d, device)
        self.mix = orthogonal_matrix(d, device)
        self.phase_bias = torch.linspace(0.0, 0.55, steps=d, device=device, dtype=torch.float32)

    def _apply_horizontal(self, psi: torch.Tensor, gate: torch.Tensor, start: int) -> torch.Tensor:
        width = psi.shape[3]
        indices = torch.arange(start, width, 2, device=psi.device)
        partners = (indices + 1) % width
        left = psi.index_select(3, indices)
        right = psi.index_select(3, partners)
        pair = torch.cat([left, right], dim=1).permute(0, 2, 3, 1)
        transformed = torch.einsum("ij,bhwj->bhwi", gate, pair).permute(0, 3, 1, 2)
        psi = psi.clone()
        psi.index_copy_(3, indices, transformed[:, : self.d])
        psi.index_copy_(3, partners, transformed[:, self.d :])
        return psi

    def _apply_vertical(self, psi: torch.Tensor, gate: torch.Tensor, start: int) -> torch.Tensor:
        height = psi.shape[2]
        indices = torch.arange(start, height, 2, device=psi.device)
        partners = (indices + 1) % height
        top = psi.index_select(2, indices)
        bottom = psi.index_select(2, partners)
        pair = torch.cat([top, bottom], dim=1).permute(0, 2, 3, 1)
        transformed = torch.einsum("ij,bhwj->bhwi", gate, pair).permute(0, 3, 1, 2)
        psi = psi.clone()
        psi.index_copy_(2, indices, transformed[:, : self.d])
        psi.index_copy_(2, partners, transformed[:, self.d :])
        return psi

    def step(self, state: UniverseState) -> None:
        phase_shift = torch.sin(self.phase_bias.view(1, -1, 1, 1) + state.step * 0.021)
        if state.step % 2 == 0:
            psi = self._apply_horizontal(state.psi, self.horizontal_gate, start=(state.step // 2) % 2)
        else:
            psi = self._apply_vertical(state.psi, self.vertical_gate, start=(state.step // 2) % 2)

        lap_psi = laplace(psi, state.lap4)
        mixed = torch.einsum("ij,bjhw->bihw", self.mix, psi)
        curl = torch.roll(psi, shifts=1, dims=2) - torch.roll(psi, shifts=-1, dims=3)
        drive = 0.11 * torch.sin(mixed * 1.8 + phase_shift)
        psi = psi + 0.18 * lap_psi + 0.14 * mixed + 0.06 * curl + drive - 0.12 * psi
        psi = psi / (psi.norm(dim=1, keepdim=True) + 1e-4)

        state.psi = psi
        state.phase = torch.atan2(psi[:, 1:2], psi[:, 0:1])
        state.coherence = 0.5 + 0.5 * torch.tanh(psi[:, :2].norm(dim=1, keepdim=True) * 1.7 - 0.9)
        state.matter_type = torch.argmax(torch.abs(psi), dim=1, keepdim=True).to(torch.float32)

