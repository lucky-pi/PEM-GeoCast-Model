"""PEM-GeoCast model architecture used in the final Compact-B experiments.

This file contains only the network definition and directly required custom
layers. It intentionally excludes data preparation, training, evaluation,
checkpoints, graph assets, and experiment orchestration.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DISEASES = ("ASF", "FMD", "PPR")
HORIZONS = (1, 2, 3)
EXPECTED_NODES = 3236
EXPECTED_INPUT_DIM = 155
EXPECTED_HISTORY = 6
EXPECTED_EVENT_CHANNELS = 3

RELATION_NAMES = (
    "LOCAL_SUSCEPTIBILITY",
    "GEO_EXACT_HOP1",
    "GEO_EXACT_HOP2",
    "GEO_EXACT_HOP3",
    "ADAPTIVE_TOPK",
)


class ExactShellMean(nn.Module):
    """
    Incoming mean aggregation on one exact/exclusive geographic shell.

        y_i = mean_{j in shell(i)} x_j

    No self loop is added here. Own/local information is a separate relation
    expert, preserving interpretability of the relation router.

    Sparse accumulation is forced to float32 for torch 1.12 AMP safety.
    """

    def __init__(self, edge_index_np: np.ndarray, n_nodes: int):
        super().__init__()

        e = np.asarray(edge_index_np, dtype=np.int64)
        if e.ndim != 2:
            raise ValueError(f"edge_index must be rank-2, got {e.shape}")
        if e.shape[0] != 2 and e.shape[1] == 2:
            e = e.T
        if e.shape[0] != 2:
            raise ValueError(f"Expected edge_index [2,E], got {e.shape}")

        src = torch.from_numpy(e[0].copy()).long()
        dst = torch.from_numpy(e[1].copy()).long()

        if src.numel() == 0:
            raise ValueError("Empty exact-shell edge set")
        if int(src.min()) < 0 or int(dst.min()) < 0:
            raise ValueError("Negative graph node index")
        if int(src.max()) >= n_nodes or int(dst.max()) >= n_nodes:
            raise ValueError("Graph node index outside frozen ADM1 axis")
        if bool(torch.any(src == dst)):
            raise ValueError("Exact shell unexpectedly contains self-edge")

        degree = torch.zeros(n_nodes, dtype=torch.float32)
        degree.index_add_(
            0,
            dst,
            torch.ones_like(dst, dtype=torch.float32),
        )
        degree = degree.clamp_min(1.0)
        weight = 1.0 / degree[dst]

        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("weight", weight)
        self.n_nodes = int(n_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B_like, N, H]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [B,N,H], got {tuple(x.shape)}")
        if x.shape[1] != self.n_nodes:
            raise ValueError(
                f"Expected {self.n_nodes} nodes, got {x.shape[1]}"
            )

        dtype = x.dtype
        x32 = x.float()

        msg = (
            x32[:, self.src, :]
            * self.weight.float().view(1, -1, 1)
        )
        out = torch.zeros(
            x32.shape,
            dtype=torch.float32,
            device=x.device,
        )
        out.index_add_(1, self.dst, msg)
        return out.to(dtype=dtype)


class SparseAdaptiveRelation(nn.Module):
    """
    Universal learned ADM1 relation used ONLY inside the Emergence expert.

    S_ij = ReLU(E_target_i E_source_j^T)
    A_i  = softmax(TopK_j S_ij)

    Relation topology is learned, but disease/horizon dependence is handled
    by the downstream Disease-Horizon Relation Router. This keeps the core
    experiment interpretable and computationally tractable.
    """

    def __init__(
        self,
        n_nodes: int,
        embed_dim: int = 16,
        topk: int = 20,
    ):
        super().__init__()

        if topk < 1 or topk >= n_nodes:
            raise ValueError(
                "adaptive top-k must be in [1, n_nodes-1]"
            )

        self.n_nodes = int(n_nodes)
        self.embed_dim = int(embed_dim)
        self.topk = int(topk)

        scale = 1.0 / math.sqrt(embed_dim)
        self.target_emb = nn.Parameter(
            torch.randn(n_nodes, embed_dim) * scale
        )
        self.source_emb = nn.Parameter(
            torch.randn(n_nodes, embed_dim) * scale
        )

        # The degree vector is recreated from n_nodes and the supplied edges.
        self.register_buffer(
            "self_mask",
            torch.eye(n_nodes, dtype=torch.bool),
            persistent=False,
        )

    def topk_relation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stable FP32 affinity even under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            score = torch.matmul(
                self.target_emb.float(),
                self.source_emb.float().t(),
            )
            score = F.relu(score)

            # Out-of-place masking: avoids the ReLU in-place autograd bug
            # encountered in the Graph WaveNet baseline.
            score = score.masked_fill(
                self.self_mask,
                -1e9,
            )

            value, index = torch.topk(
                score,
                k=self.topk,
                dim=1,
                largest=True,
                sorted=False,
            )
            weight = torch.softmax(value, dim=1)

        return index, weight

    def forward(
        self,
        x: torch.Tensor,
        relation: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        x: [B_like, N, H]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [B,N,H], got {tuple(x.shape)}")
        if x.shape[1] != self.n_nodes:
            raise ValueError(
                f"Expected {self.n_nodes} nodes, got {x.shape[1]}"
            )

        dtype = x.dtype
        x32 = x.float()

        if relation is None:
            index, weight = self.topk_relation()
        else:
            index, weight = relation

        # [B_like, target, k, H]
        gathered = x32[:, index, :]
        out = (
            gathered
            * weight.float().view(
                1,
                self.n_nodes,
                self.topk,
                1,
            )
        ).sum(dim=2)

        return out.to(dtype=dtype)


class TemporalAttentionPool(nn.Module):
    """
    Learn a causal attention-weighted summary within a fixed trailing window.
    """

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.score = nn.Linear(hidden, 1)
        self.post = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B,N,T,H]
        """
        a = self.score(z).squeeze(-1)
        a = torch.softmax(a, dim=2)
        pooled = (
            z * a.unsqueeze(-1)
        ).sum(dim=2)
        return self.post(pooled)


class MultiScaleTemporalEncoder(nn.Module):
    """
    Explicit 1/3/6-month temporal states.

    The branch outputs are mixed by a learned local scale router.
    Scale weights are saved for later interpretation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        dropout: float,
    ):
        super().__init__()

        self.in_norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pool1 = TemporalAttentionPool(hidden, dropout)
        self.pool3 = TemporalAttentionPool(hidden, dropout)
        self.pool6 = TemporalAttentionPool(hidden, dropout)

        self.scale_score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Start near equal use of the three temporal scales.
        nn.init.zeros_(self.scale_score[-1].weight)
        nn.init.zeros_(self.scale_score[-1].bias)

        self.out_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B,N,6,F_local]
        returns:
          fused          [B,N,H]
          scale_weights  [B,N,3]
          scale_states   [B,N,3,H]
        """
        if x.shape[2] != EXPECTED_HISTORY:
            raise ValueError(
                f"Expected 6 history months, got {x.shape[2]}"
            )

        z = self.in_proj(self.in_norm(x))

        z1 = self.pool1(z[:, :, -1:, :])
        z3 = self.pool3(z[:, :, -3:, :])
        z6 = self.pool6(z)

        scales = torch.stack([z1, z3, z6], dim=2)
        score = self.scale_score(scales).squeeze(-1)
        weight = torch.softmax(score, dim=2)

        fused = (
            scales * weight.unsqueeze(-1)
        ).sum(dim=2)

        return self.out_norm(fused), weight, scales


def disease_history_statistics(
    event_hist: torch.Tensor,
) -> torch.Tensor:
    """
    event_hist: [B,N,6,D], past-only event_observed_main channels.

    Returns six transparent state variables per disease:
      lag1
      any_last3
      any_last6
      frequency_last3
      frequency_last6
      recency_closeness = 1 - months_since_last_event/6

    No reporting/supervision masks are consumed.
    """
    if event_hist.ndim != 4:
        raise ValueError(
            f"Expected event_hist [B,N,T,D], got {tuple(event_hist.shape)}"
        )
    if event_hist.shape[2] != 6:
        raise ValueError("Disease history must have six months")
    if event_hist.shape[3] != len(DISEASES):
        raise ValueError("Disease-history disease axis mismatch")

    # Sanitize defensively.
    e = torch.nan_to_num(
        event_hist.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    lag1 = e[:, :, -1, :]
    last3 = e[:, :, -3:, :]
    any3 = last3.amax(dim=2)
    any6 = e.amax(dim=2)
    freq3 = last3.mean(dim=2)
    freq6 = e.mean(dim=2)

    # Oldest -> newest time positions: 1..6.
    # Last positive position gives months since recent positive.
    pos = torch.arange(
        1,
        7,
        device=e.device,
        dtype=e.dtype,
    ).view(1, 1, 6, 1)

    last_pos = (e * pos).amax(dim=2)
    months_since = torch.where(
        any6 > 0,
        6.0 - last_pos,
        torch.full_like(last_pos, 6.0),
    )
    recency_closeness = 1.0 - months_since / 6.0

    return torch.stack(
        [
            lag1,
            any3,
            any6,
            freq3,
            freq6,
            recency_closeness,
        ],
        dim=-1,
    )


class DiseaseHistoryEncoder(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(
        self,
        event_hist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = disease_history_statistics(event_hist)
        return self.net(stats), stats


class PEMGeoCast(nn.Module):
    """Final frozen PEM-GeoCast Compact-B architecture.

    The trainable forward graph uses the full six-month temporal state,
    disease-history encoding, local persistence expert, geographic emergence
    expert over LOCAL/GEO1/GEO2/GEO3, fixed equal relation fusion, and a learned
    disease- and horizon-conditional persistence/emergence gate.

    The adaptive graph and learned temporal/relation routers remain instantiated
    but frozen solely for strict compatibility with the formal checkpoints.
    """

    def __init__(
            self,
            input_dim: int,
            geo_edges: Dict[int, np.ndarray],
            n_nodes: int = EXPECTED_NODES,
            hidden: int = 128,
            embed_dim: int = 16,
            adaptive_topk: int = 20,
            dropout: float = 0.20,
        ):
            super().__init__()

            if input_dim != EXPECTED_INPUT_DIM:
                raise ValueError(
                    f"Stage3E-0 froze input_dim={EXPECTED_INPUT_DIM}, "
                    f"got {input_dim}"
                )

            self.input_dim = int(input_dim)
            self.n_nodes = int(n_nodes)
            self.hidden = int(hidden)
            self.embed_dim = int(embed_dim)
            self.adaptive_topk = int(adaptive_topk)
            self.dropout_p = float(dropout)

            self.n_diseases = len(DISEASES)
            self.n_horizons = len(HORIZONS)
            self.n_relations = len(RELATION_NAMES)

            # First 3 channels are frozen disease-event history channels.
            self.local_input_dim = input_dim - EXPECTED_EVENT_CHANNELS

            self.temporal = MultiScaleTemporalEncoder(
                input_dim=self.local_input_dim,
                hidden=hidden,
                dropout=dropout,
            )
            self.history_encoder = DiseaseHistoryEncoder(
                hidden=hidden,
                dropout=dropout,
            )

            self.disease_embedding = nn.Embedding(
                self.n_diseases,
                hidden,
            )
            self.horizon_embedding = nn.Embedding(
                self.n_horizons,
                hidden,
            )

            # Persistence is node-local by construction: no graph operator.
            self.persistence_base = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.persistence_task = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Emergence source state combines local susceptibility with
            # disease-specific past-event state.
            self.emergence_source = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
            )

            self.geo1 = ExactShellMean(
                geo_edges[1],
                n_nodes,
            )
            self.geo2 = ExactShellMean(
                geo_edges[2],
                n_nodes,
            )
            self.geo3 = ExactShellMean(
                geo_edges[3],
                n_nodes,
            )

            self.adaptive = SparseAdaptiveRelation(
                n_nodes=n_nodes,
                embed_dim=embed_dim,
                topk=adaptive_topk,
            )

            # Separate post-projections retain relation identity.
            self.local_rel_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.geo1_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.geo2_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.geo3_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.adapt_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )

            # Disease x Horizon conditional relation router.
            self.router_state = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.relation_router = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, self.n_relations),
            )
            # Unbiased initial condition: uniform relation routing.
            nn.init.zeros_(self.relation_router[-1].weight)
            nn.init.zeros_(self.relation_router[-1].bias)

            self.emergence_task = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # P/E gate depends only on past state + disease/horizon embeddings.
            self.gate_hist_proj = nn.Sequential(
                nn.Linear(6, hidden),
                nn.GELU(),
            )
            self.pe_gate = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
            # Neutral prior P=E=0.5 at initialization.
            nn.init.zeros_(self.pe_gate[-1].weight)
            nn.init.zeros_(self.pe_gate[-1].bias)

            self.final_norm = nn.LayerNorm(hidden)
            self.output_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

            # Allows different base prevalence for nine tasks without changing
            # the shared mechanism representation.
            self.task_bias = nn.Parameter(
                torch.zeros(
                    self.n_horizons,
                    self.n_diseases,
                )
            )

            # These two modules were instantiated in the frozen Compact-B run and
            # remain present for exact state-dictionary compatibility. They are not
            # used by the PEM-GeoCast forward graph.
            self.single_source = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.single_task = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self._freeze_inactive_compact_b_modules()


    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _freeze_inactive_compact_b_modules(self) -> None:
        self._freeze_module(self.adaptive)
        self._freeze_module(self.adapt_proj)
        self._freeze_module(self.temporal.pool1)
        self._freeze_module(self.temporal.pool3)
        self._freeze_module(self.temporal.scale_score)
        self._freeze_module(self.router_state)
        self._freeze_module(self.relation_router)
        self._freeze_module(self.single_source)
        self._freeze_module(self.single_task)

    def _task_embeddings(
            self,
            device,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            d = self.disease_embedding(
                torch.arange(
                    self.n_diseases,
                    device=device,
                )
            )
            h = self.horizon_embedding(
                torch.arange(
                    self.n_horizons,
                    device=device,
                )
            )
            return d, h

    @staticmethod
    def _active_relation_indices() -> List[int]:
        return [0, 1, 2, 3]

    def _relation_weights(
        self,
        router_query: torch.Tensor,
        active: Sequence[int],
    ) -> torch.Tensor:
        shape = (*router_query.shape[:-1], len(RELATION_NAMES))
        weights = torch.zeros(
            shape,
            dtype=router_query.dtype,
            device=router_query.device,
        )
        weights[..., list(active)] = 1.0 / float(len(active))
        return weights

    def _build_relation_stack(
        self,
        source: torch.Tensor,
        batch_size: int,
        n_nodes: int,
    ) -> torch.Tensor:
        e_flat = (
            source.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size * self.n_diseases, n_nodes, self.hidden)
        )

        zero = torch.zeros_like(source)
        r_local = self.local_rel_proj(source)
        r_geo1 = self.geo1_proj(
            self.geo1(e_flat)
            .reshape(batch_size, self.n_diseases, n_nodes, self.hidden)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        r_geo2 = self.geo2_proj(
            self.geo2(e_flat)
            .reshape(batch_size, self.n_diseases, n_nodes, self.hidden)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        r_geo3 = self.geo3_proj(
            self.geo3(e_flat)
            .reshape(batch_size, self.n_diseases, n_nodes, self.hidden)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        r_adapt = zero
        return torch.stack(
            [r_local, r_geo1, r_geo2, r_geo3, r_adapt],
            dim=3,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,N,T,F], got {tuple(x.shape)}")

        batch_size, n_nodes, history, input_dim = x.shape
        if n_nodes != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {n_nodes}")
        if history != EXPECTED_HISTORY:
            raise ValueError(f"Expected history={EXPECTED_HISTORY}, got {history}")
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")

        event_hist = x[..., :EXPECTED_EVENT_CHANNELS]
        local_x = x[..., EXPECTED_EVENT_CHANNELS:]

        local_state, scale_weight, scale_states = self.temporal(local_x)
        local_state = self.temporal.out_norm(scale_states[:, :, 2, :])
        scale_weight = torch.zeros_like(scale_weight)
        scale_weight[:, :, 2] = 1.0

        hist_state, hist_stats = self.history_encoder(event_hist)
        local_for_d = local_state.unsqueeze(2).expand(
            -1, -1, self.n_diseases, -1
        )
        state_pair = torch.cat([local_for_d, hist_state], dim=-1)

        disease_emb, horizon_emb = self._task_embeddings(x.device)
        active = self._active_relation_indices()

        p_base = self.persistence_base(state_pair)
        p_task = (
            p_base.unsqueeze(2)
            + disease_emb.view(1, 1, 1, self.n_diseases, self.hidden)
            + horizon_emb.view(1, 1, self.n_horizons, 1, self.hidden)
        )
        p_task = self.persistence_task(p_task)

        e_source = self.emergence_source(state_pair)
        relation_stack = self._build_relation_stack(
            e_source, batch_size, n_nodes
        )

        router_base = self.router_state(state_pair)
        router_query = (
            router_base.unsqueeze(2)
            + disease_emb.view(1, 1, 1, self.n_diseases, self.hidden)
            + horizon_emb.view(1, 1, self.n_horizons, 1, self.hidden)
        )
        relation_weight = self._relation_weights(router_query, active)

        e_task = (
            relation_stack.unsqueeze(2) * relation_weight.unsqueeze(-1)
        ).sum(dim=-2)
        e_task = self.emergence_task(e_task)

        gate_hist = self.gate_hist_proj(hist_stats)
        gate_query = (
            local_for_d.unsqueeze(2)
            + gate_hist.unsqueeze(2)
            + disease_emb.view(1, 1, 1, self.n_diseases, self.hidden)
            + horizon_emb.view(1, 1, self.n_horizons, 1, self.hidden)
        )
        emergence_gate = torch.sigmoid(self.pe_gate(gate_query).squeeze(-1))

        final = (
            (1.0 - emergence_gate.unsqueeze(-1)) * p_task
            + emergence_gate.unsqueeze(-1) * e_task
        )
        final = self.final_norm(final)

        logits = self.output_head(final).squeeze(-1)
        logits = logits + self.task_bias.view(
            1, 1, self.n_horizons, self.n_diseases
        )

        if not return_diagnostics:
            return logits

        return logits, {
            "pe_gate": emergence_gate,
            "relation_weights": relation_weight,
            "scale_weights": scale_weight,
            "history_stats": hist_stats,
        }
