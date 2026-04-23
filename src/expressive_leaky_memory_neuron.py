import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .modeling_utils import (
    create_interlocking_indices,
    create_overlapping_window_indices,
)
from .neuronio.neuronio_data_utils import DEFAULT_Y_TRAIN_SOMA_SCALE

PREPROCESS_CONFIGURATIONS = [None, "random_routing", "neuronio_routing"]


class ThresholdUnit(nn.Module):
    """
    Interpretable nonlinear unit: weighted sum -> learnable threshold -> sigmoid.
    """

    def __init__(self, in_dim: int, out_dim: int, gain_init: float = 5.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.threshold = nn.Parameter(torch.zeros(out_dim))
        self.log_gain = nn.Parameter(torch.full((out_dim,), float(np.log(gain_init))))
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        gain = torch.exp(self.log_gain)
        return torch.sigmoid(gain * (z - self.threshold))


class GIADA_L4P1(nn.Module):
    """
    GIADA neuron with interpretable compartment groups.

    GIADA stands for Grouped Interpretable Adaptive Dendritic Architecture.

    The public interface intentionally matches the previous `ELM` class so the
    training and evaluation utilities in this repository keep working.
    """

    def __init__(
        self,
        num_input: int,
        num_output: int,
        num_memory: int = 20,
        lambda_value: float = 10.0,
        mlp_num_layers: int = 1,
        mlp_hidden_size: Optional[int] = None,
        mlp_activation: str = "relu",
        tau_s_value: float = 5.0,
        memory_tau_min: float = 1.0,
        memory_tau_max: float = 150.0,
        learn_memory_tau: bool = True,
        w_s_value: float = 0.5,
        num_branch: Optional[int] = 45,
        num_synapse_per_branch: int = 1,
        input_to_synapse_routing: Optional[str] = None,
        delta_t: float = 1.0,
        apical_branch_start: int = 30,
    ):
        super().__init__()

        del num_memory, mlp_num_layers, mlp_hidden_size, mlp_activation
        del memory_tau_min, memory_tau_max, learn_memory_tau

        self.num_input = num_input
        self.num_output = num_output
        self.lambda_value = lambda_value
        self.tau_s_value = tau_s_value
        self.w_s_value = w_s_value
        self.delta_t = delta_t
        self.num_branch = 45 if num_branch is None else num_branch
        self.num_synapse_per_branch = num_synapse_per_branch
        self.input_to_synapse_routing = input_to_synapse_routing
        self.apical_start = apical_branch_start

        assert self.input_to_synapse_routing in PREPROCESS_CONFIGURATIONS
        if self.input_to_synapse_routing is None:
            self.num_synapse = self.num_input
        else:
            self.num_synapse = self.num_branch * self.num_synapse_per_branch

        if not (0 <= self.apical_start < self.num_branch):
            raise ValueError("apical_branch_start must be within the branch range.")
        if self.num_output != 2:
            raise ValueError("GIADA_L4P1 currently expects num_output=2.")

        self.g1 = 4
        self.g2 = 6
        self.g3 = 5
        self.g4 = 3
        self.g5 = 2
        self.dm = self.g1 + self.g2 + self.g3 + self.g4 + self.g5
        self.group_sizes = [self.g1, self.g2, self.g3, self.g4, self.g5]
        self.tau_bounds = [
            (1.0, 8.0),
            (8.0, 35.0),
            (30.0, 80.0),
            (60.0, 150.0),
            (100.0, 500.0),
        ]

        self.log_tau_s = nn.Parameter(
            torch.full((self.num_synapse,), float(np.log(tau_s_value)))
        )
        self.w_s_raw = nn.Parameter(torch.full((self.num_synapse,), w_s_value))

        self.register_buffer("branch_indices", self._build_branch_indices())

        log_tau_m_init = []
        log_tau_m_lo = []
        log_tau_m_hi = []
        for (lo, hi), size in zip(self.tau_bounds, self.group_sizes):
            taus = torch.linspace(lo, hi, size)
            log_tau_m_init.append(torch.log(taus))
            log_tau_m_lo.extend([float(np.log(lo))] * size)
            log_tau_m_hi.extend([float(np.log(hi))] * size)

        self.log_tau_m = nn.Parameter(torch.cat(log_tau_m_init))
        self.register_buffer("log_tau_m_lo", torch.tensor(log_tau_m_lo))
        self.register_buffer("log_tau_m_hi", torch.tensor(log_tau_m_hi))

        num_apical = self.num_branch - self.apical_start
        self.g1_proj = nn.Linear(self.num_branch, self.g1, bias=True)
        self.g2_thresh = ThresholdUnit(self.num_branch, self.g2, gain_init=5.0)
        self.g3_proj = nn.Linear(self.g2, self.g3, bias=False)
        self.g3_threshold = nn.Parameter(torch.zeros(self.g3))
        self.g4_thresh = ThresholdUnit(num_apical + self.g1, self.g4, gain_init=3.0)
        self.g5_proj = nn.Linear(self.g1, self.g5, bias=True)
        self.w_y = nn.Linear(self.dm, self.num_output, bias=True)

        routing_artifacts = self.create_input_to_synapse_indices()
        self.input_to_synapse_indices = routing_artifacts[0]
        self.valid_indices_mask = routing_artifacts[1]

    def _build_branch_indices(self) -> torch.Tensor:
        dbrch = max(1, self.num_synapse // self.num_branch)
        indices = []
        stride = (
            max(1, (self.num_synapse - dbrch) // (self.num_branch - 1))
            if self.num_branch > 1
            else 0
        )
        for b in range(self.num_branch):
            start = min(b * stride, max(0, self.num_synapse - dbrch))
            indices.append(torch.arange(start, start + dbrch))
        return torch.stack(indices)

    def create_input_to_synapse_indices(self):
        if self.input_to_synapse_routing == "random_routing":
            input_to_synapse_indices = torch.randint(
                self.num_input, (self.num_synapse,)
            )
            valid_indices_mask = torch.ones(self.num_synapse, dtype=torch.float32)
            return input_to_synapse_indices, valid_indices_mask
        if self.input_to_synapse_routing == "neuronio_routing":
            assert (
                math.ceil(self.num_input / self.num_branch)
                <= self.num_synapse_per_branch
            )
            interlocking_indices = create_interlocking_indices(self.num_input)
            overlapping_indices, valid_indices_mask = create_overlapping_window_indices(
                self.num_input, self.num_branch, self.num_synapse_per_branch
            )
            input_to_synapse_indices = interlocking_indices[overlapping_indices]
            return input_to_synapse_indices, valid_indices_mask.to(torch.float32)
        return None, None

    def route_input_to_synapses(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_to_synapse_routing is None:
            return x
        routed = torch.index_select(x, 2, self.input_to_synapse_indices.to(x.device))
        return routed * self.valid_indices_mask.to(x.device)

    def _get_kappas(self):
        tau_s = torch.nn.functional.softplus(self.log_tau_s) + 1e-3
        log_tau_m = torch.clamp(self.log_tau_m, self.log_tau_m_lo, self.log_tau_m_hi)
        tau_m = torch.exp(log_tau_m)
        kappa_s = torch.exp(-self.delta_t / tau_s)
        kappa_m = torch.exp(-self.delta_t / tau_m)
        return kappa_s, kappa_m

    def _group3_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x - self.g3_threshold)

    def dynamics(
        self,
        x_t: torch.Tensor,
        s_prev: torch.Tensor,
        m_prev: torch.Tensor,
        w_s: torch.Tensor,
        kappa_s: torch.Tensor,
        kappa_m: torch.Tensor,
    ):
        s_t = kappa_s * s_prev + w_s * x_t
        branches = s_t[:, self.branch_indices.to(x_t.device)].sum(dim=-1)
        m_dec = kappa_m * m_prev

        idx = 0
        m_dec_groups = []
        for size in self.group_sizes:
            m_dec_groups.append(m_dec[:, idx : idx + size])
            idx += size

        delta_g1 = torch.tanh(self.g1_proj(branches))
        delta_g2 = self.g2_thresh(branches) * 2.0 - 1.0

        g2_activity = m_dec_groups[1] + delta_g2
        delta_g3 = self._group3_activation(self.g3_proj(g2_activity))

        apical_branches = branches[:, self.apical_start :]
        g1_activity = m_dec_groups[0] + delta_g1
        g4_input = torch.cat([apical_branches, g1_activity], dim=-1)
        delta_g4 = self.g4_thresh(g4_input) * 2.0 - 1.0

        delta_g5 = torch.tanh(self.g5_proj(g1_activity))

        delta_m = torch.cat(
            [delta_g1, delta_g2, delta_g3, delta_g4, delta_g5], dim=-1
        )
        m_t = m_dec + self.lambda_value * (1.0 - kappa_m) * delta_m
        y_t = self.w_y(m_t)
        return y_t, s_t, m_t

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, _ = X.shape
        inputs = self.route_input_to_synapses(X)
        kappa_s, kappa_m = self._get_kappas()
        w_s = torch.nn.functional.softplus(self.w_s_raw)
        s_prev = torch.zeros(batch_size, self.num_synapse, device=X.device)
        m_prev = torch.zeros(batch_size, self.dm, device=X.device)
        outputs: List[torch.Tensor] = []
        for t in range(num_steps):
            y_t, s_prev, m_prev = self.dynamics(
                inputs[:, t], s_prev, m_prev, w_s, kappa_s, kappa_m
            )
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def neuronio_eval_forward(
        self, X: torch.Tensor, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ) -> torch.Tensor:
        outputs = self.forward(X)
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]
        spike_pred = torch.sigmoid(spike_pred)
        soma_pred = soma_pred / y_train_soma_scale
        return torch.stack([spike_pred, soma_pred], dim=-1)

    def neuronio_viz_forward(
        self, X: torch.Tensor, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ):
        batch_size, num_steps, _ = X.shape
        inputs = self.route_input_to_synapses(X)
        kappa_s, kappa_m = self._get_kappas()
        w_s = torch.nn.functional.softplus(self.w_s_raw)
        s_prev = torch.zeros(batch_size, self.num_synapse, device=X.device)
        m_prev = torch.zeros(batch_size, self.dm, device=X.device)

        outputs: List[torch.Tensor] = []
        s_record: List[torch.Tensor] = []
        m_record: List[torch.Tensor] = []
        for t in range(num_steps):
            y_t, s_prev, m_prev = self.dynamics(
                inputs[:, t], s_prev, m_prev, w_s, kappa_s, kappa_m
            )
            outputs.append(y_t)
            s_record.append(s_prev)
            m_record.append(m_prev)

        outputs = torch.stack(outputs, dim=1)
        s_record = torch.stack(s_record, dim=1)
        m_record = torch.stack(m_record, dim=1)

        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]
        spike_pred = torch.sigmoid(spike_pred)
        soma_pred = soma_pred / y_train_soma_scale
        outputs = torch.stack([spike_pred, soma_pred], dim=-1)
        return outputs, s_record, m_record

    def init_state(self, batch_size: int, device):
        return (
            torch.zeros(batch_size, self.num_synapse, device=device),
            torch.zeros(batch_size, self.dm, device=device),
        )

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        print("\n" + "=" * 60)
        print("GIADA_L4P1 - Architecture Summary")
        print("=" * 60)
        print(
            f"\nInput: {self.num_synapse} synapses -> {self.num_branch} branches (fixed sum)"
        )
        print(
            f"Apical branches: {self.apical_start} to {self.num_branch - 1} "
            f"({self.num_branch - self.apical_start} branches)"
        )

        groups = [
            ("Group 1 - Fast membrane", self.g1, self.tau_bounds[0], "Linear", "All branches"),
            ("Group 2 - NMDA dendritic", self.g2, self.tau_bounds[1], "Threshold", "All branches"),
            ("Group 3 - Calcium", self.g3, self.tau_bounds[2], "Linear+ReLU", "Group 2"),
            ("Group 4 - Apical spike", self.g4, self.tau_bounds[3], "Threshold", "Apical branches + Group 1"),
            ("Group 5 - Ultra-slow", self.g5, self.tau_bounds[4], "Linear", "Group 1"),
        ]
        print(
            f"\n{'Group':<28} {'Units':<7} {'tau range (ms)':<15} "
            f"{'Nonlinearity':<14} {'Input from'}"
        )
        print("-" * 90)
        for name, size, (lo, hi), nonlin, source in groups:
            print(
                f"{name:<28} {size:<7} [{lo:.0f}, {hi:.0f}]{'':<8} "
                f"{nonlin:<14} {source}"
            )
        print(f"\nOutput: all {self.dm} units -> Linear -> {self.num_output}")
        print(f"\nTotal trainable params: {self.count_params():,}")
        print("=" * 60)


# Backward-compatible alias for existing notebooks and scripts.
ELM = GIADA_L4P1
