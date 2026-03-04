"""Reusable RNO control-conditioning modules for heat notebooks.

This module centralizes shared components used by:
- ``train_RNO_broadcast.ipynb``
- ``train_RNO_sin_embed.ipynb``
- ``train_RNO_gaussian_actuators.ipynb``

Classes
-------
ControlLifterSin
    Lifts a low-dimensional control vector to a length-``embed_dim`` representation
    using sinusoidal features.
RNOControlARBroadcast
    Autoregressive wrapper where controls are broadcast as extra channels.
RNOControlARSinEmbed
    Autoregressive wrapper where controls are lifted with ``ControlLifterSin`` and
    concatenated as a second channel.
RNOControlARGaussianActuators
    Autoregressive wrapper where controls are lifted into spatial forcing fields
    using Gaussian actuator footprints, then concatenated with state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from neuralop.layers.embeddings import SinusoidalEmbedding


class ControlLifterSin:
    """Lift controls from shape ``(..., control_dim)`` to ``(..., embed_dim)``.

    The lifting is deterministic and has no trainable parameters. It applies
    ``SinusoidalEmbedding`` to the control vector at each spatial-temporal location,
    then truncates to ``embed_dim``.

    Even though this class is not an ``nn.Module``, it exposes ``to(...)`` and
    forwards that call to the internal ``SinusoidalEmbedding`` module so notebooks
    can set device/dtype explicitly.
    """

    def __init__(
        self,
        control_dim: int = 4,
        embed_dim: int = 100,
        num_frequencies: int | None = None,
        embedding_type: str = "transformer",
        max_positions: int = 10000,
    ):
        self.control_dim = int(control_dim)
        self.embed_dim = int(embed_dim)

        # Default: minimal L such that 2 * L * control_dim >= embed_dim.
        if num_frequencies is None:
            num_frequencies = int(math.ceil(self.embed_dim / (2 * self.control_dim)))

        nyquist_limit = self.embed_dim / 2.0
        assert num_frequencies < nyquist_limit, (
            "Nyquist-Shannon criterion violated: "
            f"num_frequencies={num_frequencies} must be < embed_dim/2={nyquist_limit}."
        )

        self.embedding = SinusoidalEmbedding(
            in_channels=self.control_dim,
            num_frequencies=num_frequencies,
            embedding_type=embedding_type,
            max_positions=max_positions,
        )
        self.embedding_dim = int(self.embedding.out_channels)
        if self.embedding_dim < self.embed_dim:
            raise ValueError(
                f"Embedding out_channels={self.embedding_dim} < embed_dim={self.embed_dim}"
            )

    def __call__(self, x_ctrl: torch.Tensor) -> torch.Tensor:
        """Lift controls.

        Parameters
        ----------
        x_ctrl:
            Tensor of shape ``(B, T_hist, 1, T_inner, control_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, T_hist, 1, T_inner, embed_dim)``.
        """
        if x_ctrl.ndim != 5:
            raise ValueError(f"Expected x_ctrl rank-5, got shape {tuple(x_ctrl.shape)}")

        batch_size, timesteps, channels, inner_t, control_dim_local = x_ctrl.shape
        if control_dim_local != self.control_dim:
            raise ValueError(
                f"Expected control_dim={self.control_dim}, got x_ctrl.shape[-1]={control_dim_local}"
            )

        x_flat = x_ctrl.reshape(
            batch_size * timesteps * channels * inner_t, control_dim_local
        )  # (B*T_hist*1*T_inner, control_dim)
        x_emb = self.embedding(x_flat)  # (B*T_hist*1*T_inner, embedding_dim)
        x_emb = x_emb[..., : self.embed_dim]  # (B*T_hist*1*T_inner, embed_dim)
        x_lift = x_emb.reshape(
            batch_size, timesteps, channels, inner_t, self.embed_dim
        )  # (B, T_hist, 1, T_inner, embed_dim)
        return x_lift

    def to(self, *args, **kwargs):
        """Move internal embedding module to a target device/dtype.

        Returns
        -------
        ControlLifterSin
            ``self`` for chaining (same style as ``nn.Module.to``).
        """
        self.embedding = self.embedding.to(*args, **kwargs)
        return self


class RNOControlARBroadcast(nn.Module):
    """Autoregressive RNO wrapper for broadcast control conditioning.

    Controls are treated as exogenous inputs and are spatially broadcast to match
    the state grid before channel-wise concatenation.
    """

    def __init__(self, rno: nn.Module):
        super().__init__()
        self.rno = rno

    @staticmethod
    def lift_controls_broadcast(
        x_ctrl: torch.Tensor, target_nx: int = 100
    ) -> torch.Tensor:
        """Broadcast controls over spatial grid.

        Input: ``(B, T_hist, 1, T_inner, n_ctrl)``
        Output: ``(B, T_hist, n_ctrl, T_inner, target_nx)``
        """
        if x_ctrl.ndim != 5:
            raise ValueError(f"Expected x_ctrl rank-5, got shape {tuple(x_ctrl.shape)}")
        if x_ctrl.shape[2] != 1:
            raise ValueError(f"Expected x_ctrl channel dim=1, got {x_ctrl.shape[2]}")

        ctrl = x_ctrl.squeeze(2)  # (B, T_hist, T_inner, n_ctrl)
        ctrl = ctrl.permute(0, 1, 3, 2)  # (B, T_hist, n_ctrl, T_inner)
        ctrl = ctrl.unsqueeze(-1)  # (B, T_hist, n_ctrl, T_inner, 1)
        ctrl = ctrl.expand(
            -1, -1, -1, -1, target_nx
        )  # (B, T_hist, n_ctrl, T_inner, target_nx)
        return ctrl

    @classmethod
    def build_rno_input(
        cls, x_state: torch.Tensor, x_ctrl: torch.Tensor
    ) -> torch.Tensor:
        """Build RNO input for broadcast strategy.

        ``x_state``: ``(B, T_hist, 1, T_inner, nx)``
        ``x_ctrl``: ``(B, T_hist, 1, T_inner, n_ctrl)``
        Returns ``(B, T_hist, 1+n_ctrl, T_inner, nx)``.
        """
        if x_state.ndim != 5 or x_ctrl.ndim != 5:
            raise ValueError(
                f"Expected rank-5 tensors, got x_state={tuple(x_state.shape)}, x_ctrl={tuple(x_ctrl.shape)}"
            )
        if x_state.shape[:2] != x_ctrl.shape[:2] or x_state.shape[3] != x_ctrl.shape[3]:
            raise ValueError(
                "State/control batch/time/inner-time dims must match: "
                f"x_state={tuple(x_state.shape)}, x_ctrl={tuple(x_ctrl.shape)}"
            )

        x_ctrl_lifted = cls.lift_controls_broadcast(
            x_ctrl, target_nx=x_state.shape[-1]
        )  # (B, T_hist, n_ctrl, T_inner, nx)
        return torch.cat(
            [x_state, x_ctrl_lifted], dim=2
        )  # (B, T_hist, 1+n_ctrl, T_inner, nx)

    def forward_one(
        self,
        u_t: torch.Tensor,
        ctrl_t: torch.Tensor,
        hidden_states=None,
        return_hidden_states: bool = False,
    ):
        """Predict one step ``u_{t+1}`` from ``u_t`` and exogenous ``ctrl_t``."""
        if u_t.ndim == 4:
            u_t = u_t.unsqueeze(1)  # (B, 1, 1, T_inner, nx)
        if ctrl_t.ndim == 4:
            ctrl_t = ctrl_t.unsqueeze(1)  # (B, 1, 1, T_inner, n_ctrl)

        if u_t.ndim != 5 or u_t.shape[1] != 1:
            raise ValueError(f"u_t must be (B,1,1,T_inner,nx), got {tuple(u_t.shape)}")
        if ctrl_t.ndim != 5 or ctrl_t.shape[1] != 1:
            raise ValueError(
                f"ctrl_t must be (B,1,1,T_inner,n_ctrl), got {tuple(ctrl_t.shape)}"
            )

        x_in = self.build_rno_input(u_t, ctrl_t)
        u_next, hidden_states = self.rno(
            x_in,
            init_hidden_states=hidden_states,
            return_hidden_states=True,
            keep_states_padded=True,
        )

        if return_hidden_states:
            return u_next, hidden_states
        return u_next

    def rollout(
        self,
        u0: torch.Tensor,
        ctrl_seq: torch.Tensor,
        steps: int,
        teacher_forcing_states: torch.Tensor | None = None,
        use_teacher_forcing: bool = False,
    ) -> torch.Tensor:
        """Roll out trajectory using exogenous controls aligned per step."""
        if u0.ndim == 5:
            if u0.shape[1] != 1:
                raise ValueError(
                    f"u0 with rank-5 must have time dim=1, got {tuple(u0.shape)}"
                )
            current_state = u0[:, 0]
        elif u0.ndim == 4:
            current_state = u0
        else:
            raise ValueError(f"u0 must be rank 4 or 5, got {tuple(u0.shape)}")

        if ctrl_seq.ndim != 5:
            raise ValueError(f"ctrl_seq must be rank-5, got {tuple(ctrl_seq.shape)}")
        if ctrl_seq.shape[1] < steps:
            raise ValueError(
                f"ctrl_seq has {ctrl_seq.shape[1]} steps but rollout requested {steps}"
            )

        if use_teacher_forcing:
            if teacher_forcing_states is None:
                raise ValueError(
                    "use_teacher_forcing=True requires teacher_forcing_states"
                )
            if teacher_forcing_states.shape[1] < steps:
                raise ValueError(
                    "teacher_forcing_states must have at least 'steps' timesteps "
                    f"(got {teacher_forcing_states.shape[1]} < {steps})"
                )

        traj = [current_state]
        hidden_states = None

        for k in range(steps):
            if use_teacher_forcing and k > 0:
                current_state = teacher_forcing_states[:, k]

            ctrl_t = ctrl_seq[:, k]
            u_next, hidden_states = self.forward_one(
                current_state,
                ctrl_t,
                hidden_states=hidden_states,
                return_hidden_states=True,
            )
            traj.append(u_next)
            current_state = u_next

        return torch.stack(traj, dim=1)


class RNOControlARSinEmbed(nn.Module):
    """Autoregressive RNO wrapper for sinusoidal-embedded control conditioning."""

    def __init__(self, rno: nn.Module, control_lifter):
        super().__init__()
        self.rno = rno
        self.control_lifter = control_lifter

    @staticmethod
    def build_rno_input(
        x_state: torch.Tensor,
        x_ctrl: torch.Tensor,
        control_lifter,
    ) -> torch.Tensor:
        """Build RNO input for sin-embed strategy.

        ``x_state``: ``(B, T_hist, 1, T_inner, nx)``
        ``x_ctrl``: ``(B, T_hist, 1, T_inner, control_dim)``
        Returns ``(B, T_hist, 2, T_inner, nx)``.
        """
        if control_lifter is None:
            raise ValueError("Sin-embed strategy requires a control_lifter callable")
        if x_state.ndim != 5 or x_ctrl.ndim != 5:
            raise ValueError(
                f"Expected rank-5 tensors, got x_state={tuple(x_state.shape)}, x_ctrl={tuple(x_ctrl.shape)}"
            )
        if x_state.shape[:2] != x_ctrl.shape[:2] or x_state.shape[3] != x_ctrl.shape[3]:
            raise ValueError(
                "State/control batch/time/inner-time dims must match: "
                f"x_state={tuple(x_state.shape)}, x_ctrl={tuple(x_ctrl.shape)}"
            )

        x_ctrl_lifted = control_lifter(x_ctrl)
        return torch.cat([x_state, x_ctrl_lifted], dim=2)  # (B, T_hist, 2, T_inner, nx)

    def forward_one(
        self,
        u_t: torch.Tensor,
        ctrl_t: torch.Tensor,
        hidden_states=None,
        return_hidden_states: bool = False,
    ):
        """Predict one step ``u_{t+1}`` from ``u_t`` and exogenous ``ctrl_t``."""
        if u_t.ndim == 4:
            u_t = u_t.unsqueeze(1)
        if ctrl_t.ndim == 4:
            ctrl_t = ctrl_t.unsqueeze(1)

        if u_t.ndim != 5 or u_t.shape[1] != 1:
            raise ValueError(f"u_t must be (B,1,1,T_inner,nx), got {tuple(u_t.shape)}")
        if ctrl_t.ndim != 5 or ctrl_t.shape[1] != 1:
            raise ValueError(
                f"ctrl_t must be (B,1,1,T_inner,control_dim), got {tuple(ctrl_t.shape)}"
            )

        x_in = self.build_rno_input(u_t, ctrl_t, self.control_lifter)
        u_next, hidden_states = self.rno(
            x_in,
            init_hidden_states=hidden_states,
            return_hidden_states=True,
            keep_states_padded=True,
        )

        if return_hidden_states:
            return u_next, hidden_states
        return u_next

    def rollout(
        self,
        u0: torch.Tensor,
        ctrl_seq: torch.Tensor,
        steps: int,
        teacher_forcing_states: torch.Tensor | None = None,
        use_teacher_forcing: bool = False,
    ) -> torch.Tensor:
        """Roll out trajectory using exogenous controls aligned per step."""
        if u0.ndim == 5:
            if u0.shape[1] != 1:
                raise ValueError(
                    f"u0 with rank-5 must have time dim=1, got {tuple(u0.shape)}"
                )
            current_state = u0[:, 0]
        elif u0.ndim == 4:
            current_state = u0
        else:
            raise ValueError(f"u0 must be rank 4 or 5, got {tuple(u0.shape)}")

        if ctrl_seq.ndim != 5:
            raise ValueError(f"ctrl_seq must be rank-5, got {tuple(ctrl_seq.shape)}")
        if ctrl_seq.shape[1] < steps:
            raise ValueError(
                f"ctrl_seq has {ctrl_seq.shape[1]} steps but rollout requested {steps}"
            )

        if use_teacher_forcing:
            if teacher_forcing_states is None:
                raise ValueError(
                    "use_teacher_forcing=True requires teacher_forcing_states"
                )
            if teacher_forcing_states.shape[1] < steps:
                raise ValueError(
                    "teacher_forcing_states must have at least 'steps' timesteps "
                    f"(got {teacher_forcing_states.shape[1]} < {steps})"
                )

        traj = [current_state]
        hidden_states = None

        for k in range(steps):
            if use_teacher_forcing and k > 0:
                current_state = teacher_forcing_states[:, k]

            ctrl_t = ctrl_seq[:, k]
            u_next, hidden_states = self.forward_one(
                current_state,
                ctrl_t,
                hidden_states=hidden_states,
                return_hidden_states=True,
            )
            traj.append(u_next)
            current_state = u_next

        return torch.stack(traj, dim=1)


class RNOControlARGaussianActuators(nn.Module):
    """Autoregressive RNO wrapper using Gaussian actuator control lifting.

    Controls are interpreted as actuator amplitudes and lifted to a forcing field:

    ``f(x, t) = sum_j u_j(t) * exp(- (x - mu_j)^2 / (2*sigma^2))``

    The lifted forcing channel is concatenated with state, giving RNO inputs of shape
    ``(B, T_hist, 2, T_inner, nx)``.
    """

    def __init__(
        self,
        rno: nn.Module,
        mus=(0.2, 0.4, 0.6, 0.8),
        sigma: float = 0.1,
        x_min: float = 0.0,
        x_max: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.rno = rno
        self.mus = tuple(float(m) for m in mus)
        self.sigma = float(sigma)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.normalize = bool(normalize)
        self._A_cache = {}

    def _cache_key(self, device, dtype, nx: int, nf: int):
        return (
            device.type,
            device.index,
            str(dtype),
            int(nx),
            int(nf),
            self.mus,
            self.sigma,
            self.x_min,
            self.x_max,
            self.normalize,
        )

    def _get_cached_A(self, device, dtype, nx: int, nf: int) -> torch.Tensor:
        key = self._cache_key(device=device, dtype=dtype, nx=nx, nf=nf)
        if key not in self._A_cache:
            x = torch.linspace(
                self.x_min, self.x_max, nx, device=device, dtype=dtype
            ).view(nx, 1)  # (nx, 1)
            mu = torch.tensor(self.mus, device=device, dtype=dtype).view(
                1, nf
            )  # (1, nf)
            A = torch.exp(-0.5 * ((x - mu) / self.sigma) ** 2)  # (nx, nf)
            if self.normalize:
                A = A / (A.sum(dim=0, keepdim=True) + 1e-12)
            self._A_cache[key] = A
        return self._A_cache[key]

    def build_rno_input(
        self, x_state: torch.Tensor, x_ctrl: torch.Tensor
    ) -> torch.Tensor:
        """Build RNO input for Gaussian actuator strategy.

        ``x_state``: ``(B, T_hist, 1, T_inner, nx)``
        ``x_ctrl``: ``(B, T_hist, 1, T_inner, nf)``
        Returns: ``(B, T_hist, 2, T_inner, nx)``
        """
        if x_state.ndim != 5 or x_ctrl.ndim != 5:
            raise ValueError(
                f"Expected rank-5 tensors, got x_state={tuple(x_state.shape)}, x_ctrl={tuple(x_ctrl.shape)}"
            )
        if x_state.shape[:2] != x_ctrl.shape[:2] or x_state.shape[3] != x_ctrl.shape[3]:
            raise ValueError(
                "State/control batch/time/inner-time dims must match: "
                f"x_state={tuple(x_state.shape)}, x_ctrl={tuple(x_ctrl.shape)}"
            )
        if x_ctrl.shape[2] != 1:
            raise ValueError(f"Expected x_ctrl channel dim=1, got {x_ctrl.shape[2]}")

        controls_lastdim = x_ctrl.squeeze(2)  # (B, T_hist, T_inner, nf)
        nf = controls_lastdim.shape[-1]
        if nf != len(self.mus):
            raise ValueError(
                f"Expected control dim nf={len(self.mus)} from mus, got nf={nf}"
            )

        nx = x_state.shape[-1]
        A = self._get_cached_A(
            device=x_state.device, dtype=x_state.dtype, nx=nx, nf=nf
        )  # (nx, nf)
        forcing = torch.einsum(
            "...f,xf->...x", controls_lastdim, A
        )  # (B, T_hist, T_inner, nx)
        forcing = forcing.unsqueeze(2)  # (B, T_hist, 1, T_inner, nx)

        return torch.cat([x_state, forcing], dim=2)  # (B, T_hist, 2, T_inner, nx)

    def forward_one(
        self,
        u_t: torch.Tensor,
        ctrl_t: torch.Tensor,
        hidden_states=None,
        return_hidden_states: bool = False,
    ):
        """Predict one step ``u_{t+1}`` from ``u_t`` and exogenous ``ctrl_t``."""
        if u_t.ndim == 4:
            u_t = u_t.unsqueeze(1)  # (B, 1, 1, T_inner, nx)
        if ctrl_t.ndim == 4:
            ctrl_t = ctrl_t.unsqueeze(1)  # (B, 1, 1, T_inner, nf)

        if u_t.ndim != 5 or u_t.shape[1] != 1:
            raise ValueError(f"u_t must be (B,1,1,T_inner,nx), got {tuple(u_t.shape)}")
        if ctrl_t.ndim != 5 or ctrl_t.shape[1] != 1:
            raise ValueError(
                f"ctrl_t must be (B,1,1,T_inner,nf), got {tuple(ctrl_t.shape)}"
            )

        x_in = self.build_rno_input(u_t, ctrl_t)
        u_next, hidden_states = self.rno(
            x_in,
            init_hidden_states=hidden_states,
            return_hidden_states=True,
            keep_states_padded=True,
        )

        if return_hidden_states:
            return u_next, hidden_states
        return u_next

    def rollout(
        self,
        u0: torch.Tensor,
        ctrl_seq: torch.Tensor,
        steps: int,
        teacher_forcing_states: torch.Tensor | None = None,
        use_teacher_forcing: bool = False,
    ) -> torch.Tensor:
        """Roll out trajectory with exogenous controls aligned as ``ctrl_seq[:, k]``."""
        if u0.ndim == 5:
            if u0.shape[1] != 1:
                raise ValueError(
                    f"u0 with rank-5 must have time dim=1, got {tuple(u0.shape)}"
                )
            current_state = u0[:, 0]
        elif u0.ndim == 4:
            current_state = u0
        else:
            raise ValueError(f"u0 must be rank 4 or 5, got {tuple(u0.shape)}")

        if ctrl_seq.ndim != 5:
            raise ValueError(f"ctrl_seq must be rank-5, got {tuple(ctrl_seq.shape)}")
        if ctrl_seq.shape[1] < steps:
            raise ValueError(
                f"ctrl_seq has {ctrl_seq.shape[1]} steps but rollout requested {steps}"
            )

        if use_teacher_forcing:
            if teacher_forcing_states is None:
                raise ValueError(
                    "use_teacher_forcing=True requires teacher_forcing_states"
                )
            if teacher_forcing_states.shape[1] < steps:
                raise ValueError(
                    "teacher_forcing_states must have at least 'steps' timesteps "
                    f"(got {teacher_forcing_states.shape[1]} < {steps})"
                )

        traj = [current_state]
        hidden_states = None

        for k in range(steps):
            if use_teacher_forcing and k > 0:
                current_state = teacher_forcing_states[:, k]

            ctrl_t = ctrl_seq[:, k]  # controls are exogenous and aligned with u_k
            u_next, hidden_states = self.forward_one(
                current_state,
                ctrl_t,
                hidden_states=hidden_states,
                return_hidden_states=True,
            )
            traj.append(u_next)
            current_state = u_next

        return torch.stack(traj, dim=1)


def rollout_variant_a_carry(
    ar_model_obj,
    u0: torch.Tensor,
    ctrl_seq: torch.Tensor,
    steps: int,
):
    """Variant A rollout: carry hidden state across all rollout steps.

    Parameters
    ----------
    ar_model_obj:
        Any wrapper exposing ``forward_one(u_t, ctrl_t, hidden_states, return_hidden_states)``.
    u0:
        ``(B, 1, T_inner, nx)`` initial state.
    ctrl_seq:
        ``(B, steps, 1, T_inner, n_ctrl)`` exogenous controls aligned with ``u_k``.
    steps:
        Number of autoregressive rollout steps.

    Returns
    -------
    tuple[torch.Tensor, Any]
        ``(traj, hidden_states_final)`` where ``traj`` has shape
        ``(B, steps+1, 1, T_inner, nx)``.
    """
    if ctrl_seq.ndim != 5:
        raise ValueError(f"ctrl_seq must be rank-5, got {tuple(ctrl_seq.shape)}")
    if ctrl_seq.shape[1] < steps:
        raise ValueError(
            f"ctrl_seq has {ctrl_seq.shape[1]} steps but rollout requested {steps}"
        )

    current_state = u0
    traj = [current_state]
    hidden_states = None

    for k in range(steps):
        ctrl_t = ctrl_seq[:, k]  # (B, 1, T_inner, n_ctrl)
        u_next, hidden_states = ar_model_obj.forward_one(
            current_state,
            ctrl_t,
            hidden_states=hidden_states,
            return_hidden_states=True,
        )
        traj.append(u_next)
        current_state = u_next

    return torch.stack(traj, dim=1), hidden_states


def rollout_variant_b_reset_each_step(
    ar_model_obj,
    u0: torch.Tensor,
    ctrl_seq: torch.Tensor,
    steps: int,
):
    """Variant B rollout: reset hidden state to ``None`` at each step."""
    if ctrl_seq.ndim != 5:
        raise ValueError(f"ctrl_seq must be rank-5, got {tuple(ctrl_seq.shape)}")
    if ctrl_seq.shape[1] < steps:
        raise ValueError(
            f"ctrl_seq has {ctrl_seq.shape[1]} steps but rollout requested {steps}"
        )

    current_state = u0
    traj = [current_state]

    for k in range(steps):
        ctrl_t = ctrl_seq[:, k]  # (B, 1, T_inner, n_ctrl)
        u_next = ar_model_obj.forward_one(
            current_state,
            ctrl_t,
            hidden_states=None,
            return_hidden_states=False,
        )
        traj.append(u_next)
        current_state = u_next

    return torch.stack(traj, dim=1)


def rollout_variant_c_warmup_then_carry(
    ar_model_obj,
    x_state_hist: torch.Tensor,
    x_ctrl_hist: torch.Tensor,
    u0: torch.Tensor,
    ctrl_seq: torch.Tensor,
    steps: int,
    history: int | None = None,
    warmup_steps: int | None = None,
):
    """Variant C rollout: warm-up on history, then carry hidden state.

    Warm-up uses teacher forcing over history windows and excludes ``u0`` by default.
    """
    if x_state_hist.ndim != 5:
        raise ValueError(
            f"x_state_hist must be rank-5, got {tuple(x_state_hist.shape)}"
        )
    if x_ctrl_hist.ndim != 5:
        raise ValueError(f"x_ctrl_hist must be rank-5, got {tuple(x_ctrl_hist.shape)}")
    if ctrl_seq.ndim != 5:
        raise ValueError(f"ctrl_seq must be rank-5, got {tuple(ctrl_seq.shape)}")
    if x_state_hist.shape[:2] != x_ctrl_hist.shape[:2]:
        raise ValueError(
            "x_state_hist and x_ctrl_hist batch/time dims must match: "
            f"{tuple(x_state_hist.shape)} vs {tuple(x_ctrl_hist.shape)}"
        )
    if ctrl_seq.shape[1] < steps:
        raise ValueError(
            f"ctrl_seq has {ctrl_seq.shape[1]} steps but rollout requested {steps}"
        )

    t_hist = x_state_hist.shape[1]
    history_eff = t_hist if history is None else int(history)
    if history_eff < 1 or history_eff > t_hist:
        raise ValueError(f"history must be in [1, {t_hist}], got {history_eff}")

    max_warmup = history_eff - 1
    warmup_steps_eff = max_warmup if warmup_steps is None else int(warmup_steps)
    if warmup_steps_eff < 0 or warmup_steps_eff > max_warmup:
        raise ValueError(
            f"warmup_steps must be in [0, {max_warmup}] for history={history_eff}, got {warmup_steps_eff}"
        )

    history_start = t_hist - history_eff
    hidden_states = None

    for t in range(history_start, history_start + warmup_steps_eff):
        u_t_hist = x_state_hist[:, t]  # (B, 1, T_inner, nx)
        ctrl_t_hist = x_ctrl_hist[:, t]  # (B, 1, T_inner, n_ctrl)
        _, hidden_states = ar_model_obj.forward_one(
            u_t_hist,
            ctrl_t_hist,
            hidden_states=hidden_states,
            return_hidden_states=True,
        )

    current_state = u0
    traj = [current_state]
    for k in range(steps):
        ctrl_t = ctrl_seq[:, k]  # (B, 1, T_inner, n_ctrl)
        u_next, hidden_states = ar_model_obj.forward_one(
            current_state,
            ctrl_t,
            hidden_states=hidden_states,
            return_hidden_states=True,
        )
        traj.append(u_next)
        current_state = u_next

    return torch.stack(traj, dim=1), warmup_steps_eff, hidden_states


def evaluate_endpoint_metrics_variants(
    ar_model_obj,
    eval_loaders_by_h,
    eval_steps,
    device,
    h_train: int | None = None,
    history: int | None = None,
    warmup_steps: int | None = None,
    variants=("A", "B", "C"),
    dt: float | None = None,
    timesteps_per_control_action: float = 1.0,
    show_window_count: bool = False,
):
    """Evaluate endpoint Relative L2 for rollout variants A/B/C by horizon.

    Each loader batch is expected to yield:
    ``(x_state_hist, x_ctrl_hist, x_ctrl_seq, y_endpoint)`` with shapes
    ``(B, T_hist, 1, T_inner, nx)``, ``(B, T_hist, 1, T_inner, n_ctrl)``,
    ``(B, h, 1, T_inner, n_ctrl)``, ``(B, 1, T_inner, nx)``.

    Notes
    -----
    - ``h`` (from ``eval_steps``) is rollout horizon in chunk-steps.
    - ``raw_steps = h * T_inner`` is the number of raw control actions consumed per
      window under the current chunking.
    - ``traversed_steps = raw_steps * timesteps_per_control_action`` converts control
      actions to effective traversed time steps (default 1:1).
    - If ``dt`` is provided, ``physical_time = traversed_steps * dt`` is reported.
    """
    eval_steps = [int(h) for h in eval_steps]
    variants = tuple(str(v).upper() for v in variants)
    allowed = {"A", "B", "C"}
    if any(v not in allowed for v in variants):
        raise ValueError(f"variants must be subset of {allowed}, got {variants}")
    if len(eval_steps) == 0:
        raise ValueError("eval_steps must be non-empty")

    if h_train is not None and max(eval_steps) > int(h_train):
        print(
            f"Warning: Evaluating beyond trained horizon (H_train={h_train}). Errors may be unstable and not comparable."
        )

    was_training = (
        bool(ar_model_obj.training) if hasattr(ar_model_obj, "training") else False
    )
    if hasattr(ar_model_obj, "eval"):
        ar_model_obj.eval()

    results = {}
    printed_shapes = False

    with torch.no_grad():
        for h in eval_steps:
            if h not in eval_loaders_by_h:
                raise ValueError(f"No evaluation loader for horizon h={h}")

            loader = eval_loaders_by_h[h]
            per_variant_sum = {v: 0.0 for v in variants}
            n_total = 0
            t_inner_h = None

            for x_state_hist, x_ctrl_hist, x_ctrl_seq, y_endpoint in loader:
                x_state_hist = x_state_hist.to(
                    device
                ).float()  # (B, T_hist, 1, T_inner, nx)
                x_ctrl_hist = x_ctrl_hist.to(
                    device
                ).float()  # (B, T_hist, 1, T_inner, n_ctrl)
                x_ctrl_seq = x_ctrl_seq.to(device).float()  # (B, h, 1, T_inner, n_ctrl)
                y_endpoint = y_endpoint.to(device).float()  # (B, 1, T_inner, nx)

                if x_ctrl_seq.shape[1] < h:
                    raise ValueError(
                        f"Requested h={h}, but ctrl_seq has only {x_ctrl_seq.shape[1]} control steps"
                    )

                if t_inner_h is None:
                    t_inner_h = int(x_state_hist.shape[3])
                if int(x_ctrl_seq.shape[3]) != int(t_inner_h):
                    raise ValueError(
                        "State/control inner-time dims must match for rollout accounting: "
                        f"x_state_hist.shape[3]={x_state_hist.shape[3]} vs "
                        f"x_ctrl_seq.shape[3]={x_ctrl_seq.shape[3]}"
                    )

                u0 = x_state_hist[:, -1]  # (B, 1, T_inner, nx)
                ctrl_seq_h = x_ctrl_seq[:, :h]  # (B, h, 1, T_inner, n_ctrl)

                if "A" in variants:
                    pred_roll_a, _ = rollout_variant_a_carry(
                        ar_model_obj=ar_model_obj,
                        u0=u0,
                        ctrl_seq=ctrl_seq_h,
                        steps=h,
                    )
                    y_pred_a = pred_roll_a[:, -1]  # (B, 1, T_inner, nx)
                    diff_a = (y_pred_a - y_endpoint).reshape(y_endpoint.shape[0], -1)
                    y_flat = y_endpoint.reshape(y_endpoint.shape[0], -1)
                    rel_a = torch.linalg.norm(diff_a, dim=1) / (
                        torch.linalg.norm(y_flat, dim=1) + 1e-12
                    )
                    per_variant_sum["A"] += rel_a.sum().item()

                if "B" in variants:
                    pred_roll_b = rollout_variant_b_reset_each_step(
                        ar_model_obj=ar_model_obj,
                        u0=u0,
                        ctrl_seq=ctrl_seq_h,
                        steps=h,
                    )
                    y_pred_b = pred_roll_b[:, -1]  # (B, 1, T_inner, nx)
                    diff_b = (y_pred_b - y_endpoint).reshape(y_endpoint.shape[0], -1)
                    y_flat = y_endpoint.reshape(y_endpoint.shape[0], -1)
                    rel_b = torch.linalg.norm(diff_b, dim=1) / (
                        torch.linalg.norm(y_flat, dim=1) + 1e-12
                    )
                    per_variant_sum["B"] += rel_b.sum().item()

                if "C" in variants:
                    pred_roll_c, warmup_steps_eff, _ = (
                        rollout_variant_c_warmup_then_carry(
                            ar_model_obj=ar_model_obj,
                            x_state_hist=x_state_hist,
                            x_ctrl_hist=x_ctrl_hist,
                            u0=u0,
                            ctrl_seq=ctrl_seq_h,
                            steps=h,
                            history=history,
                            warmup_steps=warmup_steps,
                        )
                    )
                    y_pred_c = pred_roll_c[:, -1]  # (B, 1, T_inner, nx)
                    diff_c = (y_pred_c - y_endpoint).reshape(y_endpoint.shape[0], -1)
                    y_flat = y_endpoint.reshape(y_endpoint.shape[0], -1)
                    rel_c = torch.linalg.norm(diff_c, dim=1) / (
                        torch.linalg.norm(y_flat, dim=1) + 1e-12
                    )
                    per_variant_sum["C"] += rel_c.sum().item()

                if not printed_shapes:
                    print("x_state_hist:", tuple(x_state_hist.shape))
                    print("x_ctrl_hist:", tuple(x_ctrl_hist.shape))
                    print("ctrl_seq used:", tuple(ctrl_seq_h.shape))
                    print("y_true endpoint:", tuple(y_endpoint.shape))
                    if "A" in variants:
                        print("y_pred A endpoint:", tuple(y_pred_a.shape))
                    if "B" in variants:
                        print("y_pred B endpoint:", tuple(y_pred_b.shape))
                    if "C" in variants:
                        print("y_pred C endpoint:", tuple(y_pred_c.shape))
                        print("warmup_steps used (C):", int(warmup_steps_eff))
                    printed_shapes = True

                n_total += x_state_hist.shape[0]

            raw_steps = int(h * (t_inner_h if t_inner_h is not None else 0))
            traversed_steps = float(raw_steps * float(timesteps_per_control_action))
            results[h] = {
                "N": int(n_total),
                "chunk_steps": int(h),
                "chunk_len": int(t_inner_h if t_inner_h is not None else 0),
                "raw_steps": raw_steps,
                "control_actions": raw_steps,
                "traversed_steps": traversed_steps,
            }
            if dt is not None:
                results[h]["physical_time"] = float(traversed_steps * float(dt))
            for v in variants:
                results[h][v] = per_variant_sum[v] / max(n_total, 1)

    if hasattr(ar_model_obj, "train") and was_training:
        ar_model_obj.train()

    if dt is None:
        header = "H(chunks)  ChunkLen  RawSteps  TraversedSteps  CtrlActs"
        if show_window_count:
            header += "  N(windows)"
        header += "  " + "  ".join([f"RelL2-{v}" for v in variants])
    else:
        header = (
            "H(chunks)  ChunkLen  RawSteps  TraversedSteps  PhysicalTime  CtrlActs"
        )
        if show_window_count:
            header += "  N(windows)"
        header += "  " + "  ".join([f"RelL2-{v}" for v in variants])
    print("Endpoint Rel. L2 by horizon and variant")
    print("-" * len(header))
    print(header)
    for h in eval_steps:
        vals = "  ".join([f"{results[h][v]:.6e}" for v in variants])
        if dt is None:
            row = (
                f"{results[h]['chunk_steps']:<9d}  {results[h]['chunk_len']:<8d}  "
                f"{results[h]['raw_steps']:<8d}  {results[h]['traversed_steps']:<14.3f}  "
                f"{results[h]['control_actions']:<8d}"
            )
            if show_window_count:
                row += f"  {results[h]['N']:<10d}"
            row += f"  {vals}"
            print(row)
        else:
            row = (
                f"{results[h]['chunk_steps']:<9d}  {results[h]['chunk_len']:<8d}  "
                f"{results[h]['raw_steps']:<8d}  {results[h]['traversed_steps']:<14.3f}  "
                f"{results[h]['physical_time']:<12.6f}  {results[h]['control_actions']:<8d}"
            )
            if show_window_count:
                row += f"  {results[h]['N']:<10d}"
            row += f"  {vals}"
            print(row)

    return results


def evaluate_single_trajectory_full_rollout(
    ar_model_obj,
    sol_chunks,
    ctrl_chunks,
    traj_idx: int = 0,
    start_chunk: int = 0,
    history: int = 5,
    variant: str = "A",
    warmup_steps: int | None = None,
    device=None,
    dt: float | None = None,
    timesteps_per_control_action: float = 1.0,
):
    """Evaluate full autoregressive rollout on a single trajectory.

    Parameters
    ----------
    ar_model_obj:
        Control-conditioned AR wrapper exposing ``forward_one`` and rollout helpers.
    sol_chunks:
        State chunks with shape ``(n_traj, n_chunks, T_inner, nx, 1)`` or a single
        trajectory ``(n_chunks, T_inner, nx, 1)``.
    ctrl_chunks:
        Control chunks with shape ``(n_traj, n_chunks, T_inner, n_ctrl, 1)`` or a single
        trajectory ``(n_chunks, T_inner, n_ctrl, 1)``.
    traj_idx:
        Trajectory index used when inputs include ``n_traj``.
    start_chunk:
        Chunk index where evaluation window starts.
    history:
        Number of history chunks used for context.
    variant:
        Hidden-state handling variant: ``"A"``, ``"B"``, or ``"C"``.
    warmup_steps:
        Warm-up steps for Variant C. If ``None``, defaults to ``history - 1``.
    device:
        Target torch device. If ``None``, inferred from ``ar_model_obj``.
    dt:
        Physical time step size per traversed raw step.
    timesteps_per_control_action:
        Conversion from control actions to traversed raw timesteps (default 1.0).

    Returns
    -------
    dict
        Contains summary metrics, traversal counts, and rollout tensors.
    """
    variant = str(variant).upper()
    if variant not in {"A", "B", "C"}:
        raise ValueError(f"variant must be one of ('A','B','C'), got {variant}")

    if not torch.is_tensor(sol_chunks):
        sol_chunks = torch.as_tensor(sol_chunks)
    if not torch.is_tensor(ctrl_chunks):
        ctrl_chunks = torch.as_tensor(ctrl_chunks)

    if sol_chunks.ndim == 5:
        traj_sol = sol_chunks[int(traj_idx)]  # (n_chunks, T_inner, nx, 1)
    elif sol_chunks.ndim == 4:
        traj_sol = sol_chunks  # (n_chunks, T_inner, nx, 1)
    else:
        raise ValueError(
            f"sol_chunks must have rank 4 or 5, got shape {tuple(sol_chunks.shape)}"
        )

    if ctrl_chunks.ndim == 5:
        traj_ctrl = ctrl_chunks[int(traj_idx)]  # (n_chunks, T_inner, n_ctrl, 1)
    elif ctrl_chunks.ndim == 4:
        traj_ctrl = ctrl_chunks  # (n_chunks, T_inner, n_ctrl, 1)
    else:
        raise ValueError(
            f"ctrl_chunks must have rank 4 or 5, got shape {tuple(ctrl_chunks.shape)}"
        )

    if traj_sol.ndim != 4 or traj_ctrl.ndim != 4:
        raise ValueError(
            f"Expected rank-4 single-trajectory chunks, got sol={tuple(traj_sol.shape)}, ctrl={tuple(traj_ctrl.shape)}"
        )
    if traj_sol.shape[0] != traj_ctrl.shape[0]:
        raise ValueError(
            f"State/control chunk count mismatch: sol={traj_sol.shape[0]}, ctrl={traj_ctrl.shape[0]}"
        )
    if traj_sol.shape[1] != traj_ctrl.shape[1]:
        raise ValueError(
            f"State/control chunk_len mismatch: sol={traj_sol.shape[1]}, ctrl={traj_ctrl.shape[1]}"
        )

    n_chunks = int(traj_sol.shape[0])
    t_inner = int(traj_sol.shape[1])
    nx = int(traj_sol.shape[2])
    n_ctrl = int(traj_ctrl.shape[2])

    start_chunk = int(start_chunk)
    history = int(history)
    if start_chunk < 0 or start_chunk >= n_chunks:
        raise ValueError(f"start_chunk must be in [0, {n_chunks-1}], got {start_chunk}")
    if history < 1 or (start_chunk + history) >= n_chunks:
        raise ValueError(
            f"history must satisfy 1 <= history < n_chunks-start_chunk ({n_chunks-start_chunk}), got {history}"
        )

    steps = n_chunks - start_chunk - history
    if steps < 1:
        raise ValueError("No forecast steps available for the provided start/history")

    if device is None:
        try:
            device = next(ar_model_obj.parameters()).device
        except Exception:
            device = torch.device("cpu")
    device = torch.device(device)

    was_training = (
        bool(ar_model_obj.training) if hasattr(ar_model_obj, "training") else False
    )
    if hasattr(ar_model_obj, "eval"):
        ar_model_obj.eval()

    with torch.no_grad():
        state_hist_chunks = traj_sol[
            start_chunk : start_chunk + history
        ]  # (history, T_inner, nx, 1)
        ctrl_hist_chunks = traj_ctrl[
            start_chunk : start_chunk + history
        ]  # (history, T_inner, n_ctrl, 1)
        ctrl_seq_chunks = traj_ctrl[
            start_chunk + history - 1 : start_chunk + history - 1 + steps
        ]  # (steps, T_inner, n_ctrl, 1)
        true_future_chunks = traj_sol[
            start_chunk + history : start_chunk + history + steps
        ]  # (steps, T_inner, nx, 1)

        if int(ctrl_seq_chunks.shape[0]) != steps:
            raise ValueError(
                f"Control sequence length mismatch: expected {steps}, got {ctrl_seq_chunks.shape[0]}"
            )

        x_state_hist = (
            state_hist_chunks.permute(0, 3, 1, 2).unsqueeze(0).to(device).float()
        )  # (1, history, 1, T_inner, nx)
        x_ctrl_hist = (
            ctrl_hist_chunks.permute(0, 3, 1, 2).unsqueeze(0).to(device).float()
        )  # (1, history, 1, T_inner, n_ctrl)
        u0 = x_state_hist[:, -1]  # (1, 1, T_inner, nx)
        ctrl_seq = (
            ctrl_seq_chunks.permute(0, 3, 1, 2).unsqueeze(0).to(device).float()
        )  # (1, steps, 1, T_inner, n_ctrl)

        if variant == "A":
            pred_roll, hidden_states_final = rollout_variant_a_carry(
                ar_model_obj=ar_model_obj,
                u0=u0,
                ctrl_seq=ctrl_seq,
                steps=steps,
            )
            warmup_steps_used = 0
        elif variant == "B":
            pred_roll = rollout_variant_b_reset_each_step(
                ar_model_obj=ar_model_obj,
                u0=u0,
                ctrl_seq=ctrl_seq,
                steps=steps,
            )
            hidden_states_final = None
            warmup_steps_used = 0
        else:
            pred_roll, warmup_steps_used, hidden_states_final = (
                rollout_variant_c_warmup_then_carry(
                    ar_model_obj=ar_model_obj,
                    x_state_hist=x_state_hist,
                    x_ctrl_hist=x_ctrl_hist,
                    u0=u0,
                    ctrl_seq=ctrl_seq,
                    steps=steps,
                    history=history,
                    warmup_steps=warmup_steps,
                )
            )

        pred_future_chunks = pred_roll[0, 1:].permute(
            0, 2, 3, 1
        )  # (steps, T_inner, nx, 1)
        true_future_chunks = true_future_chunks.to(device).float()  # (steps, T_inner, nx, 1)
        if pred_future_chunks.shape != true_future_chunks.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: pred={tuple(pred_future_chunks.shape)}, true={tuple(true_future_chunks.shape)}"
            )

        pred_future_raw = pred_future_chunks.squeeze(-1).reshape(
            steps * t_inner, nx
        )  # (steps*T_inner, nx)
        true_future_raw = true_future_chunks.squeeze(-1).reshape(
            steps * t_inner, nx
        )  # (steps*T_inner, nx)

        history_raw = traj_sol[start_chunk : start_chunk + history].to(device).float()
        history_raw = history_raw.squeeze(-1).reshape(history * t_inner, nx)
        pred_full_raw = torch.cat([history_raw, pred_future_raw], dim=0)

        true_full_raw = traj_sol[
            start_chunk : start_chunk + history + steps
        ].to(device).float()
        true_full_raw = true_full_raw.squeeze(-1).reshape((history + steps) * t_inner, nx)

        diff = pred_future_raw - true_future_raw
        rel_l2_full = (
            torch.linalg.norm(diff.reshape(-1))
            / (torch.linalg.norm(true_future_raw.reshape(-1)) + 1e-12)
        ).item()
        mse_full = torch.mean(diff**2).item()

        y_end_pred = pred_roll[:, -1]  # (1, 1, T_inner, nx)
        y_end_true = true_future_chunks[-1].permute(
            2, 0, 1
        ).unsqueeze(0)  # (1, 1, T_inner, nx)
        endpoint_rel_l2 = (
            torch.linalg.norm((y_end_pred - y_end_true).reshape(1, -1), dim=1)
            / (torch.linalg.norm(y_end_true.reshape(1, -1), dim=1) + 1e-12)
        ).mean().item()

        chunk_rel_l2 = []
        for k in range(steps):
            diff_k = (pred_future_chunks[k] - true_future_chunks[k]).reshape(-1)
            true_k = true_future_chunks[k].reshape(-1)
            rel_k = torch.linalg.norm(diff_k) / (torch.linalg.norm(true_k) + 1e-12)
            chunk_rel_l2.append(float(rel_k.item()))

        raw_rel_l2_curve = torch.linalg.norm(diff, dim=1) / (
            torch.linalg.norm(true_future_raw, dim=1) + 1e-12
        )  # (steps*T_inner,)

        raw_steps = int(steps * t_inner)
        traversed_steps = float(raw_steps * float(timesteps_per_control_action))

    if hasattr(ar_model_obj, "train") and was_training:
        ar_model_obj.train()

    result = {
        "variant": variant,
        "traj_idx": int(traj_idx),
        "start_chunk": int(start_chunk),
        "history": int(history),
        "warmup_steps_used": int(warmup_steps_used),
        "steps_chunk": int(steps),
        "chunk_len": int(t_inner),
        "nx": int(nx),
        "n_ctrl": int(n_ctrl),
        "raw_steps": int(raw_steps),
        "control_actions": int(raw_steps),
        "traversed_steps": float(traversed_steps),
        "physical_time": (
            float(traversed_steps * float(dt)) if dt is not None else None
        ),
        "metrics": {
            "rel_l2_full": float(rel_l2_full),
            "mse_full": float(mse_full),
            "endpoint_rel_l2": float(endpoint_rel_l2),
            "chunk_rel_l2": chunk_rel_l2,
        },
        "pred_roll": pred_roll.detach().cpu(),  # (1, steps+1, 1, T_inner, nx)
        "pred_future_chunks": pred_future_chunks.detach().cpu(),  # (steps, T_inner, nx, 1)
        "true_future_chunks": true_future_chunks.detach().cpu(),  # (steps, T_inner, nx, 1)
        "pred_future_raw": pred_future_raw.detach().cpu(),  # (steps*T_inner, nx)
        "true_future_raw": true_future_raw.detach().cpu(),  # (steps*T_inner, nx)
        "pred_full_raw": pred_full_raw.detach().cpu(),  # ((history+steps)*T_inner, nx)
        "true_full_raw": true_full_raw.detach().cpu(),  # ((history+steps)*T_inner, nx)
        "raw_rel_l2_curve": raw_rel_l2_curve.detach().cpu(),  # (steps*T_inner,)
        "ctrl_seq_used": ctrl_seq.detach().cpu(),  # (1, steps, 1, T_inner, n_ctrl)
        "hidden_states_final": hidden_states_final,
    }
    return result


def plot_single_trajectory_rollout_contours(
    true_states,
    rollout_pred,
    x_cord,
    dt_value: float,
    sample_id_value: int | None = None,
    title_prefix: str = "RNO",
    training_horizon_raw_step: int | None = None,
    normalize_error_to_max_true: bool = True,
):
    """Plot true/predicted rollout contours and absolute error for one trajectory.

    Parameters
    ----------
    true_states:
        Array-like with shape ``(T_total, nx)``.
    rollout_pred:
        Array-like with shape ``(T_total, nx)``.
    x_cord:
        Spatial coordinates of shape ``(nx,)`` or ``(nx,1)``.
    dt_value:
        Physical time step between consecutive raw states.
    sample_id_value:
        Optional trajectory id for plot title.
    title_prefix:
        Label prefix for the predicted panel title.
    training_horizon_raw_step:
        Optional raw-step index (in plotted trajectory coordinates) at which to draw
        a dotted horizontal line on the time axis (e.g., history + H_train boundary).
    normalize_error_to_max_true:
        If True, error panel shows ``|u_true-u_pred| / max(|u_true|) * 100``.
        If False, error panel shows absolute error ``|u_true-u_pred|``.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    true_states = np.asarray(true_states)
    rollout_pred = np.asarray(rollout_pred)
    x_axis = np.asarray(x_cord).reshape(-1)

    if true_states.ndim != 2 or rollout_pred.ndim != 2:
        raise ValueError(
            f"true_states and rollout_pred must be rank-2 (T_total,nx), got "
            f"{true_states.shape} and {rollout_pred.shape}"
        )
    if true_states.shape != rollout_pred.shape:
        raise ValueError(
            f"Shape mismatch: true_states={true_states.shape}, rollout_pred={rollout_pred.shape}"
        )
    if true_states.shape[1] != x_axis.shape[0]:
        raise ValueError(
            f"x_cord length {x_axis.shape[0]} must match spatial size {true_states.shape[1]}"
        )

    T_total = int(true_states.shape[0])
    time_axis = np.linspace(0.0, float(dt_value) * (T_total - 1), T_total)
    abs_error = np.abs(true_states - rollout_pred)
    if normalize_error_to_max_true:
        max_true_mag = float(np.max(np.abs(true_states)))
        error = 100.0 * abs_error / (max_true_mag + 1e-12)
        error_label = r"|u-u_hat| / max|u_true| (%)"
        error_title = "Normalized Abs Error (%)"
    else:
        error = abs_error
        error_label = "Abs. Error"
        error_title = "Absolute Error"

    vmin = float(min(true_states.min(), rollout_pred.min()))
    vmax = float(max(true_states.max(), rollout_pred.max()))

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(
        3,
        3,
        height_ratios=[20, 0.3, 1.0],
        width_ratios=[1, 1, 1],
        hspace=0.15,
        wspace=0.2,
        bottom=0.15,
        top=0.88,
        left=0.08,
        right=0.95,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)

    im0 = ax0.contourf(
        x_axis, time_axis, true_states, levels=120, cmap="inferno", vmin=vmin, vmax=vmax
    )
    ax0.set_title("Ground Truth", fontsize=18)
    ax0.set_xlabel("x", fontsize=14)
    ax0.set_ylabel("t", fontsize=14)
    ax0.tick_params(labelsize=12)

    im1 = ax1.contourf(
        x_axis,
        time_axis,
        rollout_pred,
        levels=120,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    pred_title = f"{title_prefix} Prediction"
    ax1.set_title(pred_title, fontsize=18)
    ax1.set_xlabel("x", fontsize=14)
    ax1.tick_params(labelsize=12, labelleft=False)

    im2 = ax2.contourf(x_axis, time_axis, error, levels=120, cmap="magma")
    ax2.set_title(error_title, fontsize=18)
    ax2.set_xlabel("x", fontsize=14)
    ax2.tick_params(labelsize=12, labelleft=False)

    # Optional horizontal marker for training horizon along t-axis.
    if training_horizon_raw_step is not None:
        idx = int(training_horizon_raw_step)
        if 0 <= idx < T_total:
            t_mark = float(time_axis[idx])
            for ax in (ax0, ax1, ax2):
                ax.axhline(
                    y=t_mark,
                    color="white",
                    linestyle=":",
                    linewidth=2.0,
                    alpha=0.95,
                )
        else:
            # Keep plotting robust if caller passes a marker outside current window.
            pass

    cbar_ax1 = fig.add_subplot(gs[2, 0:2])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label("u", fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    cbar_ax2 = fig.add_subplot(gs[2, 2])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label(error_label, fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    suptitle = "Single-Trajectory Full Rollout"
    if sample_id_value is not None:
        suptitle += f" (sample={int(sample_id_value)})"
    fig.suptitle(suptitle, fontsize=18)

    plt.show()
