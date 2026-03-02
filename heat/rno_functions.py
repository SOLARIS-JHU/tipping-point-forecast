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
):
    """Evaluate endpoint Relative L2 for rollout variants A/B/C by horizon.

    Each loader batch is expected to yield:
    ``(x_state_hist, x_ctrl_hist, x_ctrl_seq, y_endpoint)`` with shapes
    ``(B, T_hist, 1, T_inner, nx)``, ``(B, T_hist, 1, T_inner, n_ctrl)``,
    ``(B, h, 1, T_inner, n_ctrl)``, ``(B, 1, T_inner, nx)``.
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

            results[h] = {"N": int(n_total)}
            for v in variants:
                results[h][v] = per_variant_sum[v] / max(n_total, 1)

    if hasattr(ar_model_obj, "train") and was_training:
        ar_model_obj.train()

    header = "Steps  N    " + "  ".join([f"RelL2-{v}" for v in variants])
    print("Endpoint Rel. L2 by horizon and variant")
    print("-" * len(header))
    print(header)
    for h in eval_steps:
        vals = "  ".join([f"{results[h][v]:.6e}" for v in variants])
        print(f"{h:<5d}  {results[h]['N']:<4d} {vals}")

    return results
