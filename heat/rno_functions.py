"""Reusable RNO control-conditioning modules for heat notebooks.

This module centralizes shared components used by:
- ``train_RNO_broadcast.ipynb``
- ``train_RNO_sin_embed.ipynb``

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
