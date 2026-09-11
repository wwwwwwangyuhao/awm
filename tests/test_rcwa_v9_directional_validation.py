from __future__ import annotations

import torch

from awm.rcwa.agent_v9 import RCWAV9Agent, RCWAV9Hyperparameters
from awm.rcwa.directional_validation_v9 import _signed_alpha_for_kl


def test_signed_kl_solve_is_always_referenced_to_same_base_policy():
    h = RCWAV9Hyperparameters(
        state_dim=2,
        actor_hidden_dims=(8,4),
        reward_critic_hidden_dims=(8,4),
        risk_critic_hidden_dims=(8,4),
        update_epochs=1,
        minibatch_size=18,
        risk_credit_prefit_epochs=1,
        tail_q_prefit_epochs=1,
    )
    a = RCWAV9Agent(hyperparameters=h, seed=7, device='cpu')
    states = torch.randn(64,2,generator=torch.Generator().manual_seed(9))
    params = list(a.actor.parameters())
    base = a._flat_parameters(params)
    direction = torch.randn(base.shape, generator=torch.Generator().manual_seed(11))
    direction = direction / torch.linalg.vector_norm(direction)
    target = 5e-4
    plus_alpha, plus_kl = _signed_alpha_for_kl(
        a, states=states, base=base, direction=direction, target_kl=target, sign=1.0
    )
    # Deliberately leave the actor at the plus policy to reproduce the original bug.
    a._set_flat_parameters(params, base + plus_alpha * direction)
    minus_alpha, minus_kl = _signed_alpha_for_kl(
        a, states=states, base=base, direction=direction, target_kl=target, sign=-1.0
    )
    assert plus_alpha > 0 and minus_alpha < 0
    assert abs(plus_kl-target) < 2e-7
    assert abs(minus_kl-target) < 2e-7
    assert abs(minus_alpha) > 1e-6
