"""Rectified conditional flow interpolation and fixed-step ODE solvers."""

import torch


def interpolate(initial, endpoint, time):
    return (1 - time) * initial + time * endpoint, endpoint - initial


def supervised_endpoint(seeds, data, match, cfg):
    endpoint = seeds.state.detach().clone()
    endpoint[:, 3] = float(cfg.inactive_logit)
    endpoint[match.source, :3] = data.target_xyz[match.target]
    endpoint[match.source, 3] = float(cfg.active_logit)
    return endpoint


@torch.no_grad()
def integrate(model, seeds, memory, *, steps=20, solver="heun", on_step=None):
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("positive flow steps and euler/heun solver required")
    velocity_field = getattr(model, "predict_velocity", None)
    if velocity_field is None:
        def velocity_field(state, time, seeds, memory):
            return model(state, time, seeds, memory)["velocity"]
    state = seeds.state.clone()
    if on_step:
        on_step(0, 0.0, state.clone())
    for index in range(steps):
        time = state.new_tensor(index / steps)
        velocity = velocity_field(state, time, seeds, memory)
        proposal = state + velocity / steps
        if solver == "heun":
            end = velocity_field(proposal, time + 1/steps, seeds, memory)
            proposal = state + (velocity + end) / (2 * steps)
        state = proposal
        if not torch.isfinite(state).all():
            raise FloatingPointError("non-finite AG-GFM ODE state")
        if on_step:
            on_step(index + 1, (index + 1)/steps, state.clone())
    return state, model(state, state.new_tensor(1.0), seeds, memory)
