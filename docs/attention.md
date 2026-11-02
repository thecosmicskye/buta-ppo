# Attention models

Export the actor-critic as TorchScript with `torch.jit.trace` or `torch.jit.script`, then set `BUTA_SHARED_MODEL` to the file path before starting training:

```sh
export BUTA_SHARED_MODEL=/path/to/model.pt
./build/buta_ppo
```

The model receives `[players, observations]` and returns `(logits, values)` with shapes `[players, actions]` and `[players]`. Its input order and size must match the active observation builder. Buta's default builder uses 319 features with `max_players_per_team = 3`; its default action space has 90 actions and applies action masks.

The RocketChad3 example expects 168 features, so it needs a matching observation builder.

For the RocketChad attention model, parameters under `base.blocks` and `base.readout` use the hidden-weight optimizer group, and `base.actor` and `base.critic` use their respective sections. Other model layouts may need a different `parameter_layout` in `src/rl/actor_critic.cpp`.

To resume, keep `BUTA_SHARED_MODEL` set to the same architecture and buffers. Scripted checkpoints are for the trainer and are not compatible with the stock RLBot deployment.
