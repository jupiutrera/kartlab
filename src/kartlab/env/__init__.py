"""Gymnasium environment: ray sensors, continuous actions and reward on top of ``sim``.

Must not import pygame or any rendering code.
"""

import gymnasium as gym

ENV_ID = "kartlab/Kart-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="kartlab.env.kart_env:KartEnv")
