"""KART LAB: a kart that learns to drive with reinforcement learning.

Layers (imports only flow downwards):

- ``common``: dependency-free helpers shared by every layer.
- ``sim``: 2D top-down simulation (tracks, physics). numpy only.
- ``env``: Gymnasium environment on top of ``sim``.
- ``train``: PPO training, checkpoints and metrics.
- ``render``: pseudo-3D renderer that reads ``sim`` state. Uses pygame-ce.
- ``show``: viewer, videos and ghost races.

``sim``, ``env`` and ``train`` never import ``render``, ``show`` or pygame.
"""

__version__ = "0.1.0"
