"""Version of the driving physics, part of the key of every saved time-trial record.

A record is a lap driven with a given physics: if the kart handles differently, the
same poses are no longer a lap that can be driven, and comparing times is unfair.

Bump ``PHYSICS_VERSION`` whenever a change alters how a kart moves for the same
inputs: ``sim/kart.py`` (``KartParams`` defaults, ``step_kart``), ``sim/drift.py``
(``DriftParams``, ``DriftState``), the ``DT`` step, or how ``Racer`` counts laps.
Old records are then ignored (never deleted), so nobody has to remember anything
else. ``tests/test_time_trial.py::test_physics_fingerprint`` fails when the physics
changes without a bump: update the version and its pinned numbers together.
"""

PHYSICS_VERSION = 1
