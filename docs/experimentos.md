# Experimentos

Registro de los entrenamientos relevantes, el más reciente arriba. Cada entrada sigue esta plantilla:

```markdown
## AAAA-MM-DD · <nombre del run>

- **Commit:** `abc1234`
- **Run:** `runs/AAAA-MM-DD_<nombre>/`
- **Configuración:** `configs/<fichero>.yaml` y qué cambia respecto al anterior
- **Qué se probó:**
- **Resultados:** pasos, tiempo, pasos por segundo, recompensa media, vueltas completadas y % de circuitos de evaluación superados
- **Conclusiones:**
- **Siguiente paso:**
```

## 2026-11-04 · fixed-track (circuito fijo)

- **Commit:** `6bba42b`
- **Run:** `runs/2026-11-04_2101_fixed-track/`
- **Configuración:** `configs/fixed_track.yaml`.
  - Meseta: 1.334 m, 15 m de ancho, radio mínimo de 24 m.
  - PPO con 8 × 512 pasos por rollout, lote de 1024 y 10 épocas.
  - SubprocVecEnv ×8 con 1 hilo de PyTorch.
  - 18 observaciones (con volante y derrape) y recompensa por defecto.
- **Qué se probó:** el primer paso de la fase 3, completar vueltas en un circuito fijo y sencillo.
- **Resultados:** 504k pasos en 4 min 54 s, unos 1.700 pasos/s de media.
  - **Primera vuelta completada** a los 41k pasos (40 s de entrenamiento).
  - **Recompensa:** de −3 a 368. Desde unos 300k pasos completa 2 vueltas por episodio, el máximo que da el límite de 150 s desde una salida aleatoria.
  - **En Meseta:** 3 vueltas sin salirse, con vueltas de 54,6-56,7 s (el piloto automático de referencia tarda 60 s).
  - **En los 20 circuitos de evaluación:**

    | Modelo | Éxito |
    | --- | --- |
    | Final | 35 % |
    | Mejor de la evaluación periódica (a los 100k pasos) | 55 % |

  - **En Puerto y Cañón** se sale en la primera horquilla.
- **Conclusiones:**
  - La recompensa y las observaciones bastan para aprender a completar vueltas. En el circuito de entrenamiento aprende muy rápido (menos de 1 minuto hasta la primera vuelta).
  - Se especializa: Meseta no tiene curvas de menos de 24 m, así que no aprende a frenar para las cerradas.
  - Al seguir entrenando, la generalización empeora (55 % → 35 %), una señal de sobreajuste al único circuito.
  - Para la demo en directo (`configs/demo_live.yaml`, mismo circuito y mismos hiperparámetros) se espera que empiece a completar vueltas en menos de un minuto.
- **Siguiente paso:** entrenar con circuitos aleatorios (`configs/random_tracks.yaml`).
