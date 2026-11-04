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

## 2026-11-04 · random-tracks con la física corregida

- **Commit:** `cbb9126`
- **Run:** `runs/2026-11-04_2129_random-tracks/`
- **Configuración:** igual que el anterior, pero con el rozamiento sin zona muerta y el desempate del mejor modelo.
- **Qué se probó:** si quitar la zona muerta del acelerador arregla `eval:14`.
- **Resultados:** 1,5M pasos en 23 min 47 s.
  - El ritmo cayó a ratos a 600 pasos/s; algo ajeno al entrenamiento cargó el portátil.
  - **Primera vuelta:** a los 46k pasos.
  - **Evaluación periódica:** 90 % a los 100k pasos y 100 % a los 200k, pero luego oscila entre el 70 y el 100 %.
  - **Modelo final:** 90 % en los 20 de evaluación (falla `eval:2` y `eval:14`) y se sale en Puerto y Cañón.
  - **Modelo del run anterior evaluado con la física nueva:** sigue en 95 % y completa los tres escaparate. Sigue atascándose en `eval:14`.
- **Diagnóstico de `eval:14`:**
  - Llega a la horquilla (R 16 m tras una recta de radio > 900 m) a 23,6 m/s y todavía con gas, frena tarde y se para.
  - Se queda mirando 25° hacia el exterior de la curva, con el borde a 6 m delante. Ahí la política pide un poco de freno (−0,06 a −0,12) y no sale.
  - La zona muerta no era la causa principal: es un estado de recuperación que casi no ve al entrenar, porque las salidas aleatorias solo se desvían ±14°.
- **Conclusiones:**
  - El criterio de éxito se cumple (90 %), pero entre entrenamientos hay bastante variabilidad.
  - La oscilación de la evaluación al final apunta a una tasa de aprendizaje demasiado alta para afinar.
- **Siguiente paso:**
  - Salidas con un desvío de hasta ±34° (`start_heading: 0.6`), para aprender a recuperarse.
  - Tasa de aprendizaje que baja linealmente hasta 0 (`lr_schedule: linear`).

## 2026-11-04 · random-tracks (circuitos aleatorios)

- **Commit:** `6fe963e`
- **Run:** `runs/2026-11-04_2108_random-tracks/`
- **Configuración:** `configs/random_tracks.yaml` con `--steps 1500000`. Respecto al anterior:
  - un circuito aleatorio nuevo en cada episodio, de 700-1.200 m, 10-14 m de ancho y curvatura entre 0,2 y 0,8;
  - el resto, igual. Sin currículo.
- **Qué se probó:** el segundo paso de la fase 3, generalizar a circuitos que nunca ha visto.
- **Resultados:** 1,5M pasos en 15 min 9 s, unos 1.650 pasos/s de media.
  - **Primera vuelta completada** a los 45k pasos (23 s).
  - **Evaluación periódica** (10 circuitos reservados, 3 vueltas): 50 % a los 100k pasos, 90 % a los 200k y 100 % desde los 800k, con vueltas medias de 38 s.
  - **Modelo final, 20 circuitos de evaluación:** 95 % de éxito (19/20). Vueltas de 37-40 s; el piloto automático de referencia tarda unos 41 s.
  - **Escaparate** (no vistos nunca): completa Puerto, Cañón y Meseta, con vueltas de 41,3 s, 47,9 s y 54,6 s.
  - **«Mejor» modelo** (el primero con 100 %, a los 300k pasos): también 95 %, pero se sale en Puerto y Cañón.
- **Diagnóstico del fallo** (`eval:14`):
  - Llega a 20 m/s a una curva de 16 m de radio al final de una recta muy larga y frena hasta pararse en plena curva.
  - Parado, la política pide un 6 % de acelerador. El rozamiento constante de 2 m/s² anulaba cualquier acelerador por debajo del 14 %, así que el kart no se movía y acababa en «atasco».
- **Conclusiones:**
  - Con circuitos aleatorios generaliza muy bien, sin currículo, y se cumple el criterio de éxito (95 % ≥ 90 %).
  - Ver circuitos variados también le enseña a frenar para las horquillas, que no aprendía en el circuito fijo.
  - El fallo restante viene de un defecto de la física (zona muerta del acelerador) más que del aprendizaje.
  - Con empate al 100 %, el criterio del mejor modelo se quedaba con el primero, que es peor.
- **Siguiente paso:**
  - Rozamiento de 0,5 m/s² más 0,065 por m/s, que mantiene la velocidad de crucero de 24 m/s y quita la zona muerta (`5e62911`).
  - Desempate del mejor modelo por el tiempo de vuelta (`cbb9126`).
  - Volver a entrenar con la misma configuración.

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
