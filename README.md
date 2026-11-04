# KART LAB

Un kart que aprende a conducir solo, con aprendizaje por refuerzo (PPO), por circuitos generados proceduralmente. Un visor pseudo-3D de estilo 16 bits (suelo en perspectiva tipo «Mode 7», cámara detrás del kart y pixel art) permite ver cómo mejora, en directo y en vídeo.

## Principios

- **Simulación y visualización separadas.** La simulación es 2D (vista cenital), rápida y sin gráficos, y es lo único que usa el entrenamiento. La IA aprende de sensores numéricos (rayos de distancia, velocidad…), nunca de píxeles. El render pseudo-3D es una capa aparte que solo lee el estado de la simulación.
- **Todo en CPU.** Nada de CUDA, OpenGL ni shaders. PyTorch se instala en su versión solo CPU. Pensado para un portátil modesto sin tarjeta gráfica.
- **Todo original.** Gráficos generados por código o con licencia CC0 y fuentes con licencia OFL. La procedencia de cada recurso está en [docs/assets.md](docs/assets.md).

## Requisitos

- Windows o Linux (la CI prueba los dos).
- [uv](https://docs.astral.sh/uv/). Si no tienes Python 3.12 o 3.13, uv lo descarga.
- No hace falta tarjeta gráfica.

## Instalación

```sh
git clone https://github.com/jupiutrera/kartlab.git
cd kartlab
uv sync
uv run pre-commit install
```

## Conducir a mano

```sh
uv run python scripts/drive.py                    # circuito escaparate «Puerto»
uv run python scripts/drive.py --track canyon     # harbor, canyon, plateau, eval:1-20 o una semilla
uv run python scripts/drive.py --help             # todas las opciones
```

| Tecla | Acción |
| --- | --- |
| Flechas o WASD | Girar, acelerar y frenar (con el kart parado, marcha atrás) |
| R | Volver a la línea de salida |
| 1, 2, 3 | Circuitos escaparate: Puerto, Cañón y Meseta |
| E | Siguiente circuito de evaluación |
| N | Circuito aleatorio nuevo |
| C | Cámara de seguimiento o circuito completo |
| Tab | Piloto automático de referencia |
| P | Pausa |
| F11 | Pantalla completa |
| H | Mostrar u ocultar la ayuda |
| Esc | Salir |

La ventana usa la mayor escala entera de 320×180 que cabe en la pantalla: ×4 (1280×720) en una de 1366×768 y ×5 (1600×900) en una de 1920×1080.

Para ver de un vistazo cómo salen los circuitos generados:

```sh
uv run python scripts/track_gallery.py --showcase --count 8 --out galeria.png
```

## Entorno de aprendizaje por refuerzo

`KartEnv` ([src/kartlab/env/kart_env.py](src/kartlab/env/kart_env.py)) es un entorno Gymnasium registrado como `kartlab/Kart-v0`:

- **Observación**: 11 rayos de distancia a los bordes de la pista en un abanico frontal de 180°, velocidad, ángulo respecto a la dirección de la pista (seno y coseno), acción anterior, posición real del volante y velocidad lateral (derrape).
- **Acción**: dirección y acelerador/freno, continuos en [-1, 1]. El agente decide cada 4 pasos de física (unos 15 Hz).
- **Recompensa**: avance neto a lo largo de la línea central (negativo hacia atrás), bonus por vuelta, penalización pequeña por volantazos y penalización con fin de episodio al salirse. La velocidad no se premia por sí misma.
- **Fin de episodio**: al salirse, si no progresa en 4 s y por límite de tiempo.
- **Circuitos**: uno aleatorio nuevo y una salida aleatoria en cada episodio, o un circuito fijo (`track="harbor"`). Los de evaluación nunca salen al entrenar.

Todos los parámetros están en `EnvConfig`.

```python
from kartlab.env.kart_env import KartEnv

env = KartEnv(track="harbor", terminate_off_track=False)
obs, info = env.reset(seed=0)
```

Para medir los pasos por segundo (un entorno, y `DummyVecEnv` frente a `SubprocVecEnv` con 1, 4, 6 y 8):

```sh
uv run python scripts/benchmark_env.py
```

## Entrenar

Con `configs/random_tracks.yaml` (1,5M pasos, unos 20 minutos en un portátil con un i5-1135G7 y sin tarjeta gráfica), el agente completa 3 vueltas sin salirse en los 20 circuitos de evaluación, que nunca ve durante el entrenamiento. También las completa en los tres circuitos escaparate. Los detalles están en [docs/experimentos.md](docs/experimentos.md).

```sh
uv run python scripts/train.py configs/fixed_track.yaml      # un circuito fijo sencillo
uv run python scripts/train.py configs/random_tracks.yaml    # circuitos aleatorios (generalizar)
uv run python scripts/live.py                                # visor del último run, en otra terminal
uv run python scripts/evaluate.py --tracks 20                # 3 vueltas en los 20 circuitos reservados
uv run tensorboard --logdir runs
```

Cada entrenamiento crea `runs/<fecha>_<hora>_<nombre>/` con:
- la configuración y el commit;
- los checkpoints, que llevan los pasos en el nombre y van con sus estadísticas de VecNormalize;
- las métricas en `metrics.csv` y TensorBoard;
- las evaluaciones en `eval.csv` y el mejor modelo en `best_model.zip`.

El entrenamiento va con prioridad baja (también sus subprocesos), así que el visor puede quedarse abierto al lado:
- muestra la última versión conduciendo en un circuito reservado, o en el del entrenamiento si es fijo;
- deja las versiones anteriores como fantasmas;
- dibuja al lado la curva de recompensa.

Cerrar el visor no afecta al entrenamiento.

`configs/demo_live.yaml` es la configuración para enseñar en directo cómo aprende desde cero. Los resultados de cada entrenamiento se anotan en [docs/experimentos.md](docs/experimentos.md).

Para elegir el tipo de VecEnv, el número de entornos y los hilos de PyTorch más rápidos en tu equipo:

```sh
uv run python scripts/benchmark_train.py
```

## Comandos

| Qué | Comando |
| --- | --- |
| Tests | `uv run pytest` |
| Solo el smoke test | `uv run pytest -m smoke` |
| Lint | `uv run ruff check` |
| Formato | `uv run ruff format` |
| Todos los hooks de pre-commit | `uv run pre-commit run --all-files` |

## Estructura

```
src/kartlab/
  common/   utilidades compartidas (config, semillas, ficheros)
  sim/      simulación 2D: circuitos y física (solo numpy)
  env/      entorno Gymnasium
  train/    entrenamiento PPO, checkpoints y métricas
  render/   render pseudo-3D (numpy + pygame-ce)
  show/     visor, vídeos y carreras de fantasmas
scripts/    puntos de entrada
configs/    configuraciones de entrenamiento (YAML)
assets/     fuentes y paleta
docs/       procedencia de recursos y registro de experimentos
tests/
```

`sim`, `env` y `train` nunca importan `render`, `show` ni pygame; un test lo comprueba.

## Estado

| Fase | Contenido | Estado |
| --- | --- | --- |
| 0 | Esqueleto del repo | Hecha |
| 1 | Simulación 2D y control manual | Hecha |
| 2 | Entorno Gymnasium | Hecha |
| 3 | Entrenamiento | En revisión |
| 4 | Render pseudo-3D y dirección de arte | Pendiente |
| 5 | Herramientas de exhibición | Pendiente |
| 6 | Editor de circuitos (opcional) | Pendiente |

## Licencia

Código bajo licencia [MIT](LICENSE). Los recursos de terceros conservan su propia licencia; ver [docs/assets.md](docs/assets.md).
