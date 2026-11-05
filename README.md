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
uv run python scripts/drive.py                    # circuito escaparate «Puerto», vista 3D
uv run python scripts/drive.py --track canyon     # harbor, canyon, plateau, eval:1-20 o una semilla
uv run python scripts/drive.py --variant night    # day, sunset o night
uv run python scripts/drive.py --help             # todas las opciones
```

![Vista de persecución](docs/img/fase-4-persecucion.png)

| Tecla | Acción |
| --- | --- |
| Flechas o WASD | Girar, acelerar y frenar (con el kart parado, marcha atrás) |
| R | Volver a la línea de salida |
| 1, 2, 3 | Circuitos escaparate: Puerto, Cañón y Meseta |
| E | Siguiente circuito de evaluación |
| N | Circuito aleatorio nuevo |
| C | Cámara: persecución, retransmisión o cenital |
| V | Día, atardecer o noche |
| G | Mostrar los sensores (rayos) sobre el suelo |
| T | Vista cenital de depuración |
| Tab | Piloto automático de referencia |
| P | Pausa |
| F3 | Diagnóstico: fps y milisegundos de cada parte del dibujo |
| L | Modo ligero (30 fps, suelo a media resolución vertical, menos efectos) |
| F11 | Pantalla completa |
| H | Mostrar u ocultar la ayuda |
| Esc | Salir |

La ventana usa la mayor escala entera de 320×180 que cabe en la pantalla: ×4 (1280×720) en una de 1366×768 y ×5 (1600×900) en una de 1920×1080. A pantalla completa en 1080p, ×6 encaja exacto.

### Cómo se dibuja

- **El mundo** se dibuja a 320×180 con la paleta Resurrect 64 y se escala por un factor entero, sin suavizado. Texto, paneles y minimapa van encima, a la resolución de la ventana.
- **El suelo** está en perspectiva, calculado por filas con numpy.
  - La textura sale del circuito: asfalto, pianos, hierba segada y meta a cuadros.
  - Tiene la iluminación y las sombras pintadas al generarla, y mipmaps.
  - Lleva niebla tramada hacia el horizonte.
- **El cielo** tiene degradado tramado, y montañas, colinas y nubes en parallax.
- **Los karts** son modelos de vóxeles dibujados por código con *sprite stacking* en 32 ángulos y al tamaño exacto en que se ven. El decorado también se dibuja por código.
- **Efectos:** sombras, polvo, humo, marcas de derrape que se quedan y halos de los fantasmas.
- **Día, atardecer y noche** son reasignaciones de la misma paleta, sin coste por fotograma.

Para medir el tiempo de cada parte en tu equipo:

```sh
uv run python scripts/benchmark_render.py
```

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
- dibuja la curva de recompensa;
- usa la vista 3D con cámara de retransmisión, y la T cambia a la vista cenital.

Cerrar el visor no afecta al entrenamiento.

`configs/demo_live.yaml` es la configuración para enseñar en directo cómo aprende desde cero. Los resultados de cada entrenamiento se anotan en [docs/experimentos.md](docs/experimentos.md).

Para elegir el tipo de VecEnv, el número de entornos y los hilos de PyTorch más rápidos en tu equipo:

```sh
uv run python scripts/benchmark_train.py
```

## Vídeos

Los vídeos se renderizan fuera de tiempo real y van directos a ffmpeg (sin fotogramas sueltos en disco), siempre a 1920×1080 y 60 fps exactos (H.264, `yuv420p`, crf 18). El mundo de 320×180 se escala ×6 y el HUD se dibuja a 1080p. No hace falta ventana ni GPU. Cada paso de física (1/60 s) es un fotograma, con acciones deterministas, así que el mismo run y el mismo circuito dan siempre el mismo vídeo.

```sh
RUN=runs/2026-11-04_2159_random-tracks
uv run python scripts/make_video.py progression --run $RUN --track harbor --out videos/progresion.mp4
uv run python scripts/make_video.py race --run $RUN --track harbor --laps 2 --out videos/carrera.mp4
uv run python scripts/make_video.py demo --run $RUN --track eval:3 --laps 3 --out videos/demo.mp4
```

- `progression`: el mismo circuito recorrido por checkpoints en escala logarítmica de pasos (sin entrenar, 10k, 30k, 100k, 300k, 1M y el final). Cada versión rueda una vuelta o `--seconds` segundos (30 por defecto; el doble para la última). Lleva tarjeta de versión con transición suave, contador de pasos, curva de recompensa con un marcador y la cinta de versiones. Las malas siguen rodando por la hierba y se reinician al atascarse 3 s; tras el segundo reinicio se pasa a la siguiente versión.
- `race`: todas las versiones a la vez desde una parrilla de dos columnas, con la mejor al final, sin colisiones. Cámara de retransmisión siguiendo a la mejor, clasificación, minimapa y una etiqueta y un color por versión. Termina cuando la mejor completa `--laps` vueltas.
- `demo`: `best_model.zip` en un circuito que no vio durante el entrenamiento (`eval:3` por defecto), con cámara de persecución y un resumen de vueltas al final.

![Vídeo de progresión](docs/img/fase-5-progresion.png)

![Carrera de versiones](docs/img/fase-5-carrera.png)

![Demo en un circuito inédito](docs/img/fase-5-demo.png)

Otras opciones: `--camera broadcast|chase|top`, `--variant day|sunset|night`, `--steps 0,10000,100000` (versiones a mostrar) y `--normal-priority` (por defecto el proceso y ffmpeg van con prioridad baja). En un i5-1135G7 se renderizan unos 20-25 fotogramas por segundo de vídeo: la progresión de referencia (200 s de vídeo) tarda unos 10 minutos y pesa unos 170 MB. `videos/` está en `.gitignore`.

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
| 3 | Entrenamiento | Hecha |
| 4 | Render pseudo-3D y dirección de arte | En revisión |
| 5 | Herramientas de exhibición | En curso (vídeos hechos) |
| 6 | Editor de circuitos (opcional) | Pendiente |

## Licencia

Código bajo licencia [MIT](LICENSE). Los recursos de terceros conservan su propia licencia; ver [docs/assets.md](docs/assets.md).
