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
| 0 | Esqueleto del repo | En curso |
| 1 | Simulación 2D y control manual | Pendiente |
| 2 | Entorno Gymnasium | Pendiente |
| 3 | Entrenamiento | Pendiente |
| 4 | Render pseudo-3D y dirección de arte | Pendiente |
| 5 | Herramientas de exhibición | Pendiente |
| 6 | Editor de circuitos (opcional) | Pendiente |

## Licencia

Código bajo licencia [MIT](LICENSE). Los recursos de terceros conservan su propia licencia; ver [docs/assets.md](docs/assets.md).
