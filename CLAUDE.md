# CLAUDE.md

KART LAB: un kart que aprende a conducir con RL (PPO) sobre una simulación 2D, con un visor pseudo-3D de 16 bits aparte.

La especificación completa está en `docs/especificacion.md`. Es solo local (está en `.gitignore`); si no existe, pídesela al usuario. Si algo de la especificación es ambiguo o ves una opción mejor, proponlo antes de implementarlo.

## Comandos

```sh
uv sync                              # instalar / sincronizar el entorno
uv run pytest                        # todos los tests
uv run pytest -m smoke               # smoke test (lo que ejecuta la CI al final)
uv run ruff check --fix              # lint
uv run ruff format                   # formato
uv run pre-commit run --all-files    # todos los hooks
uv add <paquete>                     # dependencia (uv add --dev para desarrollo)
```

## Estructura

- `src/kartlab/common/`: utilidades sin dependencias de otras capas (config YAML, semillas, ficheros UTF-8, escritura atómica).
- `src/kartlab/sim/`: simulación 2D cenital, circuitos y física. Solo numpy.
- `src/kartlab/env/`: entorno Gymnasium sobre `sim`.
- `src/kartlab/train/`: PPO, entornos vectorizados, checkpoints, evaluación y métricas.
- `src/kartlab/render/`: render pseudo-3D que lee el estado de `sim`. numpy + pygame-ce.
- `src/kartlab/show/`: visor en directo, vídeos y carreras de fantasmas.
- `scripts/`: lanzadores finos con `if __name__ == "__main__"`; la lógica va en el paquete.
- `configs/`: YAML de entrenamiento. `assets/`: fuentes OFL y paleta.
- `docs/assets.md`: procedencia y licencia de cada recurso. `docs/experimentos.md`: registro de entrenamientos.
- `runs/<fecha>_<nombre>/` (ignorado por git): config, hash del commit, checkpoints y métricas de cada run.

## Arquitectura (no negociable)

- La IA aprende de sensores numéricos (rayos, velocidad, ángulo…), nunca de píxeles.
- `sim`, `env`, `train` y `common` nunca importan `render`, `show` ni pygame. Lo comprueba `tests/test_import_boundaries.py`.
- El render nunca se ejecuta durante el entrenamiento.
- Paso de física fijo de 1/60 s; el agente decide cada 4 pasos y su acción se mantiene entre medias.

## Nada de GPU (no negociable)

- Sin CUDA, OpenGL, shaders ni librerías que los necesiten. Todo el cálculo y el dibujo en CPU.
- PyTorch solo CPU desde el índice `pytorch-cpu` de `pyproject.toml` (`[tool.uv.sources]`). No quites esa configuración. Lo comprueba `tests/test_cpu_only.py`.
- pygame-ce solo con superficies software: nada de `pygame.OPENGL` ni de `pygame._sdl2.video.Renderer`.
- Render: precalcular todo lo posible, arrays preasignados y ningún bucle de Python por píxel. numba solo si numpy no basta.
- Entrenamiento: red MLP pequeña, hilos de PyTorch limitados y prioridad baja del proceso y sus subprocesos.

## Windows

- Los entornos en paralelo arrancan con spawn: los scripts necesitan `if __name__ == "__main__"` y nada pesado a nivel de módulo.
- Ficheros de texto siempre con `encoding="utf-8"` explícito (ruff `PLW1514` lo vigila).
- El visor declara el proceso DPI-aware antes de crear la ventana.
- Los mensajes de commit con tildes se pasan por Bash con heredoc: PowerShell 5.1 los estropea.

## Código

- Código, nombres, comentarios y docstrings en inglés. README, documentación y commits en español.
- Escrituras que lee otro proceso (checkpoints, VecNormalize, CSV de métricas): fichero temporal y `os.replace`.
- Propiedad intelectual: todo original. Nada de personajes, sprites, circuitos, nombres, logotipos, música ni sonidos de Nintendo ni de otros juegos comerciales. Cada recurso nuevo se anota en `docs/assets.md`.

## Git

- `main` siempre funciona. Una rama por fase: `feat/fase-N-<tema>`.
- Commits pequeños y frecuentes, cada vez que un paso funciona y pasan los tests. Conventional Commits en español (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:`).
- Push al acabar cada bloque de trabajo. Nunca force push a `main` ni reescribir historial publicado.
- Al acabar cada fase, PR con `gh`: qué se ha hecho, cómo probarlo y captura o GIF si hay algo visual. No fusionar sin la revisión del usuario.
- Nunca subir checkpoints, vídeos, `runs/`, ficheros de más de 10 MB ni secretos (`.env`).
- Commit antes de cualquier entrenamiento largo, para que el run quede ligado a un commit.

## Entrenamiento

- Preguntar al usuario antes de lanzar cualquier entrenamiento de más de 30 minutos.
- Configuración en YAML y semillas fijas. Checkpoints con el número de pasos en el nombre.
- Anotar cada entrenamiento relevante en `docs/experimentos.md`: commit, qué se probó, resultados y conclusiones.
- Si el agente no aprende, diagnosticar (recompensa, observaciones, escalas) antes de tocar hiperparámetros.

## Fases

0. Esqueleto del repo.
1. Simulación 2D, vista cenital de depuración y control manual.
2. Entorno Gymnasium, check_env, benchmark y tests de la recompensa.
3. Entrenamiento: circuito fijo sencillo, después circuitos aleatorios.
4. Render pseudo-3D y dirección de arte.
5. Herramientas de exhibición.
6. (Opcional) Editor de circuitos.

Al final de cada fase: parar, dar al usuario los comandos para probarla y esperar su visto bueno antes de empezar la siguiente.
