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
uv run python scripts/drive.py       # conducir a mano (3D; --view topdown para la cenital; Mayús = barrido)
uv run python scripts/drive.py --time-trial   # contrarreloj: récord y fantasma (K y F en el juego)
uv run python scripts/track_gallery.py --showcase --out galeria.png
uv run python scripts/benchmark_env.py   # pasos/s: 1 entorno y DummyVecEnv/SubprocVecEnv
uv run python scripts/benchmark_train.py # pasos/s entrenando con PPO
uv run python scripts/train.py configs/fixed_track.yaml   # entrenar (runs/<fecha>_<nombre>/)
uv run python scripts/live.py            # visor en directo del último run (--brain, --rays, --track-file)
uv run python scripts/viewer.py          # visor de exhibición: título y modos ver, carrera y progresión (--brain, B)
uv run python scripts/make_video.py progression|race|demo --run runs/<run> --out videos/x.mp4   # vídeos 1080p60
uv run python scripts/evaluate.py --tracks 20   # 3 vueltas en los circuitos reservados
uv run python scripts/benchmark_render.py   # ms de cada parte del render (--drift, --light)
uv run python scripts/evaluate.py --track-file tracks/ejemplo.json   # un circuito propio (se valida antes)
uv run python scripts/track_file.py validate tracks/ejemplo.json     # validar / exportar (export --seed N)
uv run python scripts/editor.py tracks/ejemplo.json   # editor de circuitos con ratón (--new para uno nuevo)
```

Sin ventana (CI, capturas): `SDL_VIDEODRIVER=dummy` y `scripts/drive.py --frames N --screenshot f.png` (igual en `viewer.py` y `editor.py`).

## Estructura

- `src/kartlab/common/`: utilidades sin dependencias de otras capas (config YAML, semillas, ficheros UTF-8, escritura atómica).
- `src/kartlab/sim/`: simulación 2D cenital, circuitos y física. Solo numpy. Además: barrido (`drift.py`), contrarreloj (`timetrial.py`), `PHYSICS_VERSION` (`version.py`), circuitos en JSON (`trackfile.py`, `trackissues.py`, `trackcli.py`).
- `src/kartlab/env/`: entorno Gymnasium sobre `sim`.
- `src/kartlab/train/`: PPO, entornos vectorizados, checkpoints, evaluación y métricas.
- `src/kartlab/render/`: render pseudo-3D que lee el estado de `sim`. numpy + pygame-ce.
- `src/kartlab/show/`: conducción a mano (`drive.py`), visor de exhibición (`viewer.py`, `scenes.py`, `title.py`), vista cerebro (`brain.py`), visor en directo (`live.py`), editor de circuitos (`editor.py`), conductores de políticas (`drivers.py`), versiones cargadas en segundo plano (`versions.py`), paneles compartidos (`overlays.py`), curvas (`charts.py`), argumentos de línea de comandos (`cli.py`) y vídeos (`video.py`).
- `tracks/`: circuitos en JSON (`ejemplo.json`, `ocho.json`). `records/` (ignorado por git): récords de la contrarreloj.
- `scripts/`: lanzadores finos con `if __name__ == "__main__"`; la lógica va en el paquete.
- `configs/`: YAML de entrenamiento. `assets/`: fuentes OFL y paleta.
- `docs/assets.md`: procedencia y licencia de cada recurso. `docs/experimentos.md`: registro de entrenamientos.
- `runs/<fecha>_<nombre>/` (ignorado por git): config, hash del commit, checkpoints y métricas de cada run.

## Arquitectura (no negociable)

- La IA aprende de sensores numéricos (rayos, velocidad, ángulo…), nunca de píxeles.
- `sim`, `env`, `train` y `common` nunca importan `render`, `show` ni pygame. Lo comprueba `tests/test_import_boundaries.py`.
- El render nunca se ejecuta durante el entrenamiento.
- Paso de física fijo de 1/60 s; el agente decide cada 4 pasos y su acción se mantiene entre medias.
- Convenciones de la simulación: metros, `y` hacia arriba, rumbos en radianes en sentido antihorario, desplazamiento lateral positivo a la izquierda, `steer > 0` gira a la izquierda. La muestra 0 de cada circuito es la línea de meta.
- El progreso es la distancia neta a lo largo de la línea central (`sim/progress.py`): ir hacia atrás resta y una vuelta solo cuenta una vez.
- Las semillas de evaluación (`sim/tracksets.py`) están fuera del rango de entrenamiento; nunca entrenes con ellas.
- Circuitos en fichero: JSON en `tracks/` (`sim/trackfile.py`). `load_track_file` (y `resolve_track` con una ruta `*.json`) valida siempre y construye por el mismo camino que el generador; la IA nunca recibe un circuito sin validar. `tests/test_track_pins.py` fija el hash de los circuitos de evaluación, de 10 semillas de entrenamiento y de los escaparates: no lo cambies para que un cambio pase.
- Cruces a nivel (`tracks/ocho.json`, formato 2 con `crossings` y `start`): `Track.crossings` vacío en todo circuito procedural, que sigue por el camino de siempre. Con cruces, `locate` sigue la rama del hint, `Racer.on_grass` pregunta a todo el circuito y una vuelta exige pasar por todo el trazado. Nunca llames a `locate` sin hint desde algo que se mueve (usa `locate_following`). Solo exhibición: no se entrena ni se evalúa con él.
- El barrido (`sim/drift.py`) es solo de conducción manual: `step_kart(..., drift=None)` ejecuta la física de siempre y el entorno nunca pasa un `DriftState`. `tests/test_drift.py` compara con `==` contra una copia literal de `step_kart`; no toques esa ruta.
- Si cambias cómo se mueve el kart (`sim/kart.py`, `sim/drift.py`, `DT`, el conteo de vueltas), sube `PHYSICS_VERSION` en `sim/version.py`: invalida los récords de contrarreloj guardados en `records/` (ignorado por git). `test_physics_fingerprint` en `tests/test_time_trial.py` falla si lo olvidas; actualiza ahí también los valores fijados.
- Entorno: `env/kart_env.py` (`KartEnv`, `EnvConfig`). Cualquier cambio de la recompensa necesita su test en `tests/test_env.py` (ir y volver sobre la meta y dar vueltas en círculo nunca deben sumar).

## Nada de GPU (no negociable)

- Sin CUDA, OpenGL, shaders ni librerías que los necesiten. Todo el cálculo y el dibujo en CPU.
- PyTorch solo CPU desde el índice `pytorch-cpu` de `pyproject.toml` (`[tool.uv.sources]`). No quites esa configuración. Lo comprueba `tests/test_cpu_only.py`.
- pygame-ce solo con superficies software: nada de `pygame.OPENGL` ni de `pygame._sdl2.video.Renderer`.
- Render: precalcular todo lo posible, arrays preasignados y ningún bucle de Python por píxel. numba solo si numpy no basta. En Windows, crear un array temporal de ~1,6 MB cuesta ~1,7 ms (fallos de página) frente a ~0,26 ms operando con `out=` sobre uno preasignado.
- Entrenamiento: red MLP pequeña, hilos de PyTorch limitados y prioridad baja del proceso y sus subprocesos.

## Render pseudo-3D (`render/`)

- El mundo se dibuja a 320×180 en índices de la paleta Resurrect 64 (`render/palette.py`, nombres por uso en `Ink`). Nada sale de los 64 colores, salvo los halos aditivos de los fantasmas.
- Atardecer y noche son reasignaciones de índices (`Palette.remap`); la niebla son tablas índice a índice por nivel que ya incluyen la variante.
- El suelo (`render/floor.py`) se calcula con tablas precalculadas por cámara (`Lens`) y cuatro operaciones por fotograma más un `take` por grupo de mipmap. Sin bucles de Python por píxel y con buffers preasignados.
- Textura y decorado se pintan una vez al cargar el circuito (`texture.py`, `scene.place_props`), sombras incluidas. Las marcas de derrape se estampan en la textura.
- Los sprites se generan por código al tamaño en que se ven (sprite stacking para los karts) y se cachean, con un presupuesto de 3 ms por fotograma (`SpriteBank`). No reescalar pixel art.
- La interfaz va a resolución de ventana con textos en caché (`TextCache`). Mide con F3 o con `scripts/benchmark_render.py` antes de optimizar.

## Vídeos (`show/video.py`)

- Siempre 1920×1080 a 60 fps exactos, enviados a ffmpeg con `imageio_ffmpeg.write_frames` (sin fotogramas en disco). Un paso de física por fotograma y acciones deterministas.
- Fuera de tiempo real, el reloj de `GameView` se fija con `fixed_time` (cielo, entrada del HUD y avisos); sin eso dependería del reloj de pared.
- Los vídeos van a `videos/` (ignorado por git). Un nuevo tipo de vídeo reutiliza `Compositor`, `PolicyDriver` y `VideoWriter`.

## Visor de exhibición (`show/viewer.py`)

- `ViewerApp` es dueña de la ventana, el `GameView`, el reloj y las teclas comunes; cada modo es una `Scene` (`scenes.py`: `WatchScene`, `RaceScene`, `ProgressionScene`, y `AttractScene` tras el título). Un modo nuevo es una clase más en `SCENES`.
- Nada bloquea el fotograma: `VersionLibrary` carga checkpoints en un hilo y los cachea (en modo síncrono, el de `--frames` y los tests, carga al pedirlos); el circuito nuevo se construye tras mostrar «Cargando…» (`ViewerApp.defer`). Los cambios de escena pasan por `ViewerApp.fade`.
- La física avanza con un acumulador a pasos de 1/60 s y el factor de pausa y cámara lenta (`SPEEDS`); sin ventana, un fotograma es un paso. Los paneles (`Overlays`) los comparten el visor y los vídeos.
- Teclas: coherentes con `drive.py` (C, V, G, P, L, F3, F11, H, 1-3, E, N, R, Esc). Si añades una, anótala en `HELP_SECTIONS` y en la tabla del README.
- Vista cerebro (tecla B, `show/brain.py`; también en `live.py`): `BrainPanel` solo dibuja lo que guarda `PolicyDriver` (`last_obs`, `action`, `values`). V(s) sale de `policy.predict_values` (API pública de SB3) y solo se calcula mientras el panel está abierto (`set_tracking`); se reescala con `ret_rms` a unidades de recompensa. El panel pinta una capa base una vez y cuesta unas décimas de milisegundo; sin él, cero.

## Windows

- Los entornos en paralelo arrancan con spawn: los scripts necesitan `if __name__ == "__main__"` y nada pesado a nivel de módulo.
- Ficheros de texto siempre con `encoding="utf-8"` explícito (ruff `PLW1514` lo vigila).
- El visor declara el proceso DPI-aware antes de crear la ventana.
- Los mensajes de commit con tildes se pasan por Bash con heredoc: PowerShell 5.1 los estropea.
- `Path.write_text` convierte `\n` en CRLF en Windows: usa `newline="\n"` al reescribir ficheros del repo.

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
