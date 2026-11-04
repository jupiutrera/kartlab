# Procedencia de recursos

Todo el material gráfico, tipográfico y sonoro del proyecto es original o tiene licencia libre. Cada recurso que entra en el repositorio se anota aquí.

## Reglas

- Nada de personajes, sprites, circuitos, nombres, logotipos, música ni sonidos de Nintendo ni de otros juegos comerciales.
- Gráficos generados por código o con licencia CC0.
- Fuentes con licencia SIL Open Font License (OFL). El texto de la licencia se guarda junto a la fuente en `assets/`.

## Recursos

| Recurso | Tipo | Fichero | Autor | Licencia | Origen | Añadido |
| --- | --- | --- | --- | --- | --- | --- |
| IBM Plex Mono (Regular y SemiBold) | Fuente de la interfaz de la vista de depuración | `assets/fonts/ibm-plex-mono/` | IBM Corp. | SIL OFL 1.1 (`OFL.txt` junto a la fuente) | [google/fonts](https://github.com/google/fonts/tree/main/ofl/ibmplexmono) | 2026-11-04 |

## Material propio

- **Circuitos escaparate** (Puerto, Cañón y Meseta): trazados diseñados para este proyecto en `src/kartlab/sim/showcase.py`.
- **Circuitos aleatorios y de evaluación**: generados por código a partir de una semilla (`src/kartlab/sim/track.py`).
- **Vista cenital de depuración**: colores, pianos, línea de meta y flechas dibujados por código (`src/kartlab/render/topdown.py`). La paleta definitiva del juego se elige en la fase 4.
