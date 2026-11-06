"""The interface font: a 5x7 bitmap, upper case, with two extra rows above for accents.

Every glyph is a grid of ``#`` and ``.``. Text is measured and wrapped here, with no
pygame and no vector font involved, so the layout of a screen is exact: the same string
always takes the same number of logical pixels, at any scale.

- A glyph is 7 rows tall plus 2 rows above for the accents of Á É Í Ó Ú Ñ Ü (``GLYPH_H`` = 9).
  ``CAP_TOP`` is where the capital starts, and text is placed by the top of its capitals.
- One empty column separates two glyphs (multiplied by the scale, like everything else).
- Lower case is drawn as upper case: the font has one case, as the 16-bit consoles had.
- ``tabular`` gives every digit the same width, so a ticking clock does not jitter.
"""

from __future__ import annotations

from functools import cache

import numpy as np

CAP_H = 7  # rows of a capital letter
CAP_TOP = 2  # rows above the capital, for the accents
GLYPH_H = CAP_H + CAP_TOP
DIGIT_W = 5  # cell of a tabular digit
SPACE_W = 3
LINE = GLYPH_H  # line pitch of wrapped text: capital, then room for the next line's accents

# Glyph grids, 7 rows each (the accent rows are added below).
_G: dict[str, tuple[str, ...]] = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "Ç": (".###.", "#...#", "#....", "#....", "#...#", ".###.", "..#.."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": ("###", ".#.", ".#.", ".#.", ".#.", ".#.", "###"),
    "J": ("..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": (".#.", "##.", ".#.", ".#.", ".#.", ".#.", "###"),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("####.", "....#", "....#", ".###.", "....#", "....#", "####."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": (".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."),
    " ": ("...",) * 7,
    ".": (".", ".", ".", ".", ".", ".", "#"),
    ",": ("..", "..", "..", "..", ".#", ".#", "#."),
    ":": (".", "#", ".", ".", ".", "#", "."),
    ";": ("..", ".#", "..", "..", ".#", ".#", "#."),
    "!": ("#", "#", "#", "#", "#", ".", "#"),
    "¡": ("#", ".", "#", "#", "#", "#", "#"),
    "?": (".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."),
    "¿": ("..#..", ".....", "..#..", ".#...", "#....", "#...#", ".###."),
    "%": ("##..#", "##..#", "...#.", "..#..", ".#...", "#..##", "#..##"),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "-": ("...", "...", "...", "###", "...", "...", "..."),
    "±": ("..#..", "..#..", "#####", "..#..", "..#..", ".....", "#####"),
    "/": ("....#", "...#.", "...#.", "..#..", ".#...", ".#...", "#...."),
    "\\": ("#....", ".#...", ".#...", "..#..", "...#.", "...#.", "....#"),
    "(": (".#", "#.", "#.", "#.", "#.", "#.", ".#"),
    ")": ("#.", ".#", ".#", ".#", ".#", ".#", "#."),
    "[": ("##", "#.", "#.", "#.", "#.", "#.", "##"),
    "]": ("##", ".#", ".#", ".#", ".#", ".#", "##"),
    "{": (".##", ".#.", ".#.", "#..", ".#.", ".#.", ".##"),
    "}": ("##.", ".#.", ".#.", "..#", ".#.", ".#.", "##."),
    "<": ("...#", "..#.", ".#..", "#...", ".#..", "..#.", "...#"),
    ">": ("#...", ".#..", "..#.", "...#", "..#.", ".#..", "#..."),
    "=": (".....", ".....", "#####", ".....", "#####", ".....", "....."),
    "$": ("..#..", ".####", "#.#..", ".###.", "..#.#", "####.", "..#.."),
    "*": (".....", "#.#.#", ".###.", "#####", ".###.", "#.#.#", "....."),
    "#": (".#.#.", "#####", ".#.#.", ".#.#.", ".#.#.", "#####", ".#.#."),
    "&": (".##..", "#..#.", "#.#..", ".#...", "#.#.#", "#..#.", ".##.#"),
    "@": (".###.", "#...#", "#.###", "#.#.#", "#.###", "#....", ".###."),
    "|": ("#",) * 7,
    "_": (".....",) * 6 + ("#####",),
    "^": ("..#..", ".#.#.", "#...#", ".....", ".....", ".....", "....."),
    "~": (".....", ".....", ".##.#", "#.##.", ".....", ".....", "....."),
    "'": ("#", "#", ".", ".", ".", ".", "."),
    '"': ("#.#", "#.#", "...", "...", "...", "...", "..."),
    "`": ("#.", ".#", "..", "..", "..", "..", ".."),
    "×": (".....", ".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#"),
    "·": (".", ".", ".", "#", ".", ".", "."),
    "°": (".#.", "#.#", ".#.", "...", "...", "...", "..."),
    "º": (".#.", "#.#", ".#.", "###", "...", "...", "..."),
    "ª": (".##", "#.#", ".##", "###", "...", "...", "..."),
    "«": (".....", "..#.#", ".#.#.", "#.#..", ".#.#.", "..#.#", "....."),
    "»": (".....", "#.#..", ".#.#.", "..#.#", ".#.#.", "#.#..", "....."),
    "…": (".....",) * 6 + ("#.#.#",),
    "←": (".....", "..#..", ".#...", "#####", ".#...", "..#..", "....."),
    "→": (".....", "..#..", "...#.", "#####", "...#.", "..#..", "....."),
    "↑": ("..#..", ".###.", "#.#.#", "..#..", "..#..", "..#..", "....."),
    "↓": (".....", "..#..", "..#..", "..#..", "#.#.#", ".###.", "..#.."),
    "✓": (".....", "....#", "....#", "...#.", "#.#..", ".#...", "....."),
    "✗": (".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "....."),
    "■": (".....", ".....", ".###.", ".###.", ".###.", ".....", "....."),
    "▸": ("....", "#...", "##..", "###.", "##..", "#...", "...."),
}

# Characters drawn as another one (the proper minus sign, dashes).
_ALIAS = {"−": "-", "–": "-", "—": "-", "‑": "-"}

_ACUTE = ("...#.", "..#..")
_TILDE = (".##.#", "#.##.")
_GRAVE = (".#...", "..#..")
_ACCENTED: dict[str, tuple[str, tuple[str, str]]] = {
    "À": ("A", _GRAVE),
    "È": ("E", _GRAVE),
    "Ì": ("I", ("#..", ".#.")),
    "Ò": ("O", _GRAVE),
    "Ù": ("U", _GRAVE),
    "Ï": ("I", ("#.#", "...")),
    "Á": ("A", _ACUTE),
    "É": ("E", _ACUTE),
    "Í": ("I", ("..#", ".#.")),
    "Ó": ("O", _ACUTE),
    "Ú": ("U", _ACUTE),
    "Ñ": ("N", _TILDE),
    "Ü": ("U", (".#.#.", ".....")),
}

FALLBACK = "?"


def has_glyph(ch: str) -> bool:
    """True if ``ch`` (in either case) has its own glyph: the tests use it to check that
    no string of the interface falls back to the question mark."""
    ch = ch.upper()
    return ch in _G or ch in _ACCENTED or ch in _ALIAS


def known_characters() -> str:
    return "".join(sorted({*_G, *_ACCENTED, *_ALIAS}))


@cache
def glyph_rows(ch: str) -> tuple[str, ...]:
    """The 9 rows of a character: two for the accent, then the 7 of the glyph."""
    ch = _ALIAS.get(ch, ch)
    accent = _ACCENTED.get(ch)
    if accent is not None:
        base = _G[accent[0]]
        width = len(base[0])
        marks = tuple(row.ljust(width, ".")[:width] for row in accent[1])
        return (*marks, *base)
    rows = _G.get(ch)
    if rows is None:
        rows = _G[FALLBACK]
    blank = "." * len(rows[0])
    return (blank, blank, *rows)


@cache
def glyph_array(ch: str, tabular: bool = False) -> np.ndarray:
    """The glyph as a (GLYPH_H, width) boolean array; a tabular digit is centred in its cell."""
    rows = glyph_rows(ch)
    array = np.array([[c == "#" for c in row] for row in rows], dtype=bool)
    if tabular and ch.isdigit():
        left = (DIGIT_W - array.shape[1]) // 2
        padded = np.zeros((GLYPH_H, DIGIT_W), dtype=bool)
        padded[:, left : left + array.shape[1]] = array
        return padded
    return array


@cache
def measure(text: str, scale: int = 1, tabular: bool = False) -> int:
    """Width in logical pixels of ``text`` at integer ``scale``."""
    if not text:
        return 0
    total = 0
    for ch in text.upper():
        total += glyph_array(ch, tabular).shape[1] + 1
    return (total - 1) * scale


def text_mask(text: str, scale: int = 1, tabular: bool = False) -> np.ndarray:
    """The text as a (GLYPH_H * scale, width) boolean array."""
    if not text:
        return np.zeros((GLYPH_H * scale, 0), dtype=bool)
    gap = np.zeros((GLYPH_H, 1), dtype=bool)
    parts: list[np.ndarray] = []
    for ch in text.upper():
        parts.append(glyph_array(ch, tabular))
        parts.append(gap)
    mask = np.concatenate(parts[:-1], axis=1)
    if scale > 1:
        mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
    return mask


def wrap(text: str, max_width: int, scale: int = 1) -> list[str]:
    """Greedy word wrap into lines of at most ``max_width`` pixels.

    ``\\n`` forces a break. A word wider than the line (a path has no spaces) is cut.
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            while measure(word, scale) > max_width and len(word) > 1:
                cut = len(word) - 1
                while cut > 1 and measure(word[:cut], scale) > max_width:
                    cut -= 1
                if line:
                    lines.append(line)
                    line = ""
                lines.append(word[:cut])
                word = word[cut:]
            trial = f"{line} {word}" if line else word
            if line and measure(trial, scale) > max_width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    while len(lines) > 1 and not lines[-1]:
        lines.pop()
    return lines


def wrap_clip(text: str, max_width: int, max_lines: int, scale: int = 1) -> list[str]:
    """``wrap`` into at most ``max_lines`` lines; what does not fit ends the last one in an
    ellipsis (so a name is split in two or three lines before anything is cut off)."""
    lines = wrap(text, max_width, scale)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = ellipsize(lines[-1] + " …", max_width, scale)
    return lines


def ellipsize(text: str, max_width: int, scale: int = 1, keep: str = "start") -> str:
    """``text`` shortened with an ellipsis to fit ``max_width``; ``keep`` says which end
    stays: ``start`` for names, ``end`` for paths (the file name matters most)."""
    if measure(text, scale) <= max_width:
        return text
    while text and measure("…" + text if keep == "end" else text + "…", scale) > max_width:
        text = text[1:] if keep == "end" else text[:-1]
    return "…" + text if keep == "end" else text + "…"
