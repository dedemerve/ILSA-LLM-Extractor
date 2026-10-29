"""
Source-text preparation shared by SFT data building and inference (v2).

Training and inference must see byte-identical passages and prompts; both
build_sft_v2.py and extract_v2.py import these functions instead of
re-implementing them (the Stage 7/9 run truncated sources at different points
during training and inference).
"""

from __future__ import annotations

import re
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "prompts" / "v2"

CHUNK_CHARS = 6000        # about 1.5k Mistral/Qwen tokens of academic English
CHUNK_OVERLAP = 600
METHODS_MAX_CHARS = 12000
MIN_CHUNK_CHARS = 400

_REFERENCES_HEADING = re.compile(
    r"^\s*(references|reference list|bibliography|literatur|referencias)\s*$", re.I | re.M)
_METHODS_KEYWORDS = re.compile(
    r"plausible value|\bPVs?\b|PV1|rubin|sampling weight|student weight|final weight|"
    r"senate weight|TOTWGT|W_FSTUWT|SPFWT|TCHWGT|replicate weight|jackknife|\bBRR\b|"
    r"\bweight(?:s|ed|ing)?\b", re.I)


def clean_source(text: str) -> str:
    """Normalize whitespace and drop the reference list (the last References heading onward)."""
    text = text.replace("\r\n", "\n").replace("­", "")
    text = re.sub(r"[ \t]+", " ", text)
    matches = list(_REFERENCES_HEADING.finditer(text))
    if matches and matches[-1].start() > 0.5 * len(text):
        text = text[:matches[-1].start()]
    # PDF extraction wraps lines mid-sentence: undo hyphenation, then join single line breaks
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, str]]:
    """Split into overlapping windows, preferring paragraph then sentence boundaries.
    Returns (start_offset, chunk) tuples; offsets index into `text`."""
    chunks: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = window.rfind("\n\n", int(size * 0.6))
            if cut == -1:
                cut = max(window.rfind(". ", int(size * 0.6)), -1)
                cut = cut + 1 if cut != -1 else -1
            if cut != -1:
                end = start + cut
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS or not chunks:
            chunks.append((start, piece))
        elif chunks:
            prev_start, prev = chunks[-1]
            chunks[-1] = (prev_start, (prev + "\n\n" + piece).strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def methods_passages(chunks: list[tuple[int, str]], max_chars: int = METHODS_MAX_CHARS) -> str:
    """Concatenate the chunks with the densest PV / weighting vocabulary, in document order."""
    scored = [(len(_METHODS_KEYWORDS.findall(c)), k) for k, (_, c) in enumerate(chunks)]
    picked: list[int] = []
    total = 0
    for hits, k in sorted(scored, key=lambda x: (-x[0], x[1])):
        if hits == 0 and picked:
            break
        if total + len(chunks[k][1]) > max_chars and picked:
            continue
        picked.append(k)
        total += len(chunks[k][1])
    return "\n\n[...]\n\n".join(chunks[k][1] for k in sorted(picked))


def _template(name: str) -> Template:
    return Template((PROMPT_DIR / name).read_text(encoding="utf-8"))


def render_chunk_prompt(title: str, program: str, cycle: str, index: int, n: int, passage: str) -> str:
    return _template("chunk_extraction.txt").substitute(
        title=title, program=program, cycle=cycle, chunk_index=index, n_chunks=n, passage=passage)


def render_methods_prompt(title: str, program: str, cycle: str, passage: str) -> str:
    return _template("methods_extraction.txt").substitute(
        title=title, program=program, cycle=cycle, passage=passage)


def user_messages(prompt: str) -> list[dict[str, str]]:
    """Instructions travel in the user turn. A separate system turn is avoided because the
    Mistral v0.3 chat template renders it only when the user turn is the last message,
    i.e. never in a training conversation."""
    return [{"role": "user", "content": prompt}]
