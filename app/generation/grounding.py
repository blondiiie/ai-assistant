from __future__ import annotations

import re

CITED_RE = re.compile(r"c(\d+)")
WORD_RE = re.compile(r"[0-9a-zа-яё]{2,}", re.IGNORECASE)

_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "о", "из",
    "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь",
    "там", "потом", "себя", "ничего", "может", "тут", "где", "есть", "надо",
    "для", "это", "этот", "эта", "эти", "we", "the", "and", "for", "with",
    "that", "this", "are", "is",
}


def parse_cited_ids(answer: str) -> list[int]:
    return [int(m) for m in CITED_RE.findall(answer)]


def _content_tokens(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text) if t.lower() not in _STOPWORDS]


def _token_set(text: str) -> set[str]:
    return set(_content_tokens(text))


def check(answer: str, context_chunk_ids: set[int]) -> tuple[list[int], bool, set[int]]:
    cited = parse_cited_ids(answer)
    valid_ordered = [cid for cid in cited if cid in context_chunk_ids]
    valid_unique = list(dict.fromkeys(valid_ordered))
    invalid = {cid for cid in cited if cid not in context_chunk_ids}
    return valid_unique, bool(valid_unique), invalid


def is_supported(
    answer: str,
    cited_contents: list[str],
    min_overlap: float,
) -> bool:
    if not cited_contents:
        return False
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return False
    answer_set = set(answer_tokens)
    cited_set: set[str] = set()
    for content in cited_contents:
        cited_set.update(_content_tokens(content))
    overlap = len(answer_set & cited_set) / len(answer_set)
    return overlap >= min_overlap
