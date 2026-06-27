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


_REFUSAL_FORMS = (
    "noanswer",
    "невозможноответить",
    "немогуответить",
    "нетинформации",
    "недостаточноданных",
)


def _normalize_refusal(text: str) -> str:
    s = re.sub(r"[^0-9a-zа-яё]+", "", text.lower())
    return s.replace("ё", "е")


def is_refusal(raw: str) -> bool:
    if not raw:
        return False
    norm = _normalize_refusal(raw)
    return any(norm.startswith(form) for form in _REFUSAL_FORMS)


def _content_tokens(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text) if t.lower() not in _STOPWORDS]


def _token_set(text: str) -> set[str]:
    return set(_content_tokens(text))


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    s = re.sub(r"[^0-9a-zа-яё]+", "", text.lower())
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


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
    answer_grams = _char_ngrams(answer)
    if not answer_grams:
        return False
    cited_grams: set[str] = set()
    for content in cited_contents:
        cited_grams |= _char_ngrams(content)
    overlap = len(answer_grams & cited_grams) / len(answer_grams)
    return overlap >= min_overlap


_NUM_RE = re.compile(r"\d{2,}")
_CITE_TAG_RE = re.compile(r"\[c\d+\]")


def missing_distinctive_tokens(
    answer: str,
    cited_contents: list[str],
) -> set[str]:
    """Числа (2+ цифр) ответа, отсутствующие ни в одном чанке контекста.

    Числа — высокоточный сигнал галлюцинации: выдуманные HTTP-коды (200/400),
    статусы, статистика. Латинские термины намеренно НЕ проверяются: модель
    может подставлять факт. верные расшифровки аббревиатур, что даёт ложные
    срабатывания; для них достаточно 3-граммового overlap и правил промпта.
    Возвращает числа, которых нет в контексте (пусто = все подтверждены).
    """
    if not cited_contents:
        return set()
    answer = _CITE_TAG_RE.sub(" ", answer)
    ctx_lower = " \n".join(c.lower() for c in cited_contents)
    nums = set(_NUM_RE.findall(answer))
    return {n for n in nums if n not in ctx_lower}
