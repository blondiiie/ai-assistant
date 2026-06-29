from __future__ import annotations

import re

# Единый «алфавит» допустимых письменностей — источник правды и для WORD_RE
# (что считать словом), и для проверки посторонних письменностей. Менять —
# только здесь, чтобы два механизма не разошлись.
ALLOWED_ALPHABET = "0-9a-zA-Zа-яА-ЯёЁ"

CITED_RE = re.compile(r"c(\d+)")
WORD_RE = re.compile(rf"[{ALLOWED_ALPHABET}]{{2,}}")

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
CITE_TAG_RE = re.compile(r"\[c\d+\]")
_STEM_LEN = 6


# --- Этап 2.1: единый скоринговый grounding-гейт ---
# Заменяет AND-каскад (3-граммы AND word_coverage AND distinctive_tokens),
# где вероятности ложного отсева перемножались. Теперь — один score ∈ [0,1]
# = среднее из трёх компонент. Принимаем при score >= grounding_threshold.


def _word_coverage_ratio(answer: str, cited_contents: list[str]) -> float:
    """Доля содержательных слов ответа (со стеммингом) в контексте ∈ [0,1]."""
    if not cited_contents:
        return 0.0
    answer_stems = _content_stems(CITE_TAG_RE.sub(" ", answer))
    if not answer_stems:
        return 0.0
    ctx_stems: set[str] = set()
    for content in cited_contents:
        ctx_stems |= _content_stems(content)
    if not ctx_stems:
        return 0.0
    return len(answer_stems & ctx_stems) / len(answer_stems)


def _trigram_overlap_ratio(answer: str, cited_contents: list[str]) -> float:
    """Доля 3-грамм ответа, покрытых контекстом ∈ [0,1]."""
    if not cited_contents:
        return 0.0
    answer_grams = _char_ngrams(answer)
    if not answer_grams:
        return 0.0
    cited_grams: set[str] = set()
    for content in cited_contents:
        cited_grams |= _char_ngrams(content)
    return len(answer_grams & cited_grams) / len(answer_grams)


def _distinctive_tokens_ratio(answer: str, cited_contents: list[str]) -> float:
    """Доля чисел (2+ цифр) ответа, подтверждённых контекстом ∈ [0,1].

    Если чисел нет — 1.0 (нейтрально: проверка не применима). Если есть
    выдуманные числа — падает пропорционально их доле.
    """
    if not cited_contents:
        return 1.0
    answer_clean = CITE_TAG_RE.sub(" ", answer)
    nums = set(_NUM_RE.findall(answer_clean))
    if not nums:
        return 1.0
    ctx_lower = " \n".join(c.lower() for c in cited_contents)
    confirmed = {n for n in nums if n in ctx_lower}
    return len(confirmed) / len(nums)


def grounding_components(
    answer: str,
    cited_contents: list[str],
) -> dict[str, float]:
    """Компоненты единого grounding_score для логов/диагностики (Этап 3.1)."""
    return {
        "word_coverage": _word_coverage_ratio(answer, cited_contents),
        "trigram_overlap": _trigram_overlap_ratio(answer, cited_contents),
        "distinctive_tokens": _distinctive_tokens_ratio(answer, cited_contents),
    }


def grounding_score(answer: str, cited_contents: list[str]) -> float:
    """Единый grounding_score ∈ [0,1] — среднее из трёх компонент (Этап 2.1).

    Заменяет AND-каскад из пяти гейтов (вероятности перемножались → ложные
    заглушки). Жёсткий отсев has_foreign_script остаётся отдельным сейфом.
    Принимаем при score >= grounding_threshold (дефолт 0.5, настраивается).
    """
    comps = grounding_components(answer, cited_contents)
    return sum(comps.values()) / len(comps)

# Разрешённые «несмысловые» связки/маркеры списков — не режутся фильтром.
_CONNECTOR_RE = re.compile(
    r"^(?:таким\s+образом|итак|следовательно|в\s+итоге|то\s+есть|"
    r"т\.е\.|например|далее|также|кроме\s+того|однако|но|и|или)$",
    re.IGNORECASE,
)

# Запрещённый символ = любой вне алфавита + разрешённой пунктуации/пробелов.
# Алфавит берётся из ALLOWED_ALPHABET, чтобы не разойтись с WORD_RE.
# Этап 5.2: расширено технической пунктуацией, которая легитимно встречается
# в заметках (JSON/YAML/HTML/HTTP-примеры, шаблоны) и НЕ является письменностью.
# Гейт по-прежнему ловит утечку чужих письменностей (китайские/арабские/тайские
# glyph'ы и т.п.). Смысловая анти-галлюцинация держится на word_coverage /
# trigram_overlap / distinctive_tokens, а не на этом символьном фильтре.
_ALLOWED_CHARS_RE = re.compile(
    rf"[^{ALLOWED_ALPHABET}\s.,:;\-!?()/\[\]\"'«»—–\n"
    rf"{{}}<>%=+*|&@#$%^~`_\\"
    # Этап 4.1: типографские символы русскоязычных документов, которые модель
    # правомерно воспроизводит из контекста (НЕ письменность):
    rf"\u2116\u2026\u2018\u2019\u201a\u201b\u00b0\u00b1\u00d7"
    rf"\u2192\u2190\u2194\u00b7\u2264\u2265\u2260\u2022]"
)


def _stem_token(word: str) -> str:
    """Лёгкий стемминг: префикс фиксированной длины — терпимость к морфологии
    RU/EN («сервер»/«сервера»/«упал»/«упало» совпадают по префиксу)."""
    word = word.lower()
    return word[:_STEM_LEN] if len(word) > _STEM_LEN else word


def _content_stems(text: str) -> set[str]:
    return {_stem_token(t) for t in WORD_RE.findall(text) if t.lower() not in _STOPWORDS}


def is_word_supported(
    answer: str,
    cited_contents: list[str],
    min_coverage: float,
) -> bool:
    """Словарная верность ответа контексту (по всему ответу, со стеммингом).

    Доля содержательных слов ответа, присутствующих в контексте, должна быть
    >= min_coverage. Ловит дрейф в свои знания: напр. источник «500 — Сервер
    упал/выбросил исключение», а ответ «сервер столкнулся с необработанной
    ошибкой и не может выполнить запрос клиента» — много новой лексики →
    низкое покрытие (~0.38) → провал. Связный верный ответ держит покрытие
    ~0.7+, короткие связки/переводы усредняются и не рубят ответ.
    """
    if not cited_contents:
        return False
    answer_stems = _content_stems(CITE_TAG_RE.sub(" ", answer))
    if not answer_stems:
        return False
    ctx_stems: set[str] = set()
    for content in cited_contents:
        ctx_stems |= _content_stems(content)
    if not ctx_stems:
        return False
    covered = answer_stems & ctx_stems
    return len(covered) / len(answer_stems) >= min_coverage


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
    answer = CITE_TAG_RE.sub(" ", answer)
    ctx_lower = " \n".join(c.lower() for c in cited_contents)
    nums = set(_NUM_RE.findall(answer))
    return {n for n in nums if n not in ctx_lower}


# --- Посентенсная строгая проверка (точечное вырезание галлюцинаций) ---

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def has_foreign_script(text: str) -> bool:
    """True, если в тексте есть символы неразрешённых письменностей.

    Ловит утечку токенов модели в чужую письменность (китайские иероглифы,
    арабица и т.п.) — основной симптом деградации ответа. Кириллица, латиница,
    цифры, базовая пунктуация и технические символы ({}, <>, =, *, +, |, @,
    #, $, %, ^, ~, `, _, \\) считаются разрешёнными (Этап 5.2: они легитимны
    в технических заметках — JSON/YAML/HTML/HTTP — и не являются письменностью).
    """
    return bool(_ALLOWED_CHARS_RE.search(CITE_TAG_RE.sub("", text)))


_LIST_MARKER_RE = re.compile(r"^\d+[.)]?$|^[-*•]$")


def split_sentences(text: str) -> list[str]:
    """Разбивает ответ на предложения/пункты списка, сохраняя структуру.

    Раздел — по переводам строк и по завершающей пунктуации (.!?). Маркеры
    нумерованных/маркированных списков («1.», «-») приклеиваются к следующему
    за ними содержимому. Пустые фрагменты отбрасываются.
    """
    raw: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for chunk in _SENT_SPLIT_RE.split(line):
            chunk = chunk.strip()
            if chunk:
                raw.append(chunk)
    parts: list[str] = []
    i = 0
    while i < len(raw):
        frag = raw[i]
        if _LIST_MARKER_RE.match(frag) and i + 1 < len(raw):
            parts.append(f"{frag} {raw[i + 1]}")
            i += 2
        else:
            parts.append(frag)
            i += 1
    return [p for p in parts if p.strip()]


def _is_connector(sentence: str) -> bool:
    stripped = CITE_TAG_RE.sub("", sentence).strip(" .,:;-—").lower()
    return bool(stripped) and bool(_CONNECTOR_RE.match(stripped))


def _sentence_supported(
    sentence: str,
    ctx_stems: set[str],
    ctx_lower: str,
    *,
    min_coverage: float,
) -> bool:
    if _is_connector(sentence):
        return True
    sent_clean = CITE_TAG_RE.sub(" ", sentence)
    # числа 2+ цифр обязаны быть в контексте — высокоточный сигнал галлюцинации;
    # проверяем ДО раннего возврата, иначе «тощие» предложения из одних цифр
    # обходили бы числовую проверку.
    nums = _NUM_RE.findall(sent_clean)
    if nums and not all(n in ctx_lower for n in nums):
        return False
    stems = _content_stems(sent_clean)
    if not stems:
        return True
    covered = stems & ctx_stems
    return len(covered) / len(stems) >= min_coverage


def _mark_unsupported(
    sentences: list[str],
    cited_contents: list[str],
    *,
    min_coverage: float,
) -> list[tuple[int, str]]:
    """Внутренняя: индексы+причины неподдержанных предложений (без повторного
    разбиения — предложения передаются готовыми)."""
    if not cited_contents:
        return [(i, "coverage") for i in range(len(sentences))]
    ctx_stems: set[str] = set()
    for c in cited_contents:
        ctx_stems |= _content_stems(c)
    ctx_lower = " \n".join(c.lower() for c in cited_contents)

    result: list[tuple[int, str]] = []
    for i, sent in enumerate(sentences):
        if has_foreign_script(sent):
            result.append((i, "foreign"))
            continue
        if not _sentence_supported(sent, ctx_stems, ctx_lower, min_coverage=min_coverage):
            result.append((i, "coverage"))
    return result


def unsupported_sentences(
    answer: str,
    cited_contents: list[str],
    *,
    min_coverage: float,
) -> list[tuple[int, str]]:
    """Для каждого предложения флага: поддержано ли оно контекстом.

    Возвращает список (индекс, причина) неподдержанных предложений. Причины:
    'foreign' — посторонняя письменность; 'coverage' — мало слов из контекста
    либо выдуманные числа (числовая проверка входит в покрытие).
    """
    return _mark_unsupported(
        split_sentences(answer), cited_contents, min_coverage=min_coverage
    )


def filter_answer(
    answer: str,
    cited_contents: list[str],
    *,
    min_coverage: float,
    min_kept_ratio: float,
    min_kept_sentences: int,
) -> str | None:
    """Вырезает неподдержанные предложения. None = недостаточно смысла → заглушка.

    Сохраняет порядок и нумерацию списков; убирает лишь галлюцинирующие фразы
    (вкл. иероглифы, выдуманные термины, нерелевантные куски).
    """
    sentences = split_sentences(answer)
    if not sentences:
        return None
    bad = {i for i, _ in _mark_unsupported(sentences, cited_contents, min_coverage=min_coverage)}
    kept = [s for i, s in enumerate(sentences) if i not in bad]
    if not kept:
        return None
    content_kept = [s for s in kept if not _is_connector(s)]
    content_orig = [s for s in sentences if not _is_connector(s)]
    original_chars = sum(len(s) for s in sentences)
    kept_chars = sum(len(s) for s in kept)
    # Недостаточно смысла: вырезано почти всё ИЛИ остались только связки.
    # Граница min_kept_sentences применяется только к развёрнутым ответам,
    # чтобы короткие односложные grounded-ответы не превращались в заглушку.
    if not content_kept:
        return None
    if original_chars > 0 and kept_chars / original_chars < min_kept_ratio:
        return None
    if len(content_orig) >= 4 and len(content_kept) < min_kept_sentences:
        return None
    return "\n".join(kept)
