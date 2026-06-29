from __future__ import annotations

import asyncio

import pytest

from app.config import settings
from app.generation import service as gen_service
from app.generation.grounding import (
    filter_answer,
    has_foreign_script,
    split_sentences,
    unsupported_sentences,
)
from app.retrieval.service import _fit_budget
from app.schemas import ChunkResult

REST_NOTE = (
    "REST (Representational State Transfer) — парадигма проектирования API, "
    "не протокол. Принципы: 1. Клиент-серверная архитектура. 2. Stateless — "
    "сервер не хранит сессию. 3. Кэширование. 4. Единообразие интерфейса. "
    "5. Layered System. 6. Code on Demand (необязательно) — сервер может "
    "передать код клиенту. Уровни зрелости: Уровень 0, Уровень 1, Уровень 2, "
    "Уровень 3. REST не всегда требует HTTP или JSON."
)


@pytest.fixture
def rest_chunk() -> ChunkResult:
    return ChunkResult(
        chunk_id=1,
        content=REST_NOTE,
        page=None,
        section=None,
        source_name="Obsidian Vault/Resurses/REST/REST API.md",
        score=0.8,
    )


class _FakeOllama:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def chat(  # noqa: ANN001
        self,
        messages,
        *,
        temperature=0.0,
        options=None,
        seed=None,
        top_p=None,
        keep_alive=None,
    ):
        return self._reply


# --- has_foreign_script: ловит утечку иероглифов/арабицы ---

def test_has_foreign_script_detects_chinese() -> None:
    assert has_foreign_script("Layered System (层次系统)") is True


def test_has_foreign_script_detects_arabic() -> None:
    assert has_foreign_script("Ответ: مرحبا тут") is True


def test_has_foreign_script_accepts_ru_en_punct() -> None:
    assert has_foreign_script("REST — парадигма. Code on Demand (необязательно).") is False
    assert has_foreign_script("Levels: 0, 1, 2, 3 [c1]") is False


# --- Этап 5.4: регрессионные тесты фикса бага JSON/HATEOAS ---

def test_has_foreign_script_accepts_technical_punctuation() -> None:
    """Этап 5.2: техническая пунктуация из JSON/YAML/HTML/HTTP не триггерит гейт."""
    # JSON: {} [] и кавычки
    assert has_foreign_script('Пример: {"key": "value"} и массив [] [c1]') is False
    # HTML-like / шаблоны: < > =
    assert has_foreign_script("Тег <div class=\"x\"> или a = b [c1]") is False
    # HTTP / конфиги / markdown: * + | @ # $ % ^ ~ ` _
    http_syntax = (
        "Headers: X-Api*key +token | pipe @at #tag $var %pct ^caret ~tilde `code` _under [c1]"
    )
    assert has_foreign_script(http_syntax) is False


def test_has_foreign_script_still_catches_glyphs() -> None:
    """Этап 5.4: защита от регрессии — чужие письменности по-прежнему ловятся."""
    # Китайские иероглифы
    assert has_foreign_script("REST — парадигма. 客户端 [c1]") is True
    # Арабица
    assert has_foreign_script("Принцип: واجهة единообразная [c1]") is True
    # Японские каны
    assert has_foreign_script("Уровень: クライアント [c1]") is True
    # Тайские glyph'ы
    assert has_foreign_script("Пример: แม่แบบ [c1]") is True


# --- Посентенсная фильтрация ---

def test_filter_answer_removes_hallucinated_sentence() -> None:
    answer = (
        "REST — парадигма проектирования API.\n"
        "Code on done рекомендуется для гибкости системы.\n"
        "Stateless — сервер не хранит сессию."
    )
    refined = filter_answer(
        answer, [REST_NOTE], min_coverage=0.55, min_kept_ratio=0.25, min_kept_sentences=2
    )
    assert refined is not None
    assert "Code on done" not in refined
    assert "Stateless" in refined


def test_filter_answer_removes_foreign_sentence() -> None:
    answer = (
        "Layered System (层次系统).\n"
        "REST — парадигма проектирования API.\n"
        "Stateless — сервер не хранит сессию."
    )
    refined = filter_answer(
        answer, [REST_NOTE], min_coverage=0.55, min_kept_ratio=0.25, min_kept_sentences=2
    )
    assert refined is not None
    assert "层次" not in refined


def test_filter_answer_none_when_mostly_unsupported() -> None:
    answer = (
        "REST использует GraphQL и WebSockets для реал-тайма.\n"
        "Сервисы шифруют трафик алгоритмом AES-512."
    )
    refined = filter_answer(
        answer, [REST_NOTE], min_coverage=0.55, min_kept_ratio=0.25, min_kept_sentences=2
    )
    assert refined is None


def test_unsupported_sentences_keeps_connectors() -> None:
    answer = "Таким образом. REST — парадигма проектирования API."
    bad = unsupported_sentences(answer, [REST_NOTE], min_coverage=0.55)
    # Коннектор «Таким образом» не должен быть помечен как неподдержанный.
    bad_idx = {i for i, _ in bad}
    assert 0 not in bad_idx


def test_sentence_number_check_runs_even_when_words_supported() -> None:
    # «Stateless» есть в контексте, но выдуманное число 99999 — нет.
    answer = "REST — парадигма. Stateless статус код 99999."
    bad = unsupported_sentences(answer, [REST_NOTE], min_coverage=0.55)
    bad_idx = {i for i, _ in bad}
    assert 1 in bad_idx  # предложение с выдуманным числом вырезается


def test_sentence_pure_number_is_unsupported() -> None:
    answer = "REST — парадигма. Итого: 40404."
    bad = unsupported_sentences(answer, [REST_NOTE], min_coverage=0.55)
    assert 1 in {i for i, _ in bad}


def test_split_sentences_preserves_list_items() -> None:
    s = split_sentences("1. Клиент-сервер.\n2. Stateless.\n3. Кэширование.")
    assert len(s) == 3


# --- Сервисный уровень: foreign-ответ -> заглушка ---

def test_service_hallucinated_with_chinese_is_stub(monkeypatch, rest_chunk) -> None:
    hallucinated = (
        "REST — парадигма проектирования API.\n"
        "Layered System (层次系统).\n"
        "Code on done рекомендуется.\n"
        "Принципы включают кэширование [c1]"
    )
    monkeypatch.setattr(gen_service, "ollama", _FakeOllama(hallucinated))
    result = asyncio.run(gen_service.answer("расскажи всё про REST", [rest_chunk]))
    assert result.grounded is False
    assert result.answer == gen_service.STUB_ANSWER


def test_service_clean_rest_answer_is_grounded(monkeypatch, rest_chunk) -> None:
    clean = (
        "REST — парадигма проектирования API, не протокол. Принципы: "
        "клиент-серверная архитектура, Stateless, кэширование, единообразие "
        "интерфейса, Layered System, Code on Demand (необязательно). "
        "Уровни зрелости: 0, 1, 2, 3. REST не всегда требует HTTP или JSON. [c1]"
    )
    monkeypatch.setattr(gen_service, "ollama", _FakeOllama(clean))
    result = asyncio.run(gen_service.answer("расскажи всё про REST", [rest_chunk]))
    assert result.grounded is True
    assert "Code on Demand" in result.answer
    assert "Code on done" not in result.answer
    assert 1 in result.cited_chunk_ids


# --- Токен-бюджет в retrieval ---


async def _acount_stub(text: str) -> int:
    # Предсказуемая «дорогая» оценка: ~1 токен на слово.
    return max(1, len(text.split()))


def test_fit_budget_trims_overflow(monkeypatch) -> None:
    monkeypatch.setattr("app.config.settings.llm_num_ctx", 10)
    monkeypatch.setattr("app.config.settings.ctx_reserve", 0)
    monkeypatch.setattr("app.retrieval.service.acount_tokens", _acount_stub)
    rows = [{"content": "альфа бета гамма"} for _ in range(10)]
    fitted = asyncio.run(_fit_budget(rows))
    assert len(fitted) < len(rows)


def test_fit_budget_drops_oversized_single_chunk(monkeypatch) -> None:
    # Чанк крупнее всего бюджета отсекается (фикс переполнения первым чанком).
    monkeypatch.setattr("app.config.settings.llm_num_ctx", 3)
    monkeypatch.setattr("app.config.settings.ctx_reserve", 0)
    monkeypatch.setattr("app.retrieval.service.acount_tokens", _acount_stub)
    rows = [{"content": "альфа бета гамма дельта"}]
    assert asyncio.run(_fit_budget(rows)) == []


def test_fit_budget_keeps_all_when_under(monkeypatch) -> None:
    monkeypatch.setattr("app.config.settings.llm_num_ctx", 4096)
    monkeypatch.setattr("app.config.settings.ctx_reserve", 0)
    monkeypatch.setattr("app.retrieval.service.acount_tokens", _acount_stub)
    rows = [{"content": "короткий текст"} for _ in range(3)]
    assert asyncio.run(_fit_budget(rows)) == rows


def test_settings_model_is_3b() -> None:
    # Этап 1.1 рефакторинга: дефолт переключён на 3b (RAM-бюджет 16 ГБ).
    assert settings.llm_model == "qwen2.5:3b-instruct"


# --- yes/no классификатор: не должен ловить wh-вопросы ---

def test_yesno_rejects_wh_questions() -> None:
    from app.generation.service import _is_yesno

    assert _is_yesno("Какие принципы нужно соблюдать?") is False
    assert _is_yesno("Что обязательно в REST?") is False
    assert _is_yesno("Чем является HATEOAS?") is False
    assert _is_yesno("Расскажи, что это такое?") is False


def test_yesno_accepts_real_yesno() -> None:
    from app.generation.service import _is_yesno

    assert _is_yesno("Является ли REST протоколом?") is True
    assert _is_yesno("Обязательно ли REST использует JSON?") is True
    assert _is_yesno("Это правда?") is True


# --- Этап 5.4: сервисные тесты JSON/HATEOAS/REST с техническим синтаксисом ---

JSON_NOTE = (
    "JSON (JavaScript Object Notation) — текстовый формат обмена данными, "
    "основанный на JavaScript. Формат независим от JS и используется в любом "
    "языке программирования. Пример объекта: {\"name\": \"REST\", \"level\": 3}. "
    "Массив значений: [1, 2, 3]. Структура: ключ:значение, разделённые запятыми. "
    "Поддерживает типы: string, number, boolean, null, object, array."
)


@pytest.fixture
def json_chunk() -> ChunkResult:
    return ChunkResult(
        chunk_id=122,
        content=JSON_NOTE,
        page=None,
        section=None,
        source_name="Obsidian Vault/Resourses/JSON/JSON.md",
        score=0.8,
    )


HATEOAS_NOTE = (
    "HATEOAS (Hypermedia As The Engine Of Application State) — ограничение "
    "архитектуры REST, при котором сервер вместе с ответом передаёт клиенту "
    "гипермедийные ссылки на доступные действия. Это последний уровень "
    "зрелости REST (Уровень 3 по модели Ричардсона)."
)


@pytest.fixture
def hateoas_chunk() -> ChunkResult:
    return ChunkResult(
        chunk_id=128,
        content=HATEOAS_NOTE,
        page=None,
        section=None,
        source_name="Obsidian Vault/Resurses/REST/HATEOAS.md",
        score=0.8,
    )


def test_service_json_answer_is_grounded(monkeypatch, json_chunk) -> None:
    """Этап 5.4: ответ про JSON с {}/[] проходит grounding, а не превращается в stub.

    Главный кейс бага: до Этапа 5.2 {} и [] рубились has_foreign_script,
    хотя grounding_score=0.984 (word_coverage=1.0). Теперь должно проходить.
    """
    answer = (
        "JSON (JavaScript Object Notation) — текстовый формат обмена данными, "
        "основанный на JavaScript. Пример объекта: {\"name\": \"REST\", \"level\": 3}. "
        "Массив: [1, 2, 3]. Типы: string, number, boolean, null. [c122]"
    )
    monkeypatch.setattr(gen_service, "ollama", _FakeOllama(answer))
    result = asyncio.run(gen_service.answer("что такое JSON", [json_chunk]))
    assert result.grounded is True
    assert "{" in result.answer or "[" in result.answer  # технические символы сохранены
    assert 122 in result.cited_chunk_ids


def test_service_hateoas_answer_is_grounded(monkeypatch, hateoas_chunk) -> None:
    """Этап 5.4: фиксация текущего корректного поведения HATEOAS.

    Баг HATEOAS не воспроизводится (починен на Этапе 2), но тест защищает
    от регрессии при изменении порогов/фильтров.
    """
    answer = (
        "HATEOAS (Hypermedia As The Engine Of Application State) — ограничение "
        "архитектуры REST. Сервер передаёт гипермедийные ссылки на доступные "
        "действия. Это Уровень 3 зрелости REST. [c128]"
    )
    monkeypatch.setattr(gen_service, "ollama", _FakeOllama(answer))
    result = asyncio.run(gen_service.answer("что такое HATEOAS", [hateoas_chunk]))
    assert result.grounded is True
    assert 128 in result.cited_chunk_ids


def test_service_rest_answer_with_technical_syntax_is_grounded(monkeypatch, rest_chunk) -> None:
    """Этап 5.4: REST-ответ с HTML-like / техническим синтаксисом проходит grounding.

    Защита от регрессии REST-кейса из master_state.json: там has_foreign_script=True
    в raw-ответе диагностики REST (при grounding_score=0.817) — значит баг
    проявлялся и на REST-ответах с техническими символами.
    """
    answer = (
        "REST — парадигма проектирования API, не протокол. "
        "Пример эндпоинта: GET /api/speakers?format=json <resource>. "
        "Методы HTTP: GET, POST, PUT, DELETE. "
        "Stateless — сервер не хранит сессию. Уровни зрелости: 0, 1, 2, 3. "
        "REST не всегда требует HTTP или JSON. [c1]"
    )
    monkeypatch.setattr(gen_service, "ollama", _FakeOllama(answer))
    result = asyncio.run(gen_service.answer("расскажи всё про REST", [rest_chunk]))
    assert result.grounded is True
    assert 1 in result.cited_chunk_ids


# --- Этап 5.3: keyword-boost в retrieval ---

def test_content_keyword_boost_rewards_term_in_content() -> None:
    """Этап 5.3: чанк, содержащий термин в content, получает буст."""
    from app.retrieval.service import _content_keyword_boost

    rows = [
        {"content": "JSON — текстовый формат обмена данными", "sim": 0.5, "lex": 0.1},
        {"content": "REST — парадигма проектирования API", "sim": 0.5, "lex": 0.1},
    ]
    _content_keyword_boost(rows, ["json"])
    assert rows[0]["keyword_boost"] > 0.0  # содержит "json"
    assert rows[1]["keyword_boost"] == 0.0  # не содержит "json"


def test_hybrid_score_keyword_boost_ranks_exact_match_higher() -> None:
    """Этап 5.3: при равных sim/lex чанк с точным термином ранжируется выше."""
    from app.retrieval.service import _hybrid_score

    rows = [
        {"sim": 0.6, "lex": 0.2, "content": "REST — парадигма проектирования API"},
        {"sim": 0.6, "lex": 0.2, "content": "JSON (JavaScript Object Notation) — формат"},
    ]
    out = _hybrid_score([dict(r) for r in rows], alpha=0.6, keywords=["json"])
    # Второй чанк содержит "json" → буст → выше score
    assert out[1]["score"] > out[0]["score"]
