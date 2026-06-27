from __future__ import annotations

from app.generation.grounding import (
    check,
    is_refusal,
    is_supported,
    missing_distinctive_tokens,
    parse_cited_ids,
)


def test_parse_cited_ids() -> None:
    assert parse_cited_ids("ответ [c3] и [c17] тут") == [3, 17]
    assert parse_cited_ids("нет ссылок") == []


def test_check_filters_invalid_and_keeps_order() -> None:
    valid, has_any, invalid = check("текст [c3] [c2] [c3]", context_chunk_ids={3})
    assert valid == [3]
    assert has_any is True
    assert invalid == {2}


def test_check_no_valid_citation() -> None:
    valid, has_any, invalid = check("текст [c5]", context_chunk_ids={3})
    assert valid == []
    assert has_any is False
    assert invalid == {5}


def test_is_supported_overlap() -> None:
    answer = "Заявление на отпуск подаётся за 14 дней"
    cited = ["Заявление на оплачиваемый отпуск подаётся не позднее 14 дней"]
    assert is_supported(answer, cited, min_overlap=0.35) is True


def test_is_supported_unrelated() -> None:
    answer = "Стоимость билета на Марс составляет миллион"
    cited = ["Заявление на оплачиваемый отпуск подаётся за 14 дней"]
    assert is_supported(answer, cited, min_overlap=0.35) is False


def test_is_supported_empty() -> None:
    assert is_supported("ответ", [], min_overlap=0.35) is False


def test_is_supported_rejects_hallucinated_terms() -> None:
    answer = "Уровень 0 использует кэширование и Stateless для CRUD операций"
    cited = [
        "Уровень 0: сервисы используют HTTP как транспорт и один HTTP-глагол POST"
    ]
    assert is_supported(answer, cited, min_overlap=0.35) is False


def test_is_refusal_ascii_sentinel() -> None:
    assert is_refusal("NOANSWER") is True
    assert is_refusal("noanswer") is True
    assert is_refusal("NOANSWER (отсутствуют сведения об архитектуре)") is True


def test_is_refusal_russian_variants() -> None:
    assert is_refusal("НЕВОЗМОЖНО ОТВЕТИТЬ") is True
    assert is_refusal("Невозможно ответить, данных нет") is True
    assert is_refusal("Не могу ответить") is True
    assert is_refusal("Нет информации") is True


def test_is_refusal_bug_repro_marker_with_parenthetical() -> None:
    raw = (
        "НЕВОЗМОЖНО_ОТВЕТИТЬ (отсутствуют сведения об архитектуре "
        "клиента-сервера, кэшировании и передачи кода)"
    )
    assert is_refusal(raw) is True


def test_is_refusal_not_triggered_by_legit_answer() -> None:
    assert is_refusal("Подать заявление невозможно позже 14 дней") is False
    assert is_refusal("") is False
    assert is_refusal("Отпуск составляет 28 дней") is False


def test_missing_distinctive_empty_when_supported() -> None:
    answer = "JSON — это JavaScript Object Notation, значения: true, false, null"
    cited = ["JSON (JavaScript Object Notation): true, false, null"]
    assert missing_distinctive_tokens(answer, cited) == set()


def test_missing_distinctive_catches_invented_http_codes() -> None:
    # Модель выдумала HTTP-коды 200/400 — их нет в контексте про JSON.
    answer = "JSON-объект содержит пары. Коды: 200 OK, 400 Bad Request, 404 Not Found"
    cited = ["JSON-объект — неупорядоченный набор пар ключ:значение"]
    missing = missing_distinctive_tokens(answer, cited)
    assert "200" in missing
    assert "400" in missing
    assert "404" in missing


def test_missing_distinctive_catches_invented_abbreviation() -> None:
    answer = "REST использует протокол CRUD для операций."
    cited = ["REST — концепция клиент-серверной архитектуры."]
    assert "crud" in missing_distinctive_tokens(answer, cited)


def test_missing_distinctive_ignores_trivial_latin() -> None:
    answer = "This is true and done here"
    cited = ["Заметка без латиницы"]
    assert missing_distinctive_tokens(answer, cited) == set()
