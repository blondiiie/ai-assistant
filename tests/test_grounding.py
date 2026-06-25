from __future__ import annotations

from app.generation.grounding import check, is_supported, parse_cited_ids


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
    assert is_supported(answer, cited, min_overlap=0.3) is True


def test_is_supported_unrelated() -> None:
    answer = "Стоимость билета на Марс составляет миллион"
    cited = ["Заявление на оплачиваемый отпуск подаётся за 14 дней"]
    assert is_supported(answer, cited, min_overlap=0.3) is False


def test_is_supported_empty() -> None:
    assert is_supported("ответ", [], min_overlap=0.3) is False
