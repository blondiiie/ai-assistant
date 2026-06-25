# План: Корпоративный AI-ассистент (RAG) — MVP

## Контекст и цели
Внутренний Telegram-бот, отвечающий сотрудникам на основе корпоративных
документов по принципу RAG. Киллер-фича — отсутствие галлюцинаций: бот отвечает
только по найденному в базе контексту, иначе честно признаёт отсутствие
информации.

Параллельная цель заказчика — **личная практика проектирования системы**, поэтому
часть фаз оформлена как самостоятельные проектные задания с критериями приёмки
и ревью, а не как готовая реализация.

## Согласованные решения
- **Бизнес-логика v2, MVP-подмножество**:
  - метаданные в чанках (doc_id, source_name, page, section);
  - гибридный поиск (вектор + лексический), top-k 3–5;
  - порог релевантности как главный анти-галлюцинационный механизм;
  - grounded-промпт + постпроверка цитирования;
  - семафор конкурентности к LLM.
- **Вне MVP** (post-MVP фаза 7): multi-turn история, версионирование документов.
- **Стек**: Python 3.12, FastAPI + aiogram 3 + SQLAlchemy 2 (async) + pgvector,
  ручной RAG без LangChain.
- **LLM/Embedding**: нативная локальная Ollama; модель `qwen2.5:3b-instruct`
  (Q4_K_M), эмбеддинги `nomic-embed-text`. 7B не используем ради 16 ГБ RAM и SLA.
- **Окружение**: Postgres 16 + pgvector в Docker (OrbStack); backend и Ollama —
  нативно в venv (бережно к CPU M5, Ollama в нативе быстрее на Apple Silicon).
- **Ограничения железа**: MacBook Air M5, 16 ГБ RAM, 512 ГБ SSD. Принять
  SLA ответа 3–10 сек (локальная LLM).
- **Форматы документов (MVP)**: PDF, DOCX, TXT.

## Согласованный анализ узких мест (учитываются в плане)
1. Голое чанкование без метаданных → грязные эмбеддинги и нет ссылок на источник.
2. Нет дедупа/обновления документов (в MVP — ручное управление через doc_id).
3. top-k=1 недостаточно → гибридный поиск top-k 3–5.
4. Отсутствие cosine-порога ломает киллер-фичу → обязательный порог.
5. Один промпт не гарантирует отсутствие галлюцинаций → постпроверка grounding.
6. Ссылки на источник требуют page/section в чанке; Telegram ограничивает
   гиперссылки → формат сноски «📄 Документ X, стр. Y».
7. SLA <5 сек недостижим локально на 7B → малая модель 3B + принятие 3–10 сек.
8. Ollama сериализует запросы → семафор (1–2 конкурентных).
9. Авторизация сотрудников в MVP — вне области (один общий бот/белый список
   chat_id как минимум).

## Рекомендуемая структура проекта
```
corp_assistant/
  app/
    config.py          # pydantic-settings
    db/                # engine, session, models (documents, chunks)
    ingest/            # parsing, chunking, embedding, ingest pipeline
    retrieval/         # hybrid search, threshold, rerank
    generation/        # prompt builder, grounding check, formatter
    llm/               # ollama client (embed + chat), provider abstraction
    bot/               # aiogram handlers
    api/               # FastAPI admin endpoints (upload, health)
  tests/
  docker-compose.yml   # postgres+pgvector
  Makefile / pyproject.toml
```

## Оценка локальной реализации (M5, 16 ГБ)
| Компонент | Реализуемо | Замечание |
|---|---|---|
| Telegram-бот (aiogram) | ✅ | тривиально |
| Postgres+pgvector (Docker) | ✅ | лёгкий контейнер |
| Embedding (nomic-embed-text) | ✅ | ~270 МБ, быстро |
| LLM qwen2.5:3b (Ollama) | ✅ | 3–10 сек/ответ |
| Парсинг PDF/DOCX | ✅ | медленнее на сканах |
| SLA <5 сек | ❌ | недостижим локально, принято 3–10 сек |

**Вывод**: полностью реализуем как учебный проект. Следить за совокупной RAM
(Postgres + Ollama 3B + backend ≤ ~5–6 ГБ).

---

## Фазы плана

### Фаза 0 — Окружение и базовая инфраструкра (реализация)
- [x] Установить Ollama нативно; `ollama pull nomic-embed-text`,
      `ollama pull qwen2.5:3b-instruct`.
- [x] Поднять `docker-compose.yml` с `pgvector/pgvector:pg16`.
- [x] Скелет проекта: pyproject.toml (uv), ruff, pytest, pre-commit.
- [x] `config.py` (pydantic-settings): DSN, OLLAMA_URL, MODEL_NAME,
      EMBED_MODEL, TOP_K, SIM_THRESHOLD, MAX_CONCURRENT_LLM.
- [x] Smoke-тест: запрос эмбеддинга и чата к Ollama из Python.
- **Критерий приёмки ✅**: `make up` поднимает БД (pgvector+pg_trgm созданы),
  `uv run python -m app.smoke_test` получает вектор (dim=768) и ответ от
  Ollama (qwen2.5:3b-instruct).

### Фаза 1 — ПРАКТИКА проектирования (задание заказчика)
- [x] Спроектировать **ER-схему**: таблицы `documents` (id, source_name,
      document_type, version, active, file_path, created_at, updated_at) и
      `chunks` (id, document_id FK, chunk_index, content, page, section,
      embedding vector(768), tsv tsvector GENERATED STORED).
- [x] Индексы: embedding HNSW (cosine), tsv GIN, (document_id, chunk_index) UNIQUE B-tree.
- [x] Определить **API-контракты** (JSON): admin upload, retrieve, generate.
- [x] Архитектурная **диаграмма** (Mermaid): ingest flow + query flow.
- [x] Границы модулей и протоколы (llm/ingest/retrieval/generation/api/bot).
- **Критерий приёмки ✅**: `design.md` утверждён (части I–III).

### Фаза 2 — Ингест документов (реализация)
- [x] Модели SQLAlchemy + инициализация схемы (расширения `vector`, `pg_trgm`;
      таблицы `documents`/`chunks`; индексы HNSW/GIN/UNIQUE; `tsv` GENERATED).
- [x] Парсеры: pdfplumber/pypdf (постранично), python-docx, plain text.
- [x] Чанкование: ~512 токенов, overlap ~50, с метаданными page/section.
- [x] Эмбеддинг батчами через Ollama `nomic-embed-text` (endpoint `/api/embed`).
- [x] Запись в `chunks` (content, embedding, tsv, метаданные).
- [x] Admin-эндпоинт FastAPI `POST /documents` (multipart) + хранение файла.
- **Критерий приёмки ✅**: загруженный документ нарезан на чанки с метаданными и
  векторами, в БД видны строки (document_id=1/2, версия инкрементируется, tsv
  наполняется русским конфигом, embedding != NULL).

### Фаза 3 — ПРАКТИКА стратегии поиска (задание заказчика)
- [x] Спроектирован и реализован **гибридный поиск**: векторный
      (`1 - (embedding <=> q)`, HNSW) + лексический (`ts_rank` по `tsv`),
      нормализация per-query, гибридный скор `alpha*sim + (1-alpha)*lex`.
- [x] Зафиксирована семантика **порога**: similarity = `1 - distance`,
      сравнение `sim >= sim_threshold` (config).
- [x] top-k реранк по гибридному скору; кандидаты = `retrieval_candidates`.
- **Критерий приёмки ✅**: параметры обоснованы и реализованы в
      `app/retrieval/service.py` (`design.md` дополнен semantics порога).

### Фаза 4 — Retrieval + Generation (реализация)
- [x] Модуль `retrieval`: гибридный запрос к pgvector, фильтр по порогу;
      `found=False` → заглушка.
- [x] `generation.prompt`: системное правило + контекст с `[c<id>]` + вопрос.
- [x] `generation.grounding`: парсинг cited id, отсев невалидных, accept если
      есть хотя бы один валидный id (робастность к спурьным тегам 3B-модели).
- [x] `llm.ollama_client`: chat stream=False, таймауты/ретраи через цикл
      grounding (`grounding_max_retries`).
- [x] Семафор `MAX_CONCURRENT_LLM` вокруг вызовов LLM.
- [x] Оркестратор `app/assistant.ask()`.
- **Критерий приёмки ✅**: нерелевантный вопрос → заглушка; релевантный →
      grounded-ответ со ссылкой на chunk (проверено на регламенте отпусков).

### Фаза 5 — Telegram-бот (реализация)
- [x] aiogram 3: `/start`, `/help`, свободный ввод.
- [x] Белый список `chat_id` из конфига (если пусто — dev-режим: доступ всем).
- [x] Обработка ошибок/таймаутов; статусное сообщение «Ищу в регламентах…».
- [x] Форматирование: inline `[cNN]` убираются, сноски «📄 Документ X, стр. Y».
- [ ] Команда `/upload` (админ) — оставлено на post-MVP (есть FastAPI `/documents`).
- **Критерий приёмки ✅ (часть)**: полный цикл вопрос→ответ+сноски работает через
      `assistant.ask()` и форматтер бота; запуск `make bot` требует токен в `.env`.

### Фаза 6 — Тестирование и калибровка на M5 (реализация)
- [ ] Тестовый корпус Q&A (≥20 пар: relevant + out-of-scope).
- [ ] Метрики: % ответов-заглушек на out-of-scope (анти-галлюцинация),
      точность источника, латентность энд-ту-енд.
- [ ] Профиль RAM/CPU при 1–2 конкурентных запросах; корректировка
      MAX_CONCURRENT_LLM.
- [ ] Юнит-тесты: чанкование, гибридный скор, grounding_check.
- **Критерий приёмки**: out-of-scope вопросы получают заглушку ≥90%;
  p95 латентности зафиксирован.

### Фаза 7 — Расширение post-MVP (по желанию)
- [ ] Multi-turn история диалога (последние N сообщений на пользователя).
- [ ] Версионирование документов: повторная загрузка инвалидирует старые чанки
      того же doc_id (soft-delete), дедуп.
- [ ] OCR для сканов; извлечение таблиц.

---

## Риски
> Важно: различать **процессор (CPU)** и **память (RAM)** — это разные ресурсы.

- **Процессор (CPU) — риска нет.** Ollama в нативе на Apple Silicon
  использует GPU/Neural Engine, а не CPU; модель 3B лёгкая; семафор
  ограничивает конкурентность до 1–2. CPU будет почти в простое. Стек
  спроектирован бережно к M5.
- **RAM (16 ГБ) — запас есть, страхуемся мягко.** Postgres idle ~60 МБ,
  Ollama(3B Q4) ~2 ГБ, backend ~150 МБ → итого ~2.5–3 ГБ из 16. Реальной
  угрозы нет. На всякий случай: при нехватке снижать MAX_CONCURRENT_LLM и
  ограничивать память контейнера БД (`mem_limit`, `shared_buffers=128MB`).
- **Версия Postgres**: память слабо зависит от версии (13/15/16/17 ~одинаковы
  в idle). Снижение версии выигрыша по RAM не даст. Ограничение: pgvector
  требует PostgreSQL ≥ 13. Оставляем **Postgres 16** (свежий, стабильный);
  экономию RAM даём через тюнинг контейнера, а не понижение версии.
- **Качество 3B-модели** на русском: qwen2.5:3b справляется, но проверять
  инструкцию-следование; при проблемах рассмотреть 7B только для inference
  в режиме one-by-one.
- **Порог релевантности** чувствителен к корпусу — обязательно калибровать
  (фаза 3, 6), не фиксировать «на глаз».
- **Парсинг PDF со сканами/таблицами** — слабое место, для MVP ограничить
  текстовыми PDF.

## План валидации
- Smoke: Fаза 0.
- Ингест: Fаза 2 (видны чанки в БД).
- Retrieval/grounding: Fаза 4 + юнит-тесты.
- E2E в Telegram: Fаза 5.
- Метрики качества и нагрузка: Fаза 6.

## Открытые вопросы (post-MVP)
- Авторизация/ролевая модель сотрудников.
- Источник документов (Confluence/Notion) — интеграции.
- Развёртывание за пределами ноутбука (прод/сервер).
