# Klubok

Сервис для оценки релевантности места (POI) широкому запросу пользователя: по паре «запрос — описание места» возвращает `1` (релевантно) или `0` (нерелевантно).

## Архитектура решения

Система собрана из двух основных компонентов, которые можно комбинировать:

- **Бейзлайн: дообученный BERT-классификатор**
  - Cross-encoder на базе [`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2), дообученный с `BinaryCrossEntropyLoss` (`sentence-transformers` `CrossEncoder`).
  - Модель подгружается из директории `./models/classifier/content/classifier`.
  - На вход получает пару `(user_query, place_description)` и возвращает скор (float). Далее скор порогуется в `0/1` (порог `bert_threshold`, по умолчанию `0.5`).

- **Агент (LLM) с опциональными инструментами**
  - Реализован на `pydantic_ai`, LLM — `openai/gpt-5-nano` через OpenAI-совместимый API (по умолчанию [VseGPT](https://vsegpt.ru)). Модель задаётся в `config.py`.
  - Агент может работать:
    - **самостоятельно** (только LLM),
    - **с подсказкой от BERT** (инструмент `bert-classifier`),
    - **с веб-поиском** (инструмент `tavily_search`),
    - **с BERT + веб-поиск вместе**.
  - В API это управляется флагами `use_agent`, `use_bert`, `use_search`.

## Метрики

Валидация на 500 примерах из `data/data_final_for_dls_eval_new.jsonl` (183 нерелевантных, 317 релевантных):

| Конфигурация | Accuracy | F1 (класс 0) | F1 (класс 1) | Macro F1 |
|---|---|---|---|---|
| Только BERT | 0.66 | 0.59 | 0.71 | 0.65 |
| Только агент | 0.76 | 0.67 | 0.82 | 0.74 |
| Агент + веб-поиск | **0.79** | **0.68** | **0.84** | **0.76** |
| Агент + BERT + веб-поиск | 0.77 | 0.65 | 0.83 | 0.74 |

Полные результаты (таргеты, предсказания и `classification_report`) лежат в `data/evaluation`:

- `data/evaluation/bert.txt` — только BERT
- `data/evaluation/agent.txt` — только агент
- `data/evaluation/agent_search.txt` — агент + веб-поиск
- `data/evaluation/agent_bert_search.txt` — агент + BERT + веб-поиск

## Установка

Нужен Python 3.11+.

1. Создай и активируй виртуальное окружение.

2. Установи зависимости:

```bash
pip install -r requirements.txt
```

> `requirements.txt` содержит Windows-only пакеты `pywin32` и `pywin32-ctypes`. На Linux/macOS удали эти строки перед установкой.

3. Положи веса BERT-классификатора в `./models/classifier/content/classifier` (директория `models/` не хранится в репозитории). Без модели сервис запустится, но запросы с `use_bert=true` будут падать с ошибкой.

4. Для скриптов оценки распакуй датасеты в `data/`:

```bash
unzip data/data.zip -d data/
```

Должны появиться `data/data_final_for_dls_new.jsonl` (train) и `data/data_final_for_dls_eval_new.jsonl` (валидация).

5. Настрой переменные окружения.

Скопируй `.env_example` в `.env` и заполни ключи:

- `VSEGPT_API_KEY` — ключ для LLM-провайдера
- `OPENAI_URL` — base URL для API (если не задан, используется `https://api.vsegpt.ru/v1`)
- `TAVILY_API_KEY` — ключ Tavily (нужен только если включаешь `use_search=true`)

## Запуск приложения

Запуск FastAPI-сервиса:

```bash
python server.py
```

Сервис поднимается на `http://0.0.0.0:8000`, Swagger UI доступен на `http://127.0.0.1:8000/docs`.

### `POST /relevance`

Параметры запроса (JSON):

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `user_query` | string | — | Запрос пользователя |
| `place_description` | string | — | Описание места |
| `use_agent` | bool | `true` | Использовать LLM-агента |
| `use_search` | bool | `false` | Дать агенту веб-поиск (Tavily) |
| `use_bert` | bool | `false` | Использовать BERT-классификатор |
| `bert_threshold` | float | `0.5` | Порог для скора BERT |

Как выбираются компоненты:

- `use_agent=false`, `use_bert=true` — решение принимает только BERT по порогу `bert_threshold`.
- `use_agent=true` — решение принимает агент; `use_bert` и `use_search` подключают ему соответствующие инструменты. Если ответ агента не удалось распарсить в `0/1`, а `use_bert=true`, решение берётся по скору BERT.
- `use_agent=false`, `use_bert=false` — ошибка `400`.

Пример тела запроса:

```json
{
  "user_query": "где поесть суши",
  "place_description": "Tokyo Sushi, ресторан японской кухни, роллы, доставка",
  "use_agent": true,
  "use_search": false,
  "use_bert": true,
  "bert_threshold": 0.5
}
```

Пример через `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/relevance" \
  -H "Content-Type: application/json" \
  -d "{\"user_query\":\"где поесть суши\",\"place_description\":\"Tokyo Sushi, ресторан японской кухни\",\"use_agent\":true,\"use_search\":false,\"use_bert\":true,\"bert_threshold\":0.5}"
```

Ответ содержит:

- `decision` — итог `0/1`
- `bert_score` — скор BERT (если использовался без агента или как фолбэк)
- `agent_output` — сырой вывод агента (если использовался)
- `used` — какие компоненты были задействованы
- `note` — служебная информация (например, использованный порог BERT)

## Оценка на датасете (скрипты)

Обе утилиты читают `data/data_final_for_dls_eval_new.jsonl` (примеры с `relevance_new == 0.1` отбрасываются, описание места собирается из адреса, названия, рубрики, цен и отзывов — см. `utils.py`).

- `python tools.py` — прогон бейзлайна BERT и вывод `classification_report` в консоль.
- `python evaluation.py` — прогон агента на первых 500 примерах; отчёт сохраняется в `data/evaluation/tools_only_search.txt`. Набор инструментов агента (по умолчанию только `tavily_search`) и имя файла отчёта задаются прямо в коде скрипта.

## Структура проекта

```
server.py        # FastAPI-сервис (/relevance)
config.py        # модель LLM и ключи из .env
dto.py           # схемы запросов/ответов
tools.py         # BERT-классификатор и оценка бейзлайна
evaluation.py    # оценка агента на датасете
utils.py         # загрузка и подготовка датасета
data/evaluation/ # сохранённые результаты прогонов
```
