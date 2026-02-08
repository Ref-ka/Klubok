# Klubok

Сервис для оценки релевантности места (POI) широкому запросу пользователя.

## Архитектура решения

Система собрана из двух основных компонентов, которые можно комбинировать:

- **Бейзлайн: дообученный BERT-классификатор**
  - Реализован как `sentence-transformers` `CrossEncoder`.
  - Модель подгружается из директории `./models/classifier/content/classifier`.
  - На вход получает пару `(user_query, place_description)` и возвращает скор (float). Далее скор порогуется в `0/1`.

- **Агент (LLM) с опциональными инструментами**
  - Реализован на `pydantic_ai`.
  - Агент может работать:
    - **самостоятельно** (только LLM),
    - **с подсказкой от BERT** (инструмент `bert-classifier`),
    - **с веб-поиском** (инструмент `tavily_search`),
    - **с BERT + веб-поиск вместе**.
  - В API это управляется флагами `use_agent`, `use_bert`, `use_search`.

## Метрики и прогоны

В директории `data/evaluation` лежат сохранённые результаты (таргеты, предсказания и `classification_report`) для прогонов по валидационному датасету в разных конфигурациях системы:

- `data/evaluation/bert.txt` — только BERT
- `data/evaluation/agent.txt` — только агент
- `data/evaluation/agent_search.txt` — агент + веб-поиск
- `data/evaluation/agent_bert_search.txt` — агент + BERT + веб-поиск

## Установка

1. Создай и активируй виртуальное окружение.

2. Установи зависимости:

```bash
pip install -r requirements.txt
```

3. Настрой переменные окружения.

Скопируй `.env_example` в `.env` и заполни ключи:

- `VSEGPT_API_KEY` — ключ для LLM-провайдера
- `OPENAI_URL` — base URL для API (по умолчанию используется `https://api.vsegpt.ru/v1`)
- `TAVILY_API_KEY` — ключ Tavily (нужен только если включаешь `use_search=true`)

## Запуск приложения

Запуск FastAPI-сервиса:

```bash
python server.py
```

Сервис поднимается на `http://0.0.0.0:8000`.

### Пример запроса

`POST /relevance`

Тело запроса (JSON):

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
- `bert_score` — скор BERT (если использовался)
- `agent_output` — сырой вывод агента (если использовался)
- `used` — какие компоненты были задействованы

## Оценка на датасете (скрипты)

- `tools.py` — прогон бейзлайна BERT по `data/data_final_for_dls_eval_new.jsonl` и вывод метрик в консоль.
- `evaluation.py` — пример прогона агента по датасету и сохранение отчёта.
