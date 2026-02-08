from typing import Optional
import re
import asyncio

from fastapi import FastAPI, HTTPException
import uvicorn

from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.tavily import tavily_search_tool

from sentence_transformers import CrossEncoder

from dto import SearchRequest, RelevanceRequest, RelevanceResponse
from config import config

app = FastAPI(title="Relevance Service")

openai_model = None
bert_model = None


@app.on_event("startup")
async def startup_event():
    global openai_model, bert_model
    try:
        openai_model = OpenAIChatModel(
            config.model_name,
            provider=OpenAIProvider(
                base_url=config.openai_url,
                api_key=config.openai_api_key
            ),
        )
    except Exception as e:
        openai_model = None
        app.logger = getattr(app, "logger", None)
        print("Warning: OpenAI model was not initialized on startup:", e)

    try:
        bert_model = CrossEncoder('./models/classifier/content/classifier')
        print("BERT model loaded.")
    except Exception as e:
        bert_model = None
        print("BERT model NOT loaded (will fail if used). Error:", e)


def _make_bert_tool():
    def bert_tool(ctx):
        if bert_model is None:
            raise RuntimeError("BERT model not available. Make sure ./models/classifier/content/classifier exists.")
        pair = (ctx.deps.user_query, ctx.deps.place_description)
        score = bert_model.predict([pair])[0]
        return float(score)

    return Tool(
        bert_tool,
        name="bert-classifier",
        description="Returns a float score for relevance (higher==more relevant).",
        takes_ctx=True
    )


def _make_agent(include_search: bool, include_bert: bool) -> Agent:
    if openai_model is None:
        raise RuntimeError("OpenAI model not initialized. Set OPENAI_API_KEY and OPENAI_URL in config.")

    tools = []
    if include_search:
        if not getattr(config, "tavily_api_key", None):
            raise RuntimeError("tavily_api_key not configured but use_search=True requested.")
        tools.append(tavily_search_tool(config.tavily_api_key))
    if include_bert:
        tools.append(_make_bert_tool())

    system_prompt = (
        "Вы - оценщик мест на карте относительно широких запросов пользователей. "
        "Ваша задача — дать точную оценку: релевантно ли место запросу пользователя. "
        "В контексте доступны две переменные: user_query и place_description. "
        "Инструменты: tavily_search (если доступен) и bert-classifier (если доступен)."
    )

    instructions = ("Для финального ответа выводи только одно число: "
                    "1 если релевантно, 0 если нет. Никакого лишнего текста.")

    agent = Agent(
        openai_model,
        tools=tools,
        system_prompt=system_prompt,
        instructions=instructions
    )
    return agent


def _parse_agent_output(agent_output: str) -> Optional[int]:
    if agent_output is None:
        return None
    s = agent_output.strip()
    if s in ("0", "1"):
        return int(s)
    m = re.search(r"\b([01])\b", agent_output)
    if m:
        return int(m.group(1))
    return None


@app.post("/relevance", response_model=RelevanceResponse)
async def relevance(req: RelevanceRequest):
    used = []
    decision = None
    agent_raw = None
    bert_score = None

    if not req.use_agent and req.use_bert:
        used.append("bert")
        if bert_model is None:
            raise HTTPException(status_code=500, detail="BERT model not loaded on server.")
        try:
            bert_score = float(bert_model.predict([(req.user_query, req.place_description)])[0])
            decision = 1 if bert_score >= req.bert_threshold else 0
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error running bert: {e}")

        return RelevanceResponse(
            decision=decision,
            bert_score=bert_score,
            agent_output=None,
            used=used,
            note=f"Threshold used: {req.bert_threshold}"
        )

    if req.use_agent:
        used.append("agent")
        if req.use_search:
            used.append("search")
        if req.use_bert:
            used.append("bert")

        try:
            agent = _make_agent(include_search=req.use_search, include_bert=req.use_bert)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        deps = SearchRequest(user_query=req.user_query, place_description=req.place_description)
        prompt = (
            f"Определи релевантность места запросу пользователя.\n"
            f"user_query: {req.user_query}\n"
            f"place_description: {req.place_description}\n"
            f"Выведи только 0 или 1."
        )

        try:
            result = await agent.run(user_prompt=prompt, deps=deps)
            agent_raw = getattr(result, "output", None) or str(result)
            parsed = _parse_agent_output(agent_raw)
            if parsed is not None:
                decision = parsed
            else:
                if req.use_bert:
                    if bert_model is None:
                        raise HTTPException(status_code=500,
                                            detail="Agent output ambiguous and BERT model not available locally.")
                    bert_score = float(bert_model.predict([(req.user_query, req.place_description)])[0])
                    decision = 1 if bert_score >= req.bert_threshold else 0
                else:
                    m = re.search(r"([01])", agent_raw or "")
                    if m:
                        decision = int(m.group(1))
                    else:
                        raise HTTPException(status_code=500, detail=f"Cannot parse agent output: {agent_raw}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent execution error: {e}")

        return RelevanceResponse(
            decision=decision,
            bert_score=bert_score,
            agent_output=agent_raw,
            used=used,
            note=None
        )

    raise HTTPException(status_code=400, detail="No component selected. Set use_agent or use_bert to True.")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
