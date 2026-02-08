import asyncio

from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.tavily import tavily_search_tool
from sklearn.metrics import classification_report

from config import config
from dto import SearchRequest
from tools import call_classifier
from utils import load_eval_data


async def main():
    model = OpenAIChatModel(
        config.model_name,
        provider=OpenAIProvider(
            base_url=config.openai_url,
            api_key=config.openai_api_key
        ),
    )

    agent = Agent(model,
                  tools=[
                      tavily_search_tool(config.tavily_api_key),
                      # Tool(
                      #     call_classifier,
                      #     name="bert-classificator",
                      #     description="Function to call bert classifier that will return only one float number - "
                      #                 "logit of place description to be relevan to user query.",
                      #     takes_ctx=True
                      # )
                  ],
                  system_prompt="Вы - оценщик мест на карте относительно широких запросов пользователей. "
                                "Вашей главной задачей является точная оценка того, "
                                "является ли место релевантным или нет. "
                                "В качестве контекста вам предоставляется две переменные: "
                                "1) user_query - это непосредственно запрос пользователя; "
                                "2) place_description - это описание потенциально релевантного места, "
                                "которое нужно оценить. "
                                "Также вам предоставляется два инструмента: "
                                # "1) bert-classifier - это дообученный трансформер, который выдает свое решение в виде "
                                # "вещественного числа в отрезке [0, 1]. Помните, что классификатор может ошибаться! "
                                "1) tavily_search - позволяет получить актуальные данные из интернета "
                                "о запрашиваемом месте.",
                  instructions="Для финального ответа выводи только одно число: "
                               "1 - если место релевантно, 0 - если место не релевантно. "
                               "Не забывай использовать инструменты, например tavily_search.")  #  bert-classifier и

    data = load_eval_data("data/data_final_for_dls_eval_new.jsonl")
    data = data[:500]
    predicted = []
    target = list(map(int, data["relevance_new"].tolist()))
    for index, row in data.iterrows():
        deps = SearchRequest(
            user_query=row["question"],
            place_description=row["text"]
        )
        result = await agent.run(
            user_prompt=f"Определи, релевантность места запросу пользователя. "
                        f"Запрос пользователя: {row['question']}. "
                        f"Данные о месте: "
                        f"название: {row['name']} "
                        f"адрес: {row['address']} "
                        f"категория: {row['normalized_main_rubric_name_ru']} "
                        f"описание меню/товаров/услуг: {row['prices_summarized']} "
                        f"суммарные отзывы: {row['reviews_summarized']} ",
            deps=deps
        )
        try:
            result = int(result.output)
            print("\n======================\n"
                  f"Query: {row['question']}, Name: {row['name']}\n"
                  f"Target: {row['relevance_new']}, Score: {result}\n")
            predicted.append(result)
        except Exception as e:
            print(e)
    report = classification_report(target, predicted)
    print(target)
    print(predicted)
    print(report)
    with open(file=f"./data/evaluation/tools_only_search.txt", mode="w") as file:
        file.writelines([str(target), str(predicted), report])


asyncio.run(main())
