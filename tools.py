from typing import List

from pydantic_ai import RunContext
from sentence_transformers import CrossEncoder
from sklearn.metrics import classification_report

from dto import SearchRequest
from utils import load_eval_data


def call_classifier(ctx: RunContext[SearchRequest]):
    print("Classifier has been called!")
    classifier = CrossEncoder('./models/classifier/content/classifier')
    print("user_query")
    print(ctx.deps.user_query)
    print("place_description")
    print(ctx.deps.place_description)
    score = classifier.predict((ctx.deps.user_query, ctx.deps.place_description))
    print(f"Classifier decision: {score.tolist()}")
    return score.tolist()


def call_classifier_t(ctx: List[SearchRequest]):
    model = CrossEncoder('./models/classifier/content/classifier')
    print("Model initialized!")
    sentences = list([sr.user_query, sr.place_description] for sr in ctx)
    score = model.predict(sentences)
    print("Score calculated!!!")
    return score.tolist()


def determinate(x: float, threshold: float = 0.5) -> int:
    if x < threshold:
        return 0
    else:
        return 1


if __name__ == "__main__":
    eval_data = load_eval_data("data/data_final_for_dls_eval_new.jsonl")
    print("Data has been loaded!")

    target = eval_data["relevance_new"].tolist()

    scores = []
    search_requests = []
    for index, row in eval_data.iterrows():
        search_requests.append(SearchRequest(
            user_query=row["question"],
            place_description=row["text"]
        ))

    scores = call_classifier_t(search_requests)

    thresholds = [0.5]  # 0.1, 0.3, , 0.8, 0.9
    for threshold in thresholds:
        print(f"Threshold = {threshold}")
        scores_new = list(map(lambda x: determinate(x, threshold), scores))
        print(target)
        print(scores_new)
        print(classification_report(target, scores_new))
        print("=================================================")
