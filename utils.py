import pandas as pd


def load_eval_data(file_name: str, keep_info_columns: bool = True) -> pd.DataFrame:
    data = pd.read_json(file_name, lines=True)
    data = data.rename(columns={"Text": "question"})
    cols_to_concatenate = ["address", "name", "normalized_main_rubric_name_ru", "prices_summarized",
                           "reviews_summarized"]
    temp_df = data[cols_to_concatenate].fillna("")

    data["text"] = temp_df.apply(lambda row: " ".join(filter(None, row)), axis=1)
    if not keep_info_columns:
        data = data.drop(
            ["address", "name", "normalized_main_rubric_name_ru", "prices_summarized",
             "reviews_summarized", "permalink", "relevance"], axis=1)
    data = data[data["relevance_new"] != 0.1]
    return data
