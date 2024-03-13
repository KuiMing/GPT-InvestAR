"""
Get GPT scores with questions for inference
"""

import os
import sys
import argparse
from datetime import datetime
import json
import glob
import time
import pandas as pd
from gpt_scores_as_features import (
    get_systemprompt_template,
    load_index,
    load_query_engine,
    initialize_and_return_models,
)

__import__("pysqlite3")

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import sqlite3


def get_gpt_generated_feature_dict(query_engine, feature_name, question):
    """
    Returns:
        A dictionary with keys as question identifiers and value as GPT scores.
    """
    response_dict = {}
    #     for feature_name, question in questions_dict.items():
    # Sleep for a short duration, not to exceed openai rate limits.
    time.sleep(1)
    try:
        response = query_engine.query(question)
    # pylint: disable=broad-except
    except Exception:
        time.sleep(60)
        response_dict = get_gpt_generated_feature_dict(
            query_engine, feature_name, question
        )
    try:
        response_dict[feature_name] = int(
            response.response.split(":")[-1].replace("}", "")
        )
    # pylint: disable=broad-except
    except Exception:
        response_dict = get_gpt_generated_feature_dict(
            query_engine, feature_name, question
        )

    return response_dict


def main(args):
    """
    main function
    """

    with open(args.config_path, encoding="utf8") as json_file:
        config_dict = json.load(json_file)
    with open(args.questions_path, encoding="utf8") as json_file:
        questions_dict = json.load(json_file)
    os.environ["OPENAI_API_KEY"] = config_dict["openai_api_key"]
    os.environ["OPENAI_API_VERSION"] = config_dict["openai_api_version"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = config_dict["azure_openai_endpoint"]
    llm, embedding_model = initialize_and_return_models(config_dict)

    embeddings_directory = args.save_directory

    path = glob.glob(os.path.abspath(embeddings_directory) + "/*/*/")
    path.sort()
    path = pd.DataFrame({"path": path})
    path["date"] = path.path.str.extract(r"([0-9]{4}-[0-9]{2}-[0-9]{2})")
    path["symbol"] = path.path.str.extract(r"([A-Z]*/2)")
    path["symbol"] = path.symbol.str.replace("/2", "")
    connect = sqlite3.connect(args.result_path)
    date = args.date
    if date is None:
        date = pd.read_sql("SELECT MAX(report_date) as date FROM feature", connect)
        date = date.date.values[0]
    path = path[path.date > date]

    for ar_date, symbol in path[["date", "symbol"]].values:
        print(ar_date, symbol)

        index = load_index(llm, embedding_model, embeddings_directory, symbol, ar_date)
        text_qa_template = get_systemprompt_template(config_dict)
        query_engine = load_query_engine(index, text_qa_template)
        # Get feature scores as dictionary
        gpt_feature_dict = dict()
        for feature_name, question in questions_dict.items():
            gpt_feature_dict.update(
                get_gpt_generated_feature_dict(query_engine, feature_name, question)
            )
        gpt_feature_df = pd.DataFrame.from_dict(gpt_feature_dict, orient="index").T
        gpt_feature_df.columns = [f"feature_{c}" for c in gpt_feature_df.columns]
        gpt_feature_df["symbol"] = symbol
        gpt_feature_df["report_date"] = ar_date
        gpt_feature_df.to_sql("feature", connect, index=False, if_exists="append")
        # if not os.path.exists(args.result_path):
        #     gpt_feature_df.to_csv(args.result_path, index=False)
        # else:
        #     gpt_feature_df.to_csv(args.result_path, index=False, mode="a", header=False)
    connect.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        dest="config_path",
        type=str,
        default="config.json",
        help="""Full path of config.json""",
    )
    parser.add_argument(
        "--questions_path",
        dest="questions_path",
        type=str,
        default="questions.json",
        help="""Full path of questions.json which contains the questions
                        for asking to the LLM""",
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        default="/root/GPT-InvestAR/inference_chroma",
        required=True,
        help="""absolute path of chroma""",
    )
    parser.add_argument("--date", type=str, default=None, help="the starting date")
    parser.add_argument(
        "--result_path",
        default="feature.sqlite",
        type=str,
        help="path of inference result",
    )

    main(args=parser.parse_args())
