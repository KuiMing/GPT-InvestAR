"""
Create embedding vector for inference
"""

import os
import glob
import time
from datetime import datetime, timedelta
import json
import argparse
import pandas as pd
from llama_index import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from embeddings_save import save_index


def get_folders(root_path):
    path = glob.glob(f"{root_path}/*/*")
    path.sort()
    path = pd.DataFrame({"path": path})
    path["date"] = path.path.str.extract(r"([0-9]{4}-[0-9]{2}-[0-9]{2})")
    path["symbol"] = path.path.str.extract(r"([A-Z]*/2)")
    path["symbol"] = path.symbol.str.replace("/2", "")
    return path


def main(args) -> None:
    """
    main function
    """
    with open(args.config_path, encoding="utf8") as json_file:
        config_dict = json.load(json_file)
    # Read the targets df generated from make_targets.py
    os.environ["OPENAI_API_KEY"] = config_dict["openai_api_key"]
    os.environ["OPENAI_API_VERSION"] = config_dict["openai_api_version"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = config_dict["azure_openai_endpoint"]
    embedding_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    # path = glob.glob("html/*/*")
    # path.sort()
    # path = pd.DataFrame({"path": path})
    # path["date"] = path.path.str.extract(r"([0-9]{4}-[0-9]{2}-[0-9]{2})")

    date = args.date
    if date is None:
        # date = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        pdf_path = get_folders("pdf")
        pdf_path = pdf_path[pdf_path.date > start]
        embedding_path = get_folders("inference_chroma")
        embedding_path = embedding_path[embedding_path.date > start]
        embedding_path["exist"] = 1
        path = pdf_path.merge(
            embedding_path[["date", "symbol", "exist"]],
            on=["date", "symbol"],
            how="left",
        )
        path = path[path.exist.isna()]
    else:
        path = path[path.date >= date]
    for i in path.iterrows():
        # print(i[1]['path'])
        start_time = time.time()
        _, symbol, ar_date = i[1]["path"].split("/")
        save_path = os.path.join(args.save_directory, symbol, ar_date)

        if os.path.exists(save_path):
            continue
        save_index(args.save_directory, embedding_model, symbol, ar_date, config_dict)
        print(f"Completed: {symbol}, {ar_date} in {round(time.time()-start_time, 2)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        dest="config_path",
        type=str,
        required=True,
        help="""Full path of config.json""",
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        default="/root/GPT-InvestAR/inference_chroma",
        required=True,
        help="""absolute path of chroma""",
    )
    parser.add_argument("--date", type=str, default=None, help="the starting date")
    main(args=parser.parse_args())
