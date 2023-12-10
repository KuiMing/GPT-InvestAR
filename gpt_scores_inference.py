import pandas as pd
import sys
import argparse
import os
import json
import pickle
import glob
import time
from datetime import datetime
import openai
from dotenv import load_dotenv
from llama_index import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index import ServiceContext, LangchainEmbedding
from llama_index.vector_stores import ChromaVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from llama_index.prompts import Prompt
from gpt_scores_as_features import *

CHUNK_SIZE = 512
CHUNK_OVERLAP = 32

def get_gpt_generated_feature_dict(query_engine, feature_name, question):
    '''
    Returns:
        A dictionary with keys as question identifiers and value as GPT scores.
    '''
    response_dict = {}
#     for feature_name, question in questions_dict.items():
        #Sleep for a short duration, not to exceed openai rate limits.
    time.sleep(1)
    try:
        response = query_engine.query(question)
    except:
        time.sleep(60)
        response_dict = get_gpt_generated_feature_dict(query_engine, feature_name, question)
    try:
        response_dict[feature_name] = int(response.response.split(":")[-1].replace("}" ,""))
    except:
        response_dict = get_gpt_generated_feature_dict(query_engine, feature_name, question)
            
    return response_dict

def main(args):
    with open(args.config_path) as json_file:
        config_dict = json.load(json_file)
    with open(args.questions_path) as json_file:
        questions_dict = json.load(json_file)
    
#     df_train, df_test = load_target_dfs(config_dict)
    llm, embedding_model = initialize_and_return_models(config_dict)
#     curr_series = df_train.loc[2]
#     symbol = curr_series['symbol']
#     ar_date = curr_series['report_date']
    embeddings_directory = "~/GPT-InvestAR/inference_chroma/"
    features_save_directory = "~/GPT-InvestAR/inference_feature/"
    path = glob.glob("inference_chroma/*/*/")
    path.sort()
    for ind, i in enumerate(path):
        symbol = i.split("/")[1]
        ar_date = i.split("/")[2]
        index = load_index(llm, embedding_model, embeddings_directory, symbol, ar_date)
        text_qa_template = get_systemprompt_template(config_dict)
        query_engine = load_query_engine(index, text_qa_template)
    #Get feature scores as dictionary
        gpt_feature_dict = dict()
        for feature_name, question in questions_dict.items():
            gpt_feature_dict.update(get_gpt_generated_feature_dict(query_engine, feature_name, question))
        gpt_feature_df = pd.DataFrame.from_dict(gpt_feature_dict, orient='index').T
        gpt_feature_df.columns = ['feature_{}'.format(c) for c in gpt_feature_df.columns]
        gpt_feature_df['meta_symbol'] = symbol
        gpt_feature_df['meta_report_date'] = ar_date
        if ind ==0 :
            gpt_feature_df.to_csv("inference_feature.csv", index=False)
        else:
            gpt_feature_df.to_csv("inference_feature.csv", index=False, mode='a', header=False)
    
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str,
                         default="config.json",
                        help='''Full path of config.json''')
    parser.add_argument('--questions_path', dest='questions_path', type=str,
                         default="questions.json",
                        help='''Full path of questions.json which contains the questions 
                        for asking to the LLM''')
    main(args=parser.parse_args())
 