"""
example code for sementic chunking and RAG
"""

import os
import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.llms import AzureOpenAI  # pylint: disable=no-name-in-module
from langchain import PromptTemplate  # pylint: disable=no-name-in-module
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from dotenv import load_dotenv
import sys

__import__("pysqlite3")

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def get_chain(llm, template: str, variables, verbose: bool = False):
    """
    Get Chain
    """
    prompt_template = PromptTemplate(
        template=template,
        input_variables=variables,
    )
    return load_qa_with_sources_chain(
        llm=llm,
        prompt=prompt_template,
        verbose=verbose,
    )


def main():
    """
    main function
    """
    config_dict = json.load(open("config.json", "r", encoding="utf8"))
    question = json.load(open("questions.json", "r", encoding="utf8"))
    os.environ["OPENAI_API_KEY"] = config_dict["openai_api_key"]
    os.environ["OPENAI_API_VERSION"] = config_dict["openai_api_version"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = config_dict["azure_openai_endpoint"]
    llm = AzureOpenAI(deployment_name="gpt-35-turbo-intruct", temperature=0.5)

    raw_documents = PyPDFLoader("pdf/A/2002-01-22/2002-01-22.pdf").load()

    text_splitter = SemanticChunker(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    documents = text_splitter.split_documents(raw_documents)
    # pylint: disable=no-member
    db = Chroma.from_documents(
        documents,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    )
    docs = db.similarity_search(question["feature_competition_1"])

    load_dotenv()

    chain = get_chain(
        llm,
        template="""Context information is below.
        =========
        {summaries}
        =========
        Given the context information, 
        please assign a confidence score ranging from 0 to 100. 
        Give the answer in json format with only one key that is: 'score'. 
        Answer the question:""",
        variables=["summaries"],
    )

    answer = chain.run(input_documents=docs, question=question["feature_competition_1"])
    print(answer)
