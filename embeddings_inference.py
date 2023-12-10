import pandas as pd
from embeddings_save import *
import glob
import time
from datetime import datetime

with open('config.json') as json_file:
    config_dict = json.load(json_file)
path = glob.glob("html/*")
path.sort()
save_directory = "~/GPT-InvestAR/inference_chroma"

embedding_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
for i in path:
    start_time = time.time()
    _, symbol = i.split('/')
    path_html = glob.glob(f"html/{symbol}/*")
    path_html.sort()
    _, symbol, ar_date = path_html[-1].split('/')
    save_path = os.path.join(save_directory, symbol, ar_date)
    if int(ar_date[:4]) < datetime.now().year:
        continue
    if os.path.exists(save_path):
        continue
    save_index(save_directory, embedding_model, 
               symbol, ar_date, config_dict)
    print("Completed: {}, {} in {:.2f}s".format(symbol, ar_date, time.time()-start_time))