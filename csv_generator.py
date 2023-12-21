import os
import time
import math
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from bs4 import BeautifulSoup

#Get working directory
dir = os.getcwd()

# Set the API base and key
openai.api_base = '<TGWUI-OPENAI-API-ENDPOINT-URL>'
openai.api_key = "..."

# Set Tokenizer
tokenizer = AutoTokenizer.from_pretrained(f'{dir}/Models/all-mpnet-base-v2')

def get_doc_model():
    '''
    Model string to calculate the embeddings.
    '''
    return (f'{dir}/Models/all-mpnet-base-v2')

def get_embeddings(text: str) -> list[float]:
    model = SentenceTransformer(get_doc_model())
    embedding = model.encode(text).tolist()
    return {"data": [{"embedding": embedding}]}
    
def get_max_num_tokens():
    return 1054
#Change based on your model's context length. Should be equivalent to the number in internal_do_chatbot.py

path = '<PATH-TO-CONFLUENCE-EXPORT-FOLDER' #Folder must be the unzipped export and files must be .html files
pages = []
query_model = get_doc_model()
collect = []
for file_path in os.listdir(path):
    if file_path.endswith('.html'):
        with open(os.path.join(path, file_path), 'rb') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            title = soup.find('title').text
            body = soup.find('div', {'id': 'main-content'}).text
            pages.append({'title': title, 'body': body})
for page in pages:
    title = page['title']
    htmlbody = page['body']
    htmlParse = BeautifulSoup(htmlbody, 'html.parser')
    body = []
    for text in htmlParse.find_all(text=True, recursive=False):
        bodyText = text.get_text()
        if bodyText != "":
            # Calculate number of tokens
            body = ''.join(bodyText)
            body = body.replace("\t", " ").replace("\r", " ").replace("\n", " ")
            tokens = len(tokenizer.tokenize(body))
            if tokens >= get_max_num_tokens():
                splits = math.ceil(tokens/get_max_num_tokens())
                bodyLength = len(body)
                chunkSize = bodyLength//splits
                remainder = bodyLength%splits
                chunks = [body[i * chunkSize:(i+1) * chunkSize] for i in range(splits)]
                if remainder:
                    chunks[-1] += body[-remainder:]
                for i, chunk in enumerate(chunks):
                    tokens = len(tokenizer.tokenize(chunk))
                    body = chunk
                    print(tokens)
                    collect += [(title, body, tokens)]
            else:
                print(tokens)
                collect += [(title, body, tokens)]
DOC_title_content_embeddings = pd.DataFrame(collect, columns=['title', 'body', 'num_tokens'])
# Caculate the embeddings
## Create a checkpoint of the embeddings in the event that the embedding functions below fail
DOC_title_content_embeddings.to_csv(f'{dir}/DOC_title_content_embeddings_chkpt.csv', index=False)
## Limit to pages with less than get_max_num_tokens tokens
DOC_title_content_embeddings = DOC_title_content_embeddings[DOC_title_content_embeddings.num_tokens<=get_max_num_tokens()]
DOC_title_content_embeddings['embedding'] = DOC_title_content_embeddings.body.apply(lambda x: get_embeddings(x)['data'][0]['embedding'])
DOC_title_content_embeddings.to_csv(f'{dir}/DOC_title_content_embeddings.csv', index=False)
