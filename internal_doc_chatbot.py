import os
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
import requests
from bs4 import BeautifulSoup

#Get working directory
dir = os.getcwd()

# Set the API base and key
openai.api_base = '<TGWUI-OPENAI-API-ENDPOINT-URL>'
openai.api_key = "..."

# Set Tokenizer
tokenizer = AutoTokenizer.from_pretrained(f'{dir}/Models/all-mpnet-base-v2')

def get_max_num_tokens():
    return 1540
  
def get_doc_model():
    '''
    Model string to calculate the embeddings.
    '''
    return (f'{dir}/Models/all-mpnet-base-v2')

def get_embeddings(text: str) -> list[float]:
    model = SentenceTransformer(f'{dir}/Models/all-mpnet-base-v2')
    embedding = model.encode(text).tolist()
    return {"data": [{"embedding": embedding}]}

#Experimental data pre-processing
def openai_complete(chosen_sections):
    COMPLETIONS_MODEL = "..."
    header = """Below is information that is going to be going to the user. Please summarize the text using one to three paragraphs. Make every word count. This means making space by combining points and removing pointless information. The summaries should be dense and easily understood without losing the original intent of the text. Here is the text for you to process: """
    prompt = header + "".join(chosen_sections)
    response = openai.Completion.create(
        prompt=prompt,
        temperature=1,
        max_tokens=512,
        top_p=1,
        min_p=0.02,
        repeat_penalty=1,
        model=COMPLETIONS_MODEL
    )
    output = response["choices"][0]["text"].strip(" \n")
    print(output)
    return output
  
def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))
  
def order_document_sections_by_query_similarity(query: str, doc_embeddings: pd.DataFrame):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embeddings(query)
    doc_embeddings['similarity'] = doc_embeddings['embedding'].apply(lambda x: vector_similarity(x, query_embedding['data'][0]['embedding']))
    doc_embeddings.sort_values(by='similarity', inplace=True, ascending=False)
    doc_embeddings.reset_index(drop=True, inplace=True)
    
    return doc_embeddings
  
def construct_prompt(query, doc_embeddings):
    
    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = "\n* "
    separator_len = len(tokenizer.tokenize(SEPARATOR))
    
    chosen_sections = []
    chosen_sections_len = 0
     
    for section_index in range(len(doc_embeddings)):
        # Add contexts until we run out of space.        
        document_section = doc_embeddings.loc[section_index]
        print(document_section.title)
        chosen_sections_len += document_section.num_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.body.replace("\n", " "))
    ##Experimental context pre-processing
    ##context = openai_complete(chosen_sections)
    header = """You are a helpful and cheerful personal assistant AI. You are assisting users on the HALO Program. You respond to the User as best as possible using the provided context contained in the "Context" below. DO NOT RESPOND AS THE USER. \n\nContext:\n"""
    #prompt = header + context + "\n\n User: " + query + "\n\n Assistant: "
    prompt = header + "".join(chosen_sections) + "\n\n User: " + query + "\n\n Assistant: "
    print(prompt)
    return (prompt)
  
def internal_doc_chatbot_answer(query, DOC_title_content_embeddings, history):
    
    # Order docs by similarity of the embeddings with the query
    DOC_title_content_embeddings = order_document_sections_by_query_similarity(query, DOC_title_content_embeddings)
    # Construct the prompt
    prompt = construct_prompt(query, DOC_title_content_embeddings)
    # Ask the question with the context to ChatGPT
    COMPLETIONS_MODEL = "..."
    title = DOC_title_content_embeddings
    response = openai.ChatCompletion.create(
        prompt=prompt,
        temperature=1,
        max_tokens=512,
        top_p=1,
        min_p=0.02,
        messages=history,
        model=COMPLETIONS_MODEL
    )
    #print(response)
    output = response["choices"][0]["message"]["content"].strip(" \n")
    
    return output
  
