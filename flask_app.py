import os
from flask import Flask, request, render_template
import internal_doc_chatbot
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, 'templates'))
history = []
           
@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        text_input = request.form['text_input']
        text_output = process_text(text_input, history)
        print(text_output)
        history.append({"role": "user", "content": text_input})
        history.append({"role": "assistant", "content": text_output})
        return render_template('index.html', text_output=text_output, text_input=text_input)
    return render_template('index.html')

def parse_numbers(s):
  return [float(x) for x in s.strip('[]').split(',')]

def return_Confluence_embeddings():

    # Current file where the embeddings of our internal Confluence document is saved
    Confluence_embeddings_file = os.path.join(ROOT_DIR, 'DOC_title_content_embeddings.csv')
    DOC_title_content_embeddings= pd.read_csv(Confluence_embeddings_file, dtype={'embedding': object})
    DOC_title_content_embeddings['embedding'] = DOC_title_content_embeddings['embedding'].apply(lambda x: parse_numbers(x))
 
    return DOC_title_content_embeddings
        
def process_text(query, history):
    
    DOC_title_content_embeddings= return_Confluence_embeddings()
    output = internal_doc_chatbot.internal_doc_chatbot_answer(query, DOC_title_content_embeddings, history)
    
    return output

if __name__ == '__main__':
    app.run()
