# LocalConfluenceChat
AI tool that allows you to chat with a local export of your Confluence space.


Reguires the following Python Libraries:
os
time
math
pandas
transformers
sentence_transformers
openai == 0.28
numpy
bs4

Before running please:
1) Download and extract an HTML export of the Confluence space you would like to "chat" with.
https://confluence.atlassian.com/doc/export-content-to-word-pdf-html-and-xml-139475.html
2) Download the pytorch_model.bin file for all-mpnet-base-v2
https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main
3) Update the links to your Confluence export folder and to your TGWUI OpenAI API endpoint in the code before running.

If you wish to use an alternative embedding model change all instances of '/Models/all-mpnet-base-v2' to point to your new model folder.
