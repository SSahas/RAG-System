# Retrieval augumented system web app

- A simple RAG web app which can be used as a document chatbot. The system uses semantic search algoritham for retriving the relevant text to the user query. 
- The system uses llm and prompt engineering techniques for better responses to the user query. streamlit is used for frontend.

## To run the model 
clone the repo
```
git clone https://github.com/SSahas/RAG-System
```
install all the requirements
```
pip install -r requirements.txt
```
To create the vector database of document.
```
python assets/create_data.py
```
Run streamlit app.
```
python app.py
```






