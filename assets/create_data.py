import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

loader = PyPDFLoader(
    file_path = "/assets/iesc111.pdf",

)

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",model_kwargs = {'trust_remote_code': True}), 
    persist_directory="vector_db",  
)


    
