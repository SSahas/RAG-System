from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def load_vector_db(path = "/data/chroma_db"):
    embed_function = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={'trust_remote_code': True}
    )
    
    loaded_db = Chroma(
        persist_directory=path,
        embedding_function=embed_function
    )
    
    return loaded_db













        










