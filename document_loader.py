# document_loader.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings import HuggingFaceEmbedding

def build_and_save_index():
    documents = SimpleDirectoryReader("data").load_data()
    #embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir="storage")

if __name__ == "__main__":
    build_and_save_index()

