from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.core import VectorStoreIndex


def get_query_engine():
    llm = LlamaCPP(
        model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.7,
        context_window=4096,
        max_new_tokens=512,
        generate_kwargs={"stop": ["</s>"]},
    )

    storage_context = StorageContext.from_defaults(persist_dir="storage")
    #embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #index = VectorStoreIndex.from_documents(storage_context, embed_model=embed_model)

    index = load_index_from_storage(storage_context,embed_model=embed_model)
    return index.as_query_engine(llm=llm)
