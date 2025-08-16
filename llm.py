from llama_cpp import Llama

def get_llm():
    return Llama(
        model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=4096,
        n_threads=8,
        temperature=0.7,
        top_p=0.95,
        stop=["</s>", "User:", "Assistant:"],
    )
