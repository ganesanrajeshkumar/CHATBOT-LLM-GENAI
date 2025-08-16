#chatbot with Mistral LLM 

✅ 3. Folder Structure
chatbot1/
├── main.py
├── chatbot.py
├── docs/                  # Upload your .txt, .pdf, .md, etc here
├── chatbot_env/           # Your Python venv


✅ 4. Install Requirements
Make sure you're in your virtual environment (chatbot_env) and then install:

pip install fastapi uvicorn transformers torch llama-index sentence-transformers

If you get permission errors on Windows:

pip install --user fastapi uvicorn transformers torch llama-index sentence-transformers

uvicorn main:app --reload


✅ 5. Start the Chatbot API Server

uvicorn main:app --reload


🧠 How to Train It
Place .txt, .md, or .pdf files in the data/ folder manually OR use:

curl -X POST -F 'file=@path/to/file.txt' http://localhost:8000/upload
Then query it:

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is in the file?"}'
