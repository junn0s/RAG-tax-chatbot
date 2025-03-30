# 🦈 Income Tax RAG Chatbot

An income tax Q&A chatbot powered by OpenAI, LangChain, and ChromaDB.

Ask questions using natural language through the Streamlit UI, and the chatbot will provide concise answers based on relevant sections from the income tax law document.

---

## Features

- Retrieval-Augmented Generation (RAG) based on the income tax law document (`tax_with_markdown.docx`).
- Semantic retrieval of the top 4 relevant law sections based on your query.
- Answer generation powered by OpenAI's GPT-4o-mini.
- Simple web-based chat interface created with Streamlit.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/junn0s/RAG-tax-chatbot.git
cd RAG-tax-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Set your OpenAI API key
```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```
### 4. Run the Streamlit app
```bash
streamlit run chat.py
```
