# RagChat  
Retrieval-Augmented Generation (RAG) chatbot that combines **OpenAI embeddings** with **vector search** to deliver context-aware responses.  

---

## 🚀 Overview  
RagChat is an intelligent chatbot that retrieves relevant document sections before generating an answer using **GPT-3.5 Turbo**.  
It uses **OpenAI’s Ada v2 embeddings** to encode queries and documents, storing them in a **vector database (FAISS)** for efficient similarity search.  

This project demonstrates how **LLMs can be enhanced with external knowledge** while remaining fast, modular, and easy to extend.  

---

## ✨ Features  
- **Retrieval-Augmented Generation** – Queries are matched against relevant document chunks.  
- **OpenAI embeddings (Ada v2)** – High-quality semantic search.  
- **GPT-3.5 Turbo integration** – Generates context-aware, coherent responses.  
- **Vector database with FAISS** – Efficient similarity search at scale.  
- **Extensible** – Swap in other embedding models or LLMs.  

---

## 🧩 Architecture  

```text
User Query ──> Embedding (OpenAI Ada v2) ──> Vector Search (FAISS) ──> Top-K Documents
           └──────────────────────────────────────────────┘
                                │
                                v
                     Context + Query ──> GPT-3.5 Turbo ──> Response
```

---

## 🛠️ Tech Stack  
- **Languages**: Python  
- **Libraries**: OpenAI API, FAISS, LangChain (optional)  
- **Model**: GPT-3.5 Turbo  
- **Storage**: FAISS (vector database)  

---

## ⚡ Quickstart  

1. Clone the repository:  
```bash
git clone https://github.com/mshaheeralam/ragchat.git
cd ragchat
```

2. Install dependencies:  
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:  
```bash
export OPENAI_API_KEY=your_api_key_here
```

4. Run the chatbot:  
```bash
python app.py
```

---

## 📂 Suggested Repository Structure  
```
ragchat/
  ├─ data/              # Documents to embed
  ├─ embeddings/        # Scripts for generating embeddings
  ├─ vectorstore/       # FAISS index storage
  ├─ app.py             # Chatbot entry point
  ├─ requirements.txt
  └─ README.md
```

---

## 🔮 Future Improvements  
- Add support for other LLMs (Claude, LLaMA, etc.).  
- Containerize with Docker for reproducible deployment.  
- Build a lightweight web UI for interaction.  
