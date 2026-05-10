# 📄 PDF Chat with RAG

> Ask questions about any PDF using Retrieval-Augmented Generation (RAG) powered by GPT-4 and FAISS.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?style=flat-square)

---

## 🧠 How It Works

```
PDF Upload → Text Extraction → Chunking → OpenAI Embeddings → FAISS Vector Store
                                                                        ↓
User Question → Embed Question → Retrieve Top-K Chunks → GPT-4 → Answer
```

1. **PDF Parsing** — Extracts text from any PDF using PyPDF
2. **Chunking** — Splits text into overlapping chunks (1000 tokens, 200 overlap) using `RecursiveCharacterTextSplitter`
3. **Embedding** — Each chunk is embedded using `text-embedding-ada-002`
4. **Vector Store** — Embeddings stored in FAISS for fast similarity search
5. **Retrieval** — Top-4 most relevant chunks retrieved for each question
6. **Generation** — GPT-4o-mini synthesizes a grounded answer from retrieved context

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/pramod019g-oss/pdf-chat-rag.git
cd pdf-chat-rag
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🗂️ Project Structure

```
pdf-chat-rag/
├── app.py                  # Streamlit UI
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py     # Core RAG logic
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-ada-002 |
| Vector Store | FAISS (Facebook AI) |
| Orchestration | LangChain |
| PDF Parsing | PyPDF |

---

## 💡 Key Concepts Demonstrated

- **RAG (Retrieval-Augmented Generation)** — grounding LLM answers in real document content
- **Vector similarity search** — finding semantically relevant chunks, not just keyword matches
- **Chunking strategy** — overlapping chunks to avoid missing context at boundaries
- **Prompt engineering** — structured prompt to keep answers grounded and prevent hallucination

---

## 🔮 Future Improvements

- [ ] Support multiple PDFs at once
- [ ] Add conversation memory (multi-turn chat)
- [ ] Export chat history as PDF
- [ ] Deploy to Streamlit Cloud / AWS
- [ ] Add support for DOCX, TXT, CSV files
- [ ] Switch to open-source embeddings (sentence-transformers)

---

## 👤 Author

**Pramod Gangula** — AI Engineer  
[GitHub](https://github.com/pramod019g-oss) 

---

## 📄 License

MIT License — feel free to use and modify.
