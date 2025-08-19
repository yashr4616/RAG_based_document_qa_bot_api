# 📄 Document QA API

This project is a **FastAPI-based Document Question-Answering API** powered by **LangChain**, **OpenAI embeddings**, and **FAISS** for vector search.  
It allows you to upload or fetch documents (PDF, DOCX, EML, MSG), process them, and ask questions directly against the document content.  

---

## 🚀 Features
- ✅ Upload & parse **PDF, Word (.docx), Emails (.eml, .msg)**  
- ✅ Splits documents into chunks using **RecursiveCharacterTextSplitter**  
- ✅ Creates embeddings with **OpenAI’s `text-embedding-3-small`**  
- ✅ Stores vectors in **FAISS** for similarity search  
- ✅ Secure API with **Bearer Token authentication**  
- ✅ Optimized for **parallel processing** with async & thread pools  
- ✅ Provides **short, precise answers** (Yes/No for binary questions)  

---

## 🛠️ Tech Stack
- **FastAPI** – API framework  
- **LangChain** – Orchestration & document loaders  
- **OpenAI** – Embeddings + LLM (Chat model)  
- **FAISS** – Vector database  
- **aiohttp** – Async document fetching  
- **PyMuPDF / Unstructured** – Document parsing  

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yashr4616/RAG_based_document_qa_bot_api.git
cd document-qa-api
```

2. **Create & activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**  
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key_here
```

---

## ▶️ Run the Server
```bash
uvicorn model:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at:  
👉 `http://localhost:8000`

---

## 🔑 Authentication
The API uses **Bearer Token authentication**.  
Default token is defined in `model.py` as:
```python
API_TOKEN = "f799dd3c9ae79667d28623cf53c3683e115c2ebb26fff88fafc7bc55225c70d1"
```

Include it in headers:
```http
Authorization: Bearer f799dd3c9ae79667d28623cf53c3683e115c2ebb26fff88fafc7bc55225c70d1
```

---

## 📌 API Endpoints

### 1. **Run Document QA**
`POST /rag_qa_bot`

#### Request Body
```json
{
  "documents": "https://example.com/sample.pdf",
  "questions": [
    "What is the patient name?",
    "Is the diagnosis positive?"
  ]
}
```

#### Response
```json
{
  "answers": [
    "John Doe",
    "Yes"
  ]
}
```

---

## 📂 Project Structure
```
.
├── model.py              # Main FastAPI app
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```


