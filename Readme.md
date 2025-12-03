# 📄 Chat with Doc

**Chat with Doc** is an AI-powered application that allows you to upload PDF documents and interact with them conversationally.
Using **RAG (Retrieval-Augmented Generation)**, the app extracts text from PDFs, embeds it, performs vector similarity search using FAISS, and answers questions using an LLM — all inside a clean Streamlit interface.

> ⚡ **Inspired by the original "Chat with your PDF" project by Mesut Duman.**
> 🙏 Credits: [mesutdmn / Chat-With-Your-PDF](https://github.com/mesutdmn/Chat-With-Your-PDF)

---

## 🚀 Features

- 📄 Upload one or multiple PDF files
- 🔍 Automatic text extraction & chunking
- 🧠 Embedding + FAISS vector search
- 🤖 LLM-powered question answering
- 🧭 LangGraph routing (memory vs vectorstore)
- 💬 Easy-to-use Streamlit interface

---

## 🛠️ Tech Stack

- **Streamlit** for UI
- **LangChain / LangGraph** for pipeline orchestration
- **FAISS** for vector indexing
- **OpenAI Embeddings + Chat Models**
- **PyPDFLoader** for PDF parsing

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/chat-with-doc.git
cd chat-with-doc
pip install -r requirements.txt
streamlit run app.py

# Original author : mesutdmn
# github : https://github.com/mesutdmn/Chat-With-Your-PDF
