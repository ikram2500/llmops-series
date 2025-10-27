# LLMOps Series - RAG Implementation

A comprehensive project demonstrating Retrieval-Augmented Generation (RAG) implementation using LangChain, OpenAI, and FAISS vector database.

## 📋 Overview

This project is part of the LLMOps series, focusing on building a practical RAG (Retrieval-Augmented Generation) system. It demonstrates how to process PDF documents, create embeddings, store them in a vector database, and use them for intelligent question-answering.

## 🚀 Features

- **PDF Document Processing**: Load and parse PDF files using PyPDFLoader
- **Text Chunking**: Intelligent text splitting with RecursiveCharacterTextSplitter
- **Vector Embeddings**: Generate embeddings using OpenAI's embedding models
- **Vector Storage**: Store and retrieve embeddings using FAISS vector database
- **Interactive Notebook**: Jupyter notebook for step-by-step demonstration

## 📁 Project Structure

```
llmops-series/
├── data/                          # Directory for PDF documents
│   └── Physics 9th Ch 1 Final.pdf
├── notebook/                      # Jupyter notebooks
│   └── rag.ipynb                 # Main RAG implementation notebook
├── main.py                       # Main application entry point
├── pyproject.toml                # Project dependencies and metadata
├── .env                          # Environment variables (not in repo)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🛠️ Tech Stack

- **Python**: 3.12+
- **LangChain**: Framework for LLM applications
- **OpenAI**: Embeddings and language models
- **FAISS**: Facebook AI Similarity Search for vector storage
- **PyPDF**: PDF processing
- **python-dotenv**: Environment variable management
- **UV**: Fast Python package installer

## 📦 Dependencies

Core dependencies include:
- `langchain-community` >= 0.4
- `langchain-openai`
- `openai`
- `pypdf` >= 6.1.3
- `faiss-cpu`
- `tiktoken`
- `rapidocr-onnxruntime` >= 1.4.4
- `ipykernel` >= 7.0.1

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ikram2500/llmops-series.git
   cd llmops-series
   ```

2. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Install dependencies**
   
   Using UV (recommended):
   ```bash
   uv pip install -r pyproject.toml
   ```
   
   Or using pip:
   ```bash
   pip install langchain-community langchain-openai openai pypdf faiss-cpu tiktoken rapidocr-onnxruntime ipykernel
   ```

## 📖 Usage

### Running the Jupyter Notebook

1. Navigate to the notebook directory:
   ```bash
   cd notebook
   ```

2. Open the notebook:
   ```bash
   jupyter notebook rag.ipynb
   ```

3. Follow the cells sequentially to:
   - Load environment variables
   - Install required packages
   - Ingest PDF documents
   - Split text into chunks
   - Create embeddings
   - Store in FAISS vector database
   - Query the RAG system

### Key Components

#### 1. Data Ingestion
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('path/to/your/document.pdf')
documents = loader.load()
```

#### 2. Text Splitting
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
text_chunks = text_splitter.split_documents(documents)
```

#### 3. Embeddings & Vector Store
```python
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(text_chunks, embeddings)
```

## 🎯 Use Cases

- Educational content Q&A system
- Document search and retrieval
- Knowledge base creation
- Semantic search implementation
- RAG system prototyping

## 🔐 Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and completions

## 📝 Notes

- The project currently uses a Physics textbook (9th grade, Chapter 1) as sample data
- Chunk size and overlap can be adjusted based on your specific use case
- FAISS provides efficient similarity search for the vector embeddings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Ikram**
- GitHub: [@ikram2500](https://github.com/ikram2500)

## 🙏 Acknowledgments

- LangChain for the amazing framework
- OpenAI for embeddings and LLM capabilities
- Facebook AI for FAISS vector database

---

**Note**: This is part of an educational series on LLM Operations (LLMOps). Stay tuned for more projects!
