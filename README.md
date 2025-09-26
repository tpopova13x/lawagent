# Legal RAG System

A Retrieval Augmented Generation (RAG) system for Swiss legal documents using the Swiss AI Platform.

## Features

- 📄 **PDF Processing**: Automatically extracts text from PDF documents in the `data` folder
- 🔍 **Semantic Search**: Uses vector embeddings to find relevant document sections
- 🤖 **AI-Powered Answers**: Leverages Swiss AI Platform (Apertus-70B) for generating comprehensive legal answers
- 💾 **Persistent Storage**: Uses scikit-learn with vector embeddings for efficient document storage and retrieval
- 💬 **Interactive CLI**: Easy-to-use command-line interface

## Setup

1. **Environment Variables**: Make sure your `.env` file contains:
   ```
   SWISS_AI_PLATFORM_API_KEY=your_api_key_here
   TOKENIZERS_PARALLELISM=false
   ```
   
   The `TOKENIZERS_PARALLELISM=false` setting prevents tokenizer parallelism warnings.

2. **Install Dependencies**: (Already installed in your virtual environment)
   ```bash
   # If you need to reinstall dependencies
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Add Documents**: Place your PDF legal documents in the `data` folder

## Important Note

**Always activate your virtual environment before running any commands:**
```bash
source .venv/bin/activate
```

Or use the full path to the Python executable:
```bash
/Users/tpopova/Documents/GitHub/Lawagent/.venv/bin/python rag_system_sklearn.py load
```

## Usage

### Method 1: Simple Interactive CLI
```bash
# Activate virtual environment first
source .venv/bin/activate
python legal_cli.py
```

This will:
- Automatically load documents from the `data` folder if needed
- Start an interactive Q&A session
- Allow you to ask questions about the legal documents

### Method 2: Advanced CLI with Commands

#### First, activate the virtual environment:
```bash
source .venv/bin/activate
```

#### Load documents into the database:
```bash
python rag_system_sklearn.py load
```

#### Ask a single question:
```bash
python rag_system_sklearn.py ask "How does the Swiss Civil Procedure Code apply to legal disputes involving startups?"
```

#### Start interactive session:
```bash
python rag_system_sklearn.py interactive
```

## Example Questions

- "How does the Swiss Civil Procedure Code apply to legal disputes involving startups?"
- "What are the procedural rules for filing a claim in Swiss courts?"
- "What costs and risks should startups consider in Swiss legal proceedings?"
- "How does the conciliation process work in Swiss civil procedure?"

## System Architecture

1. **Document Processing**: PDFs are extracted and chunked into manageable pieces
2. **Vector Database**: Text chunks are embedded using SentenceTransformer and stored with scikit-learn's NearestNeighbors
3. **Retrieval**: When you ask a question, the system finds the most relevant document chunks using cosine similarity
4. **Generation**: The Swiss AI API generates a comprehensive answer based on the retrieved context

## Files

- `rag_system_sklearn.py`: Main RAG system implementation with advanced CLI
- `legal_cli.py`: Simple interactive CLI interface
- `request_apertus.py`: Original Swiss AI API test script
- `data/`: Folder containing PDF legal documents
- `vector_db/`: Vector database storage (created automatically)
- `.env`: Environment variables (API keys)
- `requirements.txt`: Python dependencies

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` for creating document embeddings
- **Vector Database**: scikit-learn's NearestNeighbors with cosine similarity
- **AI Model**: Swiss AI Platform Apertus-70B model
- **Text Chunking**: Intelligent chunking with overlap for better context retention
# lawagent
