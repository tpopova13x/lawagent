# Legal RAG System

A Retrieval Augmented Generation (RAG) system for Swiss legal documents using the Swiss AI Platform, extended with modules for **VC–Startup Matching** and **Startup Team Size Prediction**.

---

## Features

* 📄 **PDF Processing**: Automatically extracts text from PDF documents in the `data` folder
* 🔍 **Semantic Search**: Uses vector embeddings to find relevant document sections
* 🤖 **AI-Powered Answers**: Leverages Swiss AI Platform (Apertus-70B) for generating comprehensive legal answers
* 💾 **Persistent Storage**: Uses scikit-learn with vector embeddings for efficient document storage and retrieval
* 💬 **Interactive CLI**: Easy-to-use command-line interface
* 🏢 **VC–Startup Matching (Startup_VC.py)**: Matches venture capital firms and startups based on document embeddings, metadata, and eligibility filters
* 👥 **Employee Prediction (Employee_prediction.py)**: Predicts startup team sizes using machine learning on structured datasets

---

## Setup

### Environment Variables

Make sure your `.env` file contains:

```
SWISS_AI_PLATFORM_API_KEY=your_api_key_here
TOKENIZERS_PARALLELISM=false
```

The `TOKENIZERS_PARALLELISM=false` setting prevents tokenizer parallelism warnings.

### Install Dependencies

(Already installed in your virtual environment)

```bash
# If you need to reinstall dependencies
source .venv/bin/activate
pip install -r requirements.txt
```

### Add Documents

Place your PDF legal documents in the `data` folder.

---

## Usage

### Method 1: Simple Interactive CLI

```bash
# Activate virtual environment first
source .venv/bin/activate
python legal_cli.py
```

This will:

* Automatically load documents from the data folder if needed
* Start an interactive Q&A session
* Allow you to ask questions about the legal documents

### Method 2: Advanced CLI with Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Load documents into the database
python rag_system_sklearn.py load

# Ask a single question
python rag_system_sklearn.py ask "How does the Swiss Civil Procedure Code apply to legal disputes involving startups?"

# Start interactive session
python rag_system_sklearn.py interactive
```

---

## Additional Modules

### 1. VC–Startup Matching (`Startup_VC.py`)

This module provides functionality for matching startups with venture capital firms based on document embeddings, metadata, and eligibility filters.

**Key Features:**

* Loads and cleans PDF documents into `RAGDocument` objects
* Embeds and compares startup/VC profiles using cosine similarity
* Applies hard filters (e.g., stage, geography)
* Extracts common themes/keywords between startups and VCs
* Displays ranked matches with eligibility status

**Example (mock data included in script):**

```bash
python Startup_VC.py
```

Output includes VC–Startup matches, scores, eligibility, and common themes.

---

### 2. Employee Prediction (`Employee_prediction.py`)

This module predicts the **team size** of startups using structured data and a machine learning model.

**Key Features:**

* Loads dataset (`startup_stage_dataset.csv`)
* Preprocesses features (scaling, one-hot encoding)
* Trains a **Random Forest Regressor** to predict startup team size
* Evaluates model using RMSE, MAE, and R²
* Provides feature importance insights
* Predicts team size for a new startup with given metrics

**Example:**

```bash
python Employee_prediction.py
```

Output includes:

* Model training and evaluation results
* Top features driving predictions
* Predicted team size for a sample startup profile

---

## Example Questions for Legal RAG

* "How does the Swiss Civil Procedure Code apply to legal disputes involving startups?"
* "What are the procedural rules for filing a claim in Swiss courts?"
* "What costs and risks should startups consider in Swiss legal proceedings?"
* "How does the conciliation process work in Swiss civil procedure?"

---

## System Architecture

* **Document Processing**: PDFs are extracted and chunked into manageable pieces
* **Vector Database**: Text chunks embedded with `SentenceTransformer` and stored using scikit-learn's `NearestNeighbors`
* **Retrieval**: System finds the most relevant chunks using cosine similarity
* **Generation**: Swiss AI API generates a comprehensive answer from retrieved context
* **VC–Startup Matching**: Embedding-based similarity + metadata filters (stage, geography)
* **Employee Prediction**: ML-based regression (Random Forest) on structured startup metrics

---

## Files

* `rag_system_sklearn.py`: Main RAG system implementation with advanced CLI
* `legal_cli.py`: Simple interactive CLI interface
* `request_apertus.py`: Original Swiss AI API test script
* `Startup_VC.py`: VC–Startup matching system (embeddings + filters)
* `Employee_prediction.py`: Startup team size prediction using ML
* `data/`: Folder containing PDF legal documents
* `vector_db/`: Vector database storage (created automatically)
* `.env`: Environment variables (API keys)
* `requirements.txt`: Python dependencies

---

## Technical Details

* **Embedding Model**: `all-MiniLM-L6-v2` for creating document embeddings
* **Vector Database**: scikit-learn `NearestNeighbors` with cosine similarity
* **AI Model**: Swiss AI Platform `Apertus-70B`
* **Text Chunking**: Intelligent chunking with overlap for better context retention
* **VC–Startup Matching**: HuggingFace embeddings + cosine similarity + eligibility filters
* **Employee Prediction**: Random Forest Regressor with feature scaling and one-hot encoding
