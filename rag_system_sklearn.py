import os
import PyPDF2
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv
import click
from pathlib import Path
from typing import List, Dict

# Load environment variables
load_dotenv()


class LegalRAGSystem:
    def __init__(self, data_folder: str = "data"):
        """
        Initialize the Legal RAG System with scikit-learn
        
        Args:
            data_folder: Path to folder containing PDF documents
        """
        self.data_folder = Path(data_folder)
        self.db_path = Path("vector_db")
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize the embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI client for Swiss AI
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("SWISS_AI_PLATFORM_API_KEY"),
            base_url="https://api.swisscom.com/layer/swiss-ai-weeks/apertus-70b/v1"
        )
        
        # Initialize search components
        self.nn_model = None
        self.embeddings = None
        self.documents = []
        self.metadata = []
        
        # Try to load existing database
        self.load_database()
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                # Try to break at sentence boundary
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
                elif last_newline > start + chunk_size // 2:
                    end = last_newline + 1
                else:
                    # Fall back to word boundary
                    last_space = text.rfind(' ', start, end)
                    if last_space > start + chunk_size // 2:
                        end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks
    
    def save_database(self):
        """Save the embeddings and metadata to disk"""
        if self.embeddings is not None:
            # Save embeddings, documents, and metadata
            np.save(self.db_path / "embeddings.npy", self.embeddings)
            
            with open(self.db_path / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(self.db_path / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
            
            print("Database saved to disk.")
    
    def load_database(self):
        """Load existing embeddings and metadata from disk"""
        try:
            embeddings_path = self.db_path / "embeddings.npy"
            docs_path = self.db_path / "documents.pkl"
            meta_path = self.db_path / "metadata.pkl"
            
            if all(path.exists() for path in [embeddings_path, docs_path, meta_path]):
                # Load embeddings, documents, and metadata
                self.embeddings = np.load(embeddings_path)
                
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                
                # Create NearestNeighbors model
                if len(self.embeddings) > 0:
                    self.nn_model = NearestNeighbors(n_neighbors=min(10, len(self.embeddings)), 
                                                   metric='cosine')
                    self.nn_model.fit(self.embeddings)
                
                print(f"Loaded existing database with {len(self.documents)} documents")
            else:
                # Initialize empty structures
                self.embeddings = None
                self.documents = []
                self.metadata = []
                self.nn_model = None
        except Exception as e:
            print(f"Error loading database: {e}")
            # Initialize empty structures
            self.embeddings = None
            self.documents = []
            self.metadata = []
            self.nn_model = None
    
    def load_documents(self):
        """Load all PDF documents from the data folder into the vector database"""
        print(f"Loading documents from {self.data_folder}...")
        
        pdf_files = list(self.data_folder.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in the data folder!")
            return
        
        # Clear existing data
        self.documents = []
        self.metadata = []
        all_embeddings = []
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            if not text.strip():
                continue
            
            # Split into chunks
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                # Create embedding
                embedding = self.embedding_model.encode(chunk)
                
                # Store data
                self.documents.append(chunk)
                self.metadata.append({
                    "source": pdf_file.name,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
                all_embeddings.append(embedding)
        
        # Create embeddings array and fit model
        if all_embeddings:
            self.embeddings = np.array(all_embeddings)
            
            # Create NearestNeighbors model
            self.nn_model = NearestNeighbors(n_neighbors=min(10, len(self.embeddings)), 
                                           metric='cosine')
            self.nn_model.fit(self.embeddings)
            
            # Save to disk
            self.save_database()
            
            print(f"Successfully loaded {len(self.documents)} chunks from {len(pdf_files)} PDF files.")
        else:
            print("No content was extracted from the PDF files.")
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents using cosine similarity"""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Create embedding for the query
        query_embedding = self.embedding_model.encode([query])
        
        # Use NearestNeighbors to find similar documents
        n_results = min(n_results, len(self.documents))
        distances, indices = self.nn_model.kneighbors(query_embedding, n_neighbors=n_results)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            similarity_score = 1 - distance  # Convert distance to similarity
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity_score': float(similarity_score)
            })
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate an answer using the Swiss AI API with retrieved context"""
        # Combine context documents
        context = "\n\n".join([doc['document'] for doc in context_docs])
        
        # Create the prompt
        system_prompt = """You are a legal assistant specializing in Swiss law. Provide detailed, structured answers that explain legal concepts thoroughly.

For each legal entity or concept mentioned:
1. Provide a clear definition
2. Include specific article references (e.g., "art. 532 CO (RS 220)")
3. Explain key characteristics, requirements, and implications
4. Compare different options when relevant
5. Give practical recommendations

Always cite the specific legal articles and codes (CO = Code of Obligations, etc.) throughout your explanation. Structure your answer with clear paragraphs for each option or concept discussed."""

        user_prompt = f"""Context from legal documents:
{context}

Question: {query}

Provide a detailed, structured answer that explains each legal concept thoroughly. Include specific article references (e.g., "art. 532 CO (RS 220)") throughout your explanation. If comparing multiple options, dedicate a paragraph to each one, explaining its definition, key characteristics, requirements, and practical implications. End with practical recommendations based on the context."""

        try:
            response = self.openai_client.chat.completions.create(
                model="swiss-ai/Apertus-70B",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"
    
    def ask_question(self, query: str) -> str:
        """Main method to ask a question and get an answer"""
        print(f"\nSearching for relevant documents...")
        
        # Search for relevant documents
        relevant_docs = self.search_similar_documents(query, n_results=5)
        
        if not relevant_docs:
            return "No relevant documents found in the database."
        
        print(f"Found {len(relevant_docs)} relevant document chunks.")
        print("Generating answer...")
        
        # Generate answer using Swiss AI
        answer = self.generate_answer(query, relevant_docs)
        
        return answer


@click.group()
def cli():
    """Legal RAG System CLI - Ask questions about Swiss legal documents"""
    pass


@cli.command()
@click.option('--data-folder', default='data', help='Path to folder containing PDF documents')
def load(data_folder):
    """Load PDF documents into the vector database"""
    rag = LegalRAGSystem(data_folder=data_folder)
    rag.load_documents()


@cli.command()
@click.argument('question')
@click.option('--data-folder', default='data', help='Path to folder containing PDF documents')
def ask(question, data_folder):
    """Ask a question about the loaded legal documents"""
    rag = LegalRAGSystem(data_folder=data_folder)
    
    # Check if we have documents
    if len(rag.documents) == 0:
        print("No documents found in the database. Please run 'load' command first.")
        return
    
    answer = rag.ask_question(question)
    print(f"\nAnswer:\n{answer}")


@cli.command()
def interactive():
    """Start an interactive Q&A session"""
    rag = LegalRAGSystem()
    
    # Check if we have documents
    if len(rag.documents) == 0:
        print("No documents found in the database. Please run 'load' command first.")
        return
    
    print("Legal RAG System - Interactive Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("-" * 50)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        answer = rag.ask_question(question)
        print(f"\nAnswer:\n{answer}")
        print("-" * 50)


if __name__ == "__main__":
    cli()
