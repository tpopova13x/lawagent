import io
import os
from pypdf import PdfReader, PdfWriter
from typing import List, Dict, Any
class RAGDocument:
    """A standardized object to hold document content and source metadata."""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return (f"RAGDocument(pages={self.metadata.get('page_count', 'N/A')}, "
                f"source='{self.metadata.get('source', 'N/A')}', "
                f"chars={len(self.page_content)})")
    
def load_pdf_to_rag_documents(pdf_file_buffer: io.BytesIO, source_name: str) -> List[RAGDocument]:
    """
    Implements the RAG Document Loading step for a single PDF file using pypdf.

    Args:
        pdf_file_buffer: An in-memory buffer containing the PDF file data.
        source_name: The descriptive name/path of the source file.

    Returns:
        A list of RAGDocument objects, one for each page extracted.
    """
    rag_documents = []

    try:
        # 1. Document Loading: Open the file using PdfReader
        reader = PdfReader(pdf_file_buffer)
        
        # Extract general metadata from the PDF (Document Object metadata)
        doc_metadata = reader.metadata
        
        for i, page in enumerate(reader.pages):
            # 2. Text Extraction: Extract raw text from the page
            page_content = page.extract_text() or ""
            
            # 3. Create the Document Object
            # The 'metadata' object is enriched with page-specific info
            metadata = {
                "source": source_name,
                "page_number": i + 1,
                "page_count": len(reader.pages),
                "title": doc_metadata.title,
                "author": doc_metadata.author,
                "subject": doc_metadata.subject,
                "source_type": "PDF"
            }
            
            # Create the standardized RAGDocument object
            doc = RAGDocument(
                page_content=page_content, 
                metadata=metadata
            )
            rag_documents.append(doc)

    except Exception as e:
        print(f"ERROR: Failed to process PDF {source_name}: {e}")
        return []
        
    return rag_documents

import re
def clean_document_text(documents: List[RAGDocument]) -> List[RAGDocument]:
    """
    Performs data cleaning and preprocessing on a list of RAGDocument objects.

    This function implements the three key activities: Standardization, 
    Noise Removal, and basic structural cleanup.

    Args:
        documents: A list of RAGDocument objects loaded from the source.

    Returns:
        The same list of RAGDocument objects with cleaned page_content.
    """
    cleaned_documents = []

    # --- Common Noise Patterns ---
    # 1. Regex for repeating headers/footers (e.g., 'Page 1 of 5 - Confidential')
    # This pattern simulates finding a common header structure often repeated on every page.
    HEADER_FOOTER_PATTERN = re.compile(
        r"(page \d+ of \d+ - confidential|disclaimer: this document is legally binding\.)",
        re.IGNORECASE | re.DOTALL
    )

    # 2. Regex for excessive whitespace/newlines (structural cleanup)
    # Replaces multiple newlines/spaces with a single space.
    WHITESPACE_PATTERN = re.compile(r'\s{2,}', re.DOTALL)


    for doc in documents:
        raw_text = doc.page_content

        # -----------------------------------------------
        # STEP 1: Standardization
        # -----------------------------------------------
        # Convert all text to lowercase for consistent embedding representation.
        # This prevents 'Sales' and 'sales' from being treated as completely different words.
        standardized_text = raw_text.lower()
        
        # Note: Normalizing unicode characters (e.g., é to e) can be added here
        # using the 'unicodedata' module if international characters are present.
        
        # -----------------------------------------------
        # STEP 2: Noise Removal
        # -----------------------------------------------
        
        # Remove known headers/footers and boilerplate text
        # This is highly application-specific and depends on the document type.
        noise_removed_text = HEADER_FOOTER_PATTERN.sub('', standardized_text)
        
        # -----------------------------------------------
        # STEP 3: Structural Cleanup (Basic Parsing Support)
        # -----------------------------------------------
        # Reduce excessive whitespace and clean up line breaks resulting from PDF extraction
        cleaned_text = WHITESPACE_PATTERN.sub(' ', noise_removed_text).strip()
        
        # Update the document object with the cleaned text
        doc.page_content = cleaned_text
        cleaned_documents.append(doc)

    return cleaned_documents

from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Choose your embedding model
# all-MiniLM-L6-v2 is fast and efficient for initial testing.
MODEL_NAME = "all-MiniLM-L6-v2"

# 2. Initialize the embedding model instance
# Setting device="cpu" is common, but you could use "cuda" if you have a GPU
embeddings_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)

print(f"Embedding model '{MODEL_NAME}' loaded successfully.")



# Example: Aggregate all pages for a single document (VC or Startup)
def aggregate_cleaned_text(cleaned_documents: List[RAGDocument]) -> str:
    return " ".join(doc.page_content for doc in cleaned_documents)

def get_embedding(text: str, embeddings_model) -> list:
    # HuggingFaceEmbeddings expects a list of texts
    return embeddings_model.embed_documents([text])[0]

import numpy as np

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def similarity_to_score(similarity: float) -> int:
    # Assumes similarity in [0, 1]
    return int(similarity * 100)


def apply_hard_filters(vc_metadata, startup_metadata) -> bool:
    # Example: Check stage and geography
    if vc_metadata.get("stage") != startup_metadata.get("stage"):
        return False
    if startup_metadata.get("location") not in vc_metadata.get("locations", []):
        return False
    # Add more rules as needed
    return True


def match_vcs_and_startups(vc_list, startup_list, embeddings_model):
    results = []
    for vc in vc_list:
        vc_text = aggregate_cleaned_text(vc['documents'])
        vc_vec = get_embedding(vc_text, embeddings_model)
        for startup in startup_list:
            startup_text = aggregate_cleaned_text(startup['documents'])
            startup_vec = get_embedding(startup_text, embeddings_model)
            sim = cosine_similarity(vc_vec, startup_vec)
            score = similarity_to_score(sim)
            eligible = apply_hard_filters(vc['metadata'], startup['metadata'])
            if not eligible:
                score = 10  # or "Ineligible"
            results.append({
                "vc": vc['metadata']['name'],
                "startup": startup['metadata']['name'],
                "score": score,
                "eligible": eligible
            })
    # Sort by score descending
    return sorted(results, key=lambda x: x['score'], reverse=True)


def extract_common_keywords(vc_text, startup_text, top_n=5):
    vc_words = set(vc_text.split())
    startup_words = set(startup_text.split())
    common = vc_words & startup_words
    return list(common)[:top_n]

def display_matches(matches, vc_list, startup_list):
    for match in matches:
        print(f"VC: {match['vc']} | Startup: {match['startup']} | Score: {match['score']} | Eligible: {match['eligible']}")
        if match['eligible']:
            vc = next(vc for vc in vc_list if vc['metadata']['name'] == match['vc'])
            startup = next(s for s in startup_list if s['metadata']['name'] == match['startup'])
            vc_text = aggregate_cleaned_text(vc['documents'])
            startup_text = aggregate_cleaned_text(startup['documents'])
            print("  Common themes:", extract_common_keywords(vc_text, startup_text))
        print("-" * 40)



# Mock RAGDocument creation
vc1_docs = [RAGDocument("We invest in early-stage AI startups in Europe.", {"page_number": 1})]
vc2_docs = [RAGDocument("Our focus is on climate tech and sustainability, Series A, US.", {"page_number": 1})]
startup1_docs = [RAGDocument("Our AI platform helps automate logistics for European companies.", {"page_number": 1})]
startup2_docs = [RAGDocument("We build solar panels for sustainable energy in California.", {"page_number": 1})]

# Mock VC and Startup metadata
vc_list = [
    {"metadata": {"name": "AI Ventures", "stage": "Seed", "locations": ["Europe"]}, "documents": vc1_docs},
    {"metadata": {"name": "Green Future Fund", "stage": "Series A", "locations": ["US"]}, "documents": vc2_docs}
]

startup_list = [
    {"metadata": {"name": "LogiAI", "stage": "Seed", "location": "Europe"}, "documents": startup1_docs},
    {"metadata": {"name": "SolarNow", "stage": "Series A", "location": "US"}, "documents": startup2_docs}
]

matches = match_vcs_and_startups(vc_list, startup_list, embeddings_model)
display_matches(matches, vc_list, startup_list)