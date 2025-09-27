from flask import Flask, render_template, request, jsonify
import os
import sys
from rag_system_sklearn import LegalRAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the RAG system
print("Initializing Legal RAG System...")
rag_system = LegalRAGSystem()

# Check if we have documents loaded
if len(rag_system.documents) == 0:
    print("No documents found. Loading documents...")
    rag_system.load_documents()
    print(f"Loaded {len(rag_system.documents)} document chunks")
else:
    print(f"Using existing database with {len(rag_system.documents)} document chunks")

@app.route('/')
def home():
    """Main page with the question form"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question submission"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        print(f"Processing question: {question}")
        
        # Get answer from RAG system
        answer = rag_system.ask_question(question)
        
        return jsonify({
            'question': question,
            'answer': answer
        })
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Legal RAG Web Interface...")
    print("Open your browser and go to: http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=8000)
