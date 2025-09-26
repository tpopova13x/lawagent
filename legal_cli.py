#!/usr/bin/env python3
"""
Legal RAG System CLI
A simple command-line interface for the Legal RAG system
"""

from rag_system_sklearn import LegalRAGSystem

def main():
    print("🏛️  Legal RAG System")
    print("=" * 50)
    
    # Initialize the RAG system
    rag = LegalRAGSystem()
    
    # Check if we have documents loaded
    try:
        count = len(rag.documents)
        print(f"📚 Database contains {count} document chunks")
        
        if count == 0:
            print("\n❌ No documents found in the database!")
            print("Loading documents from 'data' folder...")
            rag.load_documents()
            count = len(rag.documents)
            print(f"✅ Loaded {count} document chunks")
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        print("Loading documents from 'data' folder...")
        rag.load_documents()
    
    # Interactive Q&A loop
    print("\n💬 You can now ask questions about Swiss legal matters!")
    print("Type 'quit' or 'exit' to end the session")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🔍 Searching for relevant information...")
            answer = rag.ask_question(question)
            print(f"\n📖 Answer:\n{answer}")
            print("\n" + "=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
