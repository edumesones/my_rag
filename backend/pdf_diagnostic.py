"""
Diagnostic script to test PDF processing functionality.
Run this to identify and fix PDF processing issues.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.append(str(backend_path))

def test_imports():
    """Test if all required imports work."""
    print("🔍 Testing imports...")
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF imported successfully")
    except ImportError as e:
        print(f"⚠️ PyMuPDF import failed: {e}")
        print("   This is optional but recommended for better PDF processing")
    
    try:
        from app.utils.load import DocumentProcessor, ChunkingStrategy
        print("✅ Load modules imported successfully")
    except ImportError as e:
        print(f"❌ Load modules import failed: {e}")
        return False
    
    return True

def test_pdf_processing():
    """Test PDF processing with sample files."""
    print("\n📄 Testing PDF processing...")
    
    # Look for PDF files in data directory
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("**/*.pdf"))
    
    if not pdf_files:
        print("⚠️ No PDF files found in data directory")
        print("   Please add a PDF file to data/ for testing")
        return
    
    from app.utils.load import DocumentProcessor
    
    for pdf_file in pdf_files[:2]:  # Test first 2 PDFs
        print(f"\n🔍 Testing: {pdf_file.name}")
        
        try:
            text = DocumentProcessor.extract_text(pdf_file)
            print(f"✅ Successfully extracted {len(text)} characters")
            print(f"   Preview: {text[:100]}...")
            
            if len(text.strip()) == 0:
                print("⚠️ Warning: No text extracted (might be image-based PDF)")
                
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")

def test_chunking():
    """Test text chunking functionality."""
    print("\n✂️ Testing text chunking...")
    
    try:
        from app.utils.load import TextChunker, ChunkingConfig, ChunkingStrategy
        
        sample_text = """
        This is a sample text for testing chunking functionality. 
        It contains multiple sentences and should be split properly.
        The chunking algorithm should handle this text correctly.
        This is another sentence to make the text longer.
        """
        
        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20
        )
        
        chunker = TextChunker(config)
        chunks = chunker.chunk_text(sample_text)
        
        print(f"✅ Successfully created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {chunk[:50]}...")
            
    except Exception as e:
        print(f"❌ Chunking test failed: {e}")

def test_embeddings():
    """Test embedding generation."""
    print("\n🧠 Testing embedding generation...")
    
    try:
        from app.utils.load import get_embedding
        
        test_text = "This is a test sentence for embedding generation."
        embedding = get_embedding(test_text)
        
        print(f"✅ Successfully generated embedding with {len(embedding)} dimensions")
        print(f"   Sample values: {embedding[:5]}...")
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        if "OPENAI_API_KEY" in str(e):
            print("   Make sure OPENAI_API_KEY is set in your environment")

def test_chromadb():
    """Test ChromaDB functionality."""
    print("\n🗄️ Testing ChromaDB...")
    
    try:
        from app.utils.load import get_chroma_client
        
        client = get_chroma_client()
        collections = client.list_collections()
        
        print(f"✅ ChromaDB connected successfully")
        print(f"   Found {len(collections)} collections: {[c.name for c in collections]}")
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")

def test_full_pipeline():
    """Test the complete RAG pipeline."""
    print("\n🔄 Testing full RAG pipeline...")
    
    try:
        from app.utils.load import RAGDataLoader, ChunkingConfig, EmbeddingConfig, ChunkingStrategy
        
        # Create test file
        test_file = Path("test_document.txt")
        test_content = """
        This is a test document for RAG pipeline testing.
        It contains information about artificial intelligence and machine learning.
        The document should be processed and stored in the vector database.
        """
        
        test_file.write_text(test_content)
        
        # Initialize RAG loader
        loader = RAGDataLoader(
            data_dir="data",
            db_path="data/chroma_db",
            chunking_config=ChunkingConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=200,
                chunk_overlap=50
            ),
            embedding_config=EmbeddingConfig(
                model="text-embedding-3-small",
                batch_size=10
            )
        )
        
        # Process the test file
        loader.load_single_file(test_file, "test_collection")
        print("✅ Full pipeline test successful")
        
        # Clean up
        test_file.unlink()
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        # Clean up on error
        if test_file.exists():
            test_file.unlink()

def run_diagnostics():
    """Run all diagnostic tests."""
    print("🔧 RAG System Diagnostic Tool")
    print("=" * 40)
    
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing dependencies.")
        return
    
    test_pdf_processing()
    test_chunking()
    test_embeddings()
    test_chromadb()
    test_full_pipeline()
    
    print("\n" + "=" * 40)
    print("🎯 Diagnostic complete!")
    print("\nIf any tests failed, check the error messages above.")
    print("Common issues:")
    print("- Missing OPENAI_API_KEY environment variable")
    print("- PDF files that are image-based (need OCR)")
    print("- Missing dependencies (pip install -r requirements.txt)")

if __name__ == "__main__":
    run_diagnostics()