"""
Enhanced RAG Data Loader with PDF and TXT support.

This module provides a robust data loading system for RAG applications,
supporting multiple file formats with configurable chunking strategies.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import openai
import chromadb
from dotenv import load_dotenv
import PyPDF2
# import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
data_dir = os.getenv("DATA_DIR", "data")


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    FIXED_SIZE = "fixed_size"


class FileType(Enum):
    """Supported file types."""
    TXT = "txt"
    PDF = "pdf"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_tokens: int = 8191


class DocumentProcessor:
    """Handles document processing and text extraction."""
    
    @staticmethod
    def extract_text_from_txt(file_path: Path) -> str:
        """Extract text from a TXT file."""
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    return file_path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {file_path}")
    
    @staticmethod
    def extract_text_from_pdf_pypdf2(file_path: Path) -> str:
        """Extract text from PDF using PyPDF2."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 failed for {file_path}: {e}")
            raise
        return text
    
    @staticmethod
    def extract_text_from_pdf_pymupdf(file_path: Path) -> str:
        """Extract text from PDF using PyMuPDF (better for complex PDFs)."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {file_path}: {e}")
            raise
        return text
    
    @classmethod
    def extract_text_from_pdf(cls, file_path: Path) -> str:
        """Extract text from PDF with fallback methods."""
        try:
            # Try PyMuPDF first (generally better)
            return cls.extract_text_from_pdf_pymupdf(file_path)
        except Exception:
            logger.info(f"PyMuPDF failed for {file_path}, trying PyPDF2...")
            try:
                return cls.extract_text_from_pdf_pypdf2(file_path)
            except Exception as e:
                logger.error(f"All PDF extraction methods failed for {file_path}: {e}")
                raise
    
    @classmethod
    def extract_text(cls, file_path: Path) -> str:
        """Extract text from supported file formats."""
        file_type = FileType(file_path.suffix.lower().lstrip('.'))
        
        if file_type == FileType.TXT:
            return cls.extract_text_from_txt(file_path)
        elif file_type == FileType.PDF:
            return cls.extract_text_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")


class TextChunker:
    """Handles text chunking with different strategies."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_by_sentence(self, text: str) -> List[str]:
        """Chunk text by sentences."""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed token limit
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if self._count_tokens(potential_chunk) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def chunk_recursively(self, text: str) -> List[str]:
        """Chunk text using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=self._count_tokens
        )
        return splitter.split_text(text)
    
    def chunk_fixed_size(self, text: str) -> List[str]:
        """Chunk text into fixed-size pieces."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_tokens = self._count_tokens(word)
            if current_size + word_tokens > self.config.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_tokens
            else:
                current_chunk.append(word)
                current_size += word_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text based on configured strategy."""
        if self.config.strategy == ChunkingStrategy.SENTENCE:
            return self.chunk_by_sentence(text)
        elif self.config.strategy == ChunkingStrategy.RECURSIVE:
            return self.chunk_recursively(text)
        elif self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            return self.chunk_fixed_size(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")


class EmbeddingGenerator:
    """Handles embedding generation with OpenAI API."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _validate_text(self, text: str) -> str:
        """Validate and truncate text if necessary."""
        token_count = self._count_tokens(text)
        if token_count > self.config.max_tokens:
            logger.warning(f"Text truncated from {token_count} to {self.config.max_tokens} tokens")
            tokens = self.encoding.encode(text)[:self.config.max_tokens]
            return self.encoding.decode(tokens)
        return text
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            validated_batch = [self._validate_text(text) for text in batch]
            
            try:
                response = self.client.embeddings.create(
                    input=validated_batch,
                    model=self.config.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//self.config.batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                raise
        
        return embeddings


class ChromaDBManager:
    """Manages ChromaDB operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
    
    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        try:
            return self.client.get_or_create_collection(name=name)
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            raise
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to a ChromaDB collection."""
        collection = self.get_or_create_collection(collection_name)
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
            raise


class RAGDataLoader:
    """Main class for loading data into RAG system."""
    
    def __init__(
        self,
        data_dir: str,
        db_path: str,
        chunking_config: Optional[ChunkingConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        self.data_dir = Path(data_dir)
        self.processor = DocumentProcessor()
        self.chunker = TextChunker(chunking_config or ChunkingConfig())
        self.embedding_generator = EmbeddingGenerator(embedding_config or EmbeddingConfig())
        self.db_manager = ChromaDBManager(db_path)
        
        # Validate OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def load_single_file(
        self,
        file_path: Union[str, Path],
        collection_name: str = "default"
    ) -> None:
        """Load a single file into the RAG system."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing file: {file_path}")
        
        # Extract text
        text = self.processor.extract_text(file_path)
        logger.info(f"Extracted {len(text)} characters from {file_path}")
        
        # Chunk text
        chunks = self.chunker.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        
        # Prepare metadata
        metadatas = [
            {
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        
        # Store in ChromaDB
        import time
        timestamp = int(time.time())
        ids = [f"{file_path.stem}_{timestamp}_{i}" for i in range(len(chunks))]
        self.db_manager.add_documents(
            collection_name=collection_name,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def load_directory(
        self,
        directory_path: Union[str, Path],
        collection_name: str = "default",
        file_extensions: Optional[List[str]] = None
    ) -> None:
        """Load all supported files from a directory."""
        directory_path = Path(directory_path)
        print(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if file_extensions is None:
            file_extensions = [".txt", ".pdf"]
        
        files = []
        for ext in file_extensions:
            files.extend(directory_path.glob(f"*{ext}"))
        
        if not files:
            logger.warning(f"No supported files found in {directory_path}")
            return
        
        logger.info(f"Found {len(files)} files to process")
        
        for file_path in files:
            try:
                self.load_single_file(file_path, collection_name)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Completed processing directory: {directory_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load documents into RAG system")
    parser.add_argument("--file", "-f", type=str, help="Single file to process")
    parser.add_argument("--directory", "-d", type=str, help="Directory to process")
    parser.add_argument("--collection", "-c", type=str, default="default", help="Collection name")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--strategy", type=str, default="recursive", 
                       choices=["sentence", "recursive", "fixed_size"], help="Chunking strategy")
    
    args = parser.parse_args()
    
    # Configure chunking
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy(args.strategy),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Initialize loader
    db_path = os.path.join(args.data_dir, "chroma_db")
    loader = RAGDataLoader(
        data_dir=args.data_dir,
        db_path=db_path,
        chunking_config=chunking_config
    )
    
    # Process files
    if args.file:
        loader.load_single_file(args.file, args.collection)
    elif args.directory:
        loader.load_directory(args.directory, args.collection)
    else:
        logger.error("Either --file or --directory must be specified")

def get_chroma_client():
    return chromadb.PersistentClient(path=f"{data_dir}/chroma_db")
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Función standalone para obtener embedding de un texto usando OpenAI.
    Mantiene compatibilidad con código existente.
    """
    try:
        # Intentar usar el cliente nuevo
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except AttributeError:
        # Fallback al cliente legacy
        response = openai.Embedding.create(
            input=[text],
            model=model
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

if __name__ == "__main__":
    main()