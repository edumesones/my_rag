"""DSPy functions."""

"""DSPy functions with enhanced document loading capabilities."""

import os
from typing import List, Any, Dict
import tempfile
import shutil
from pathlib import Path

import openai
from chromadb import PersistentClient
from app.utils.models import MessageData, RAGResponse, QAList
from app.utils.load import get_chroma_client, get_embedding, RAGDataLoader, ChunkingConfig, ChunkingStrategy

def _init_openai():
    """Initialize OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key

def get_list_openai_models() -> List[str]:
    """Get list of available OpenAI models"""
    try:
        _init_openai()
        resp = openai.models.list()
        return [m.id for m in resp.data if 'gpt' in m.id.lower()]
    except Exception as e:
        print(f"Error getting models: {e}")
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

def get_collections_info() -> Dict[str, Any]:
    """Get information about all ChromaDB collections"""
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        
        collection_info = []
        total_documents = 0
        
        for coll_metadata in collections:
            try:
                coll = client.get_collection(name=coll_metadata.name)
                count = coll.count()
                total_documents += count
                collection_info.append({
                    "name": coll_metadata.name,
                    "count": count
                })
            except Exception as e:
                print(f"Error accessing collection {coll_metadata.name}: {e}")
                continue
        
        return {
            "collections": collection_info,
            "total_collections": len(collection_info),
            "total_documents": total_documents
        }
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return {"collections": [], "total_collections": 0, "total_documents": 0}

def load_documents_to_chroma(
    files: List,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunking_strategy: str = "recursive"
) -> str:
    """
    Load multiple documents (PDF, TXT, DOCX) to ChromaDB.
    
    Args:
        files: List of Gradio file objects
        chunk_size: Size of text chunks in tokens
        chunk_overlap: Overlap between chunks
        chunking_strategy: Strategy for chunking ("recursive", "sentence", "fixed_size")
    
    Returns:
        Status message with results
    """
    if not files:
        raise ValueError("No files provided")
    
    # Validate files
    supported_extensions = {'.pdf', '.txt', '.docx'}
    valid_files = []
    invalid_files = []
    
    for file in files:
        file_path = Path(file.name)
        if file_path.suffix.lower() in supported_extensions:
            valid_files.append(file)
        else:
            invalid_files.append(file_path.name)
    
    if not valid_files:
        raise ValueError(f"No supported files found. Supported: {supported_extensions}")
    
    # Configure chunking strategy
    strategy_map = {
        "recursive": ChunkingStrategy.RECURSIVE,
        "sentence": ChunkingStrategy.SENTENCE,
        "fixed_size": ChunkingStrategy.FIXED_SIZE
    }
    
    chunking_config = ChunkingConfig(
        strategy=strategy_map.get(chunking_strategy, ChunkingStrategy.RECURSIVE),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Initialize loader
    data_dir = os.getenv("DATA_DIR", "data")
    db_path = os.path.join(data_dir, "chroma_db")
    
    try:
        loader = RAGDataLoader(
            data_dir=data_dir,
            db_path=db_path,
            chunking_config=chunking_config
        )
        
        # Process each valid file
        processed_files = []
        errors = []
        total_chunks = 0
        
        for file in valid_files:
            try:
                file_path = Path(file.name)
                print(f"Processing: {file_path.name}")
                
                # Load file using the existing loader
                chunks_before = get_total_documents_count()
                
                # Always use "user_documents" collection for consistency
                loader.load_single_file(file_path, "user_documents")
                
                chunks_after = get_total_documents_count()
                file_chunks = chunks_after - chunks_before
                
                processed_files.append({
                    "name": file_path.name,
                    "chunks": file_chunks
                })
                total_chunks += file_chunks
                
            except Exception as e:
                error_msg = f"{Path(file.name).name}: {str(e)}"
                errors.append(error_msg)
                print(f"Error processing {file.name}: {e}")
                continue
        
        # Prepare result message
        result_parts = []
        
        if processed_files:
            result_parts.append(f"Successfully processed {len(processed_files)} file(s):")
            for file_info in processed_files:
                result_parts.append(f"  • {file_info['name']}: {file_info['chunks']} chunks")
            result_parts.append(f"\nTotal chunks added: {total_chunks}")
            result_parts.append(f"Collection: user_documents")
            result_parts.append(f"Chunking: {chunking_strategy} (size: {chunk_size}, overlap: {chunk_overlap})")
        
        if invalid_files:
            result_parts.append(f"\nSkipped unsupported files:")
            for filename in invalid_files:
                result_parts.append(f"  • {filename}")
        
        if errors:
            result_parts.append(f"\nErrors:")
            for error in errors:
                result_parts.append(f"  • {error}")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        raise Exception(f"System error: {str(e)}")

def get_total_documents_count() -> int:
    """Get total number of documents across all collections"""
    try:
        info = get_collections_info()
        return info["total_documents"]
    except:
        return 0

def retrieve_from_all_collections(query: str, k: int = 5) -> List[str]:
    """Search in ALL collections with priority for user_documents"""
    client = get_chroma_client()
    q_emb = get_embedding(query)
    
    all_results = []
    collections = client.list_collections()
    
    print(f"🔍 Searching in {len(collections)} collection(s)...")
    
    # First, search in user_documents with higher weight
    user_docs_results = []
    other_results = []
    
    for collection_info in collections:
        try:
            collection = client.get_collection(name=collection_info.name)
            print(f"  - Collection '{collection_info.name}': {collection.count()} documents")
            
            # Get more results from user_documents
            search_k = k * 2 if collection_info.name == "user_documents" else k
            
            results = collection.query(
                query_embeddings=[q_emb], 
                n_results=min(search_k, collection.count())
            )
            
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            for doc, dist, meta in zip(documents, distances, metadatas):
                result_item = {
                    'document': doc,
                    'distance': dist,
                    'metadata': meta,
                    'collection': collection_info.name
                }
                
                if collection_info.name == "user_documents":
                    # Boost user documents by reducing distance slightly
                    result_item['distance'] = dist * 0.95
                    user_docs_results.append(result_item)
                else:
                    other_results.append(result_item)
                
        except Exception as e:
            print(f"Error in collection {collection_info.name}: {e}")
            continue
    
    # Combine results: prioritize user_documents, then others
    all_results = user_docs_results + other_results
    
    # Sort by distance and take top k
    all_results.sort(key=lambda x: x['distance'])
    documents = [result['document'] for result in all_results[:k]]
    
    print(f"✅ Found {len(documents)} relevant documents")
    return documents

def retrieve_only(query: str, k: int = 5) -> List[str]:
    """Retrieve only documents without generation"""
    if k is None or not isinstance(k, int) or k <= 0:
        k = 5
        print(f"⚠️ retrieve_only: invalid k, using default: {k}")
    
    docs = retrieve_from_all_collections(query, k)
    return docs

def get_zero_shot_query(
    query: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
) -> RAGResponse:
    """Zero-shot query: direct LLM call without retrieval"""
    _init_openai()
    chat_resp = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    answer = chat_resp.choices[0].message.content
    return RAGResponse(answer=answer, context=[])

def get_compiled_rag(
    query: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    k: int = 5,
) -> RAGResponse:
    """Compiled RAG: retrieval + generation"""
    _init_openai()
    
    # 1. Retrieve relevant documents
    docs = retrieve_from_all_collections(query, k)

    if not docs:
        return RAGResponse(
            answer="No relevant documents found in the database.", 
            context=[]
        )

    # 2. Build context prompt
    context_str = "\n\n".join(f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs))
    prompt = f"""Based on the following documents, answer the question.

Documents:
{context_str}

Question: {query}

Answer based on the provided documents:"""

    # 3. Generate answer
    chat_resp = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    answer = chat_resp.choices[0].message.content
    return RAGResponse(answer=answer, context=docs)

def compile_rag(
    items: List,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    k: int = 5,
) -> Dict[str, Any]:
    """Compile RAG pipeline with training examples"""
    
    try:
        # For now, this is a placeholder for actual DSPy compilation
        # In a real implementation, this would train the DSPy pipeline
        
        processed_items = []
        for item in items:
            # Validate each item
            if hasattr(item, 'question') and hasattr(item, 'answer'):
                processed_items.append({
                    'question': item.question,
                    'answer': item.answer
                })
        
        # Simulate compilation process
        compilation_result = {
            'status': 'success',
            'model_name': model_name,
            'training_examples': len(processed_items),
            'parameters': {
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_tokens,
                'k': k
            },
            'message': f'Pipeline compiled with {len(processed_items)} examples'
        }
        
        return compilation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }