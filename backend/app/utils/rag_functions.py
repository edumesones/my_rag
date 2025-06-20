"""DSPy functions."""

import os
from typing import List, Any

import openai
from chromadb import PersistentClient
from app.utils.models import MessageData, RAGResponse, QAList
from app.utils.load import get_chroma_client, get_embedding

# Configurar la API Key de OpenAI
def _init_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("La variable OPENAI_API_KEY no está definida.")
    openai.api_key = api_key

# Listar modelos de OpenAI disponibles
def get_list_openai_models() -> List[str]:
    _init_openai()
    resp = openai.models.list()
    return [m.id for m in resp.data]

# Consulta zero-shot: solo prompt directo al LLM
def get_zero_shot_query(
    query: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
) -> RAGResponse:
    _init_openai()
    chat_resp = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    answer = chat_resp.choices[0].message.content
    return RAGResponse(answer=answer, context=[])  # context vacío en zero-shot

def retrieve_from_all_collections(query: str, k: int = 5) -> List[str]:
    # Validar que k sea un entero válido
    if k is None or not isinstance(k, int) or k <= 0:
        k = 5
        print(f"⚠️ k inválido, usando valor por defecto: {k}")
    """Buscar en TODAS las colecciones disponibles."""
    client = get_chroma_client()
    q_emb = get_embedding(query)
    
    all_results = []
    collections = client.list_collections()
    
    print(f"🔍 Buscando en {len(collections)} colecciones...")
    
    for collection_info in collections:
        try:
            collection = client.get_collection(name=collection_info.name)
            print(f"  - Colección '{collection_info.name}': {collection.count()} documentos")
            
            results = collection.query(
                query_embeddings=[q_emb], 
                n_results=k
            )
            
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            for doc, dist, meta in zip(documents, distances, metadatas):
                all_results.append({
                    'document': doc,
                    'distance': dist,
                    'metadata': meta,
                    'collection': collection_info.name
                })
                
        except Exception as e:
            print(f"Error en colección {collection_info.name}: {e}")
            continue
    
    # Ordenar por distancia y tomar los mejores k
    all_results.sort(key=lambda x: x['distance'])
    documents = [result['document'] for result in all_results[:k]]
    
    print(f"✅ Encontrados {len(documents)} documentos relevantes")
    return documents
# Consulta RAG compilada: recuperación + generación
def get_compiled_rag(
    query: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    k: int = 5,
) -> RAGResponse:
    _init_openai()
    # 1. Recuperar documentos relevantes
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "mis_docs"))
    q_emb = get_embedding(query)
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    docs: List[str] = results.get("documents", [[]])[0]

    if not docs:
        return RAGResponse(
            answer="No se encontraron documentos relevantes en la base de datos.", 
            context=[]
        )

    # 2. Construir prompt con contexto
    context_str = "\n".join(docs)
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

    # 3. Llamar a OpenAI con prompt enriquecido
    chat_resp = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    answer = chat_resp.choices[0].message.content
    return RAGResponse(answer=answer, context=docs)

# Compilar un batch de preguntas en RAG

def compile_rag(
    items: List[MessageData],
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    k: int = 5,
) -> List[RAGResponse]:
    responses: List[RAGResponse] = []
    for item in items:
        resp = get_compiled_rag(
            query=item.query,
            model_name=item.ollama_model_name or model_name,
            temperature=item.temperature,
            top_p=item.top_p,
            max_tokens=item.max_tokens,
            k=k,
        )
        responses.append(resp)
    return responses
def retrieve_only(query: str, k: int = 5) -> List[str]:
    """Solo recuperar documentos: devolver k respuestas más relevantes."""
    # Validar que k sea un entero válido
    if k is None or not isinstance(k, int) or k <= 0:
        k = 5
        print(f"⚠️ retrieve_only: k inválido, usando valor por defecto: {k}")
    
    docs = retrieve_from_all_collections(query, k)
    return docs  # ← List[str], NO RAGResponse