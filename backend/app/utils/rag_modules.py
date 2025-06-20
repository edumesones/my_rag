"""DSPy modules."""
import dspy
from dspy import Signature, Module, Retrieve, ChainOfThought

# Definición de firma para respuesta generada
GenerateAnswer = Signature(
    name="GenerateAnswer",
    inputs={"context": list[str], "question": str},
    outputs={"answer": str}
)

# Módulo de recuperación (vector store)
class RAGRetrieve(Module):
    def __init__(self, k: int = 5, collection_names: list[str] = None, search_all: bool = True):
        self.k = k
        self.collection_names = collection_names or ["default"]
        self.search_all = search_all  # Si True, busca en todas las colecciones

    def forward(self, question: str) -> list[str]:
        from app.utils.rag_functions import retrieve_from_all_collections
        
        print(f"🔍 RAGRetrieve buscando: '{question}'")
        docs = retrieve_from_all_collections(question, self.k)
        print(f"✅ RAGRetrieve encontró {len(docs)} documentos")
        return docs

# Módulo de generación de respuesta con contexto
class RAGGenerate(Module):
    def forward(self, context: list[str], question: str) -> str:
        import openai, os
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not context:
            return "No se encontró contexto relevante para responder la pregunta."
        
        context_str = "\n".join(context)
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
        
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[{"role": "system", "content": prompt}],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.0)),
            top_p=float(os.getenv("OPENAI_TOP_P", 1.0)),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 512)),
        )
        return resp.choices[0].message.content

# Programa RAG que une recuperación y generación
RAG = dspy.Chain(
    Retrieve=RAGRetrieve,
    Generate=RAGGenerate,
    signature=GenerateAnswer,
)