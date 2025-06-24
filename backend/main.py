# Monkey-patch para evitar error en gradio_client JSON schema
import gradio_client.utils as client_utils

# Guardamos las funciones originales ANTES de reemplazarlas
_original_json_schema_to_python_type = client_utils.json_schema_to_python_type
_original_internal = getattr(client_utils, '_json_schema_to_python_type', None)
_original_get_type = getattr(client_utils, 'get_type', None)

# Parcheamos la función interna si existe
if _original_internal:
    def safe_internal_json_schema_to_python_type(schema, defs=None):
        # Si el schema no es un diccionario, devolvemos "Any"
        if not isinstance(schema, dict):
            return "Any"
        
        # Si additionalProperties es bool, lo manejamos especialmente
        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], bool):
            # Creamos una copia del schema y reemplazamos el bool
            schema = schema.copy()
            if schema["additionalProperties"]:
                schema["additionalProperties"] = {}  # True -> objeto vacío
            else:
                schema.pop("additionalProperties")  # False -> lo removemos
        
        return _original_internal(schema, defs)
    
    # Reemplazamos la función interna
    client_utils._json_schema_to_python_type = safe_internal_json_schema_to_python_type

# Parcheamos la función principal
def safe_json_schema_to_python_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _original_json_schema_to_python_type(schema)

client_utils.json_schema_to_python_type = safe_json_schema_to_python_type

# Parcheamos get_type si existe
if _original_get_type:
    def safe_get_type(schema):
        if not isinstance(schema, dict):
            return None
        # Llamamos a la función original guardada, NO a la del módulo
        return _original_get_type(schema)
    
    client_utils.get_type = safe_get_type

import os
import logging
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from pathlib import Path

from app.api.routers.rag import rag_router
from app.gradio_ui.ui import gradio_iface
from instrument import instrument
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Instrumentación DSPy/OpenTelemetry
do_not_instrument = os.getenv("INSTRUMENT_DSPY", "true") == "false"
if not do_not_instrument:
    instrument()

# Crear FastAPI app
app = FastAPI(
    title="DSPy x FastAPI - Document RAG System", 
    description="RAG system with document upload capabilities",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configurar CORS
environment = os.getenv("ENVIRONMENT", "dev")
if environment == "dev":
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ===== INICIALIZACIÓN DE DIRECTORIOS PARA DOCUMENTOS =====
@app.on_event("startup")
async def startup_event():
    """Inicializar directorios y verificar dependencias al arrancar"""
    try:
        # Crear directorio de datos si no existe
        data_dir = os.getenv("DATA_DIR", "data")
        Path(data_dir).mkdir(exist_ok=True)
        Path(data_dir, "chroma_db").mkdir(exist_ok=True)
        Path(data_dir, "uploads").mkdir(exist_ok=True)
        logger.info(f"Data directories initialized at: {data_dir}")
        
        # Verificar dependencias críticas
        try:
            import PyPDF2
            logger.info("✅ PyPDF2 available for PDF processing")
        except ImportError:
            logger.warning("⚠️ PyPDF2 not installed - PDF support limited")
        
        try:
            import docx
            logger.info("✅ python-docx available for DOCX processing")
        except ImportError:
            logger.warning("⚠️ python-docx not installed - DOCX support disabled")
        
        # Verificar OpenAI API Key
        if os.getenv("OPENAI_API_KEY"):
            logger.info("✅ OpenAI API key configured")
        else:
            logger.warning("⚠️ OPENAI_API_KEY not set - embeddings will fail")
        
        # Verificar ChromaDB
        try:
            from app.utils.load import get_chroma_client
            client = get_chroma_client()
            collections = client.list_collections()
            logger.info(f"✅ ChromaDB initialized with {len(collections)} collections")
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {e}")
        
        logger.info("🚀 Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# ===== RUTAS PRINCIPALES =====
@app.get("/")
async def root(request: Request):
    """Página principal con información del sistema"""
    logger.info(f"Root access from IP: {request.client.host}")
    
    # Información del sistema
    try:
        from app.utils.rag_functions import get_collections_info
        db_info = get_collections_info()
        
        system_info = {
            "message": "DSPy x FastAPI Document RAG System is running",
            "status": "healthy",
            "environment": environment,
            "features": {
                "document_upload": "✅ PDF, TXT, DOCX support",
                "rag_queries": "✅ Zero-shot and compiled RAG",
                "document_retrieval": "✅ ChromaDB vector search",
                "pipeline_optimization": "✅ DSPy pipeline training"
            },
            "database": {
                "collections": db_info.get("total_collections", 0),
                "documents": db_info.get("total_documents", 0)
            },
            "endpoints": {
                "gradio_ui": "/gradio",
                "api_docs": "/docs",
                "health_check": "/health",
                "stats": "/stats"
            }
        }
        
        return system_info
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {
            "message": "DSPy x FastAPI Document RAG System", 
            "status": "running",
            "error": "Could not load system statistics"
        }

@app.get("/health")
async def health_check(request: Request):
    """Health check detallado"""
    logger.info(f"Health check from IP: {request.client.host}")
    
    health_status = {
        "status": "healthy",
        "environment": environment,
        "timestamp": __import__("time").time()
    }
    
    # Verificar componentes
    checks = {}
    
    # Check ChromaDB
    try:
        from app.utils.load import get_chroma_client
        client = get_chroma_client()
        collections = client.list_collections()
        checks["chromadb"] = {"status": "healthy", "collections": len(collections)}
    except Exception as e:
        checks["chromadb"] = {"status": "error", "error": str(e)}
    
    # Check OpenAI API
    try:
        if os.getenv("OPENAI_API_KEY"):
            checks["openai"] = {"status": "configured"}
        else:
            checks["openai"] = {"status": "warning", "message": "API key not set"}
    except Exception as e:
        checks["openai"] = {"status": "error", "error": str(e)}
    
    # Check data directories
    try:
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        checks["filesystem"] = {
            "status": "healthy" if data_dir.exists() else "warning",
            "data_dir": str(data_dir),
            "exists": data_dir.exists()
        }
    except Exception as e:
        checks["filesystem"] = {"status": "error", "error": str(e)}
    
    health_status["checks"] = checks
    
    # Determinar status general
    if any(check.get("status") == "error" for check in checks.values()):
        health_status["status"] = "unhealthy"
        return health_status
    elif any(check.get("status") == "warning" for check in checks.values()):
        health_status["status"] = "degraded"
    
    return health_status

# ===== ESTADÍSTICAS DEL SISTEMA =====
@app.get("/stats")
@limiter.limit("10/minute")  # Límite de consultas
async def get_stats(request: Request):
    """Estadísticas detalladas del sistema"""
    try:
        from app.utils.rag_functions import get_collections_info
        
        stats = {
            "system": {
                "uptime": "running",
                "environment": environment,
                "data_dir": os.getenv("DATA_DIR", "data")
            },
            "database": get_collections_info(),
            "capabilities": {
                "supported_formats": ["PDF", "TXT", "DOCX"],
                "chunking_strategies": ["recursive", "sentence", "fixed_size"],
                "embedding_models": ["text-embedding-3-small"],
                "llm_models": "OpenAI GPT family"
            }
        }
        
        logger.info(f"Stats requested from IP: {request.client.host}")
        return stats
        
    except ImportError:
        logger.warning("Stats module not fully available")
        return {"error": "Stats module not available", "basic_status": "running"}
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

# ===== ENDPOINT PARA INFORMACIÓN DE DOCUMENTOS =====
@app.get("/documents/info")
@limiter.limit("20/minute")
async def get_documents_info(request: Request):
    """Información sobre documentos cargados"""
    try:
        from app.utils.rag_functions import get_collections_info
        
        info = get_collections_info()
        logger.info(f"Document info requested from IP: {request.client.host}")
        
        return {
            "collections": info.get("collections", []),
            "total_collections": info.get("total_collections", 0),
            "total_documents": info.get("total_documents", 0),
            "primary_collection": "user_documents",
            "supported_formats": ["PDF", "TXT", "DOCX"]
        }
        
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving document information")

# ===== INCLUIR RUTAS DE LA API =====
app.include_router(rag_router, prefix="/api", tags=["RAG"])

# ===== MONTAR INTERFAZ GRADIO =====
try:
    app = gr.mount_gradio_app(app, gradio_iface, path="/gradio")
    logger.info("✅ Gradio interface mounted at /gradio")
except Exception as e:
    logger.error(f"❌ Failed to mount Gradio interface: {e}")
    raise

# ===== MANEJO DE ERRORES GLOBAL =====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Manejo global de errores"""
    logger.error(f"Global error from {request.client.host}: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }

# ===== EJECUTAR APLICACIÓN =====
if __name__ == "__main__":
    # Configuración para desarrollo
    uvicorn.run(
        app="main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        reload_dirs=["app"],
        log_level="info"
    )