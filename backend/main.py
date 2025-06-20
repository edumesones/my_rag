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
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from fastapi import FastAPI, Request

from app.api.routers.rag import rag_router
from app.gradio_ui.ui import gradio_iface
from instrument import instrument
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
limiter = Limiter(key_func=get_remote_address)

# Instrumentación DSPy/OpenTelemetry
do_not_instrument = os.getenv("INSTRUMENT_DSPY", "true") == "false"
if not do_not_instrument:
    instrument()

app = FastAPI(title="DSPy x FastAPI")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Configurar CORS en desarrollo env
environment = os.getenv("ENVIRONMENT", "dev")
if environment == "dev":
    logger = logging.getLogger("uvicorn")
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
@app.get("/")
async def root(request: Request):
    logger.info(f"Root access from IP: {request.client.host}")
    return {"message": "DSPy x FastAPI is running", "status": "healthy"}

@app.get("/health")
async def health_check(request: Request):
    logger.info(f"Health check from IP: {request.client.host}")
    return {"status": "healthy", "environment": environment}

# ✅ ESTADÍSTICAS - Nueva ruta para ver uso
@app.get("/stats")
@limiter.limit("5/minute")  # Solo 5 consultas por minuto
async def get_stats(request: Request):
    """Ver estadísticas de uso del sistema."""
    try:
        from app.utils.cost_monitor import get_usage_stats
        stats = get_usage_stats()
        logger.info(f"Stats requested from IP: {request.client.host}")
        return stats
    except ImportError:
        logger.warning("Cost monitor not available")
        return {"error": "Stats module not available"}
# Incluir rutas RAG
app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])

# Montar UI de Gradio en /gradio
app = gr.mount_gradio_app(app, gradio_iface, path="/gradio")

# Ejecutar con uvicorn
if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", reload=True)