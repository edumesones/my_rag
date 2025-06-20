import gradio as gr

from app.utils.rag_functions import get_list_openai_models, retrieve_only

with gr.Blocks(
    title="Document Retrieval System",
    theme=gr.themes.Soft(),
    css="""
    .retrieved-doc {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .doc-header {
        color: #2E7D32;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 8px;
    }
    .doc-content {
        color: #424242;
        line-height: 1.6;
        font-size: 14px;
    }
    .search-stats {
        background: #E3F2FD;
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
        text-align: center;
        color: #1976D2;
        font-weight: bold;
    }
    .no-results {
        background: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 15px;
        border-radius: 8px;
        color: #E65100;
        text-align: center;
        font-weight: bold;
    }
    """
) as retrieve_only_interface:
    
    # Header
    gr.Markdown("""
    # 🔍 **Document Retrieval System**
    ### Busca información relevante en la base de documentos
    
    *Escribe tu consulta y encuentra los fragmentos de documentos más relevantes.*
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            msg = gr.Textbox(
                placeholder="Ejemplo: ¿Qué experiencia tiene en Python?",
                label="💭 Tu consulta",
                lines=2,
                container=True
            )
        with gr.Column(scale=1):
            submit_button = gr.Button("🔍 Buscar", variant="primary", size="lg")
    
    # Results area
    results_area = gr.HTML(label="📋 Resultados de búsqueda")
    
    # Clear button
    clear_button = gr.Button("🗑️ Limpiar", variant="secondary")
    
    # Advanced settings in collapsible section
    with gr.Accordion("⚙️ Configuración avanzada", open=False):
        with gr.Row():
            k_relevant = gr.Slider(
                minimum=1, 
                maximum=10, 
                value=5, 
                step=1, 
                label="📊 Número de documentos a recuperar",
                info="Más documentos = más contexto, pero respuestas más largas"
            )
            
        with gr.Row():
            model_name = gr.Dropdown(
                choices=get_list_openai_models(),
                value=get_list_openai_models()[0] if get_list_openai_models() else "gpt-3.5-turbo",
                label="🤖 Modelo de embeddings",
                info="Modelo usado para generar embeddings de la consulta"
            )

    def format_retrieved_documents(docs, query, k):
        """Format the retrieved documents as beautiful HTML."""
        if not docs or len(docs) == 0:
            return f"""
            <div class="no-results">
                <h3>🚫 No se encontraron documentos</h3>
                <p>No se encontraron documentos relevantes para: "<em>{query}</em>"</p>
                <p>💡 Intenta reformular tu consulta o usar palabras clave diferentes.</p>
            </div>
            """
        
        # Search statistics
        stats_html = f"""
        <div class="search-stats">
            📊 <strong>Búsqueda completada:</strong> Se encontraron {len(docs)} documento(s) relevante(s) para "<em>{query}</em>"
        </div>
        """
        
        # Format each document
        docs_html = ""
        for i, doc in enumerate(docs, 1):
            # Truncate very long documents
            display_content = doc
            if len(doc) > 500:
                display_content = doc[:500] + "..."
            
            # Highlight query terms (simple approach)
            query_words = query.lower().split()
            highlighted_content = display_content
            for word in query_words:
                if len(word) > 2:  # Only highlight words longer than 2 chars
                    highlighted_content = highlighted_content.replace(
                        word, f"<mark style='background-color: #FFEB3B; padding: 2px 4px; border-radius: 3px;'>{word}</mark>"
                    )
                    # Also try capitalized version
                    highlighted_content = highlighted_content.replace(
                        word.capitalize(), f"<mark style='background-color: #FFEB3B; padding: 2px 4px; border-radius: 3px;'>{word.capitalize()}</mark>"
                    )
            
            docs_html += f"""
            <div class="retrieved-doc">
                <div class="doc-header">
                    📄 Documento {i} de {len(docs)}
                </div>
                <div class="doc-content">
                    {highlighted_content}
                </div>
            </div>
            """
        
        return stats_html + docs_html

    def search_documents(query: str, k_value: int):
        """Search for documents and return formatted HTML."""
        if not query.strip():
            return """
            <div class="no-results">
                <h3>💭 Escribe una consulta</h3>
                <p>Por favor, escribe una pregunta o consulta para buscar en los documentos.</p>
            </div>
            """
        
        try:
            # Retrieve documents
            retrieved_docs = retrieve_only(query=query.strip(), k=k_value)
            
            # Format as beautiful HTML
            formatted_result = format_retrieved_documents(retrieved_docs, query, k_value)
            
            return formatted_result
            
        except Exception as e:
            return f"""
            <div class="no-results">
                <h3>❌ Error en la búsqueda</h3>
                <p>Ocurrió un error: {str(e)}</p>
                <p>💡 Intenta de nuevo en unos segundos.</p>
            </div>
            """

    def clear_results():
        """Clear all results."""
        return "", ""

    # Event handlers
    submit_button.click(
        fn=search_documents,
        inputs=[msg, k_relevant],
        outputs=[results_area]
    )
    
    msg.submit(
        fn=search_documents,
        inputs=[msg, k_relevant],
        outputs=[results_area]
    )
    
    clear_button.click(
        fn=clear_results,
        inputs=[],
        outputs=[msg, results_area]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["¿Qué experiencia tiene en Python?"],
            ["¿Cuáles son sus habilidades principales?"],
            ["¿Ha trabajado con machine learning?"],
            ["¿Qué proyectos ha realizado?"],
            ["¿Cuál es su formación académica?"]
        ],
        inputs=[msg],
        label="💡 Ejemplos de consultas"
    )

# Make it available for import
retrieve_only_compilation = retrieve_only_interface