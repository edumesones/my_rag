import gradio as gr
import pandas as pd
import os
from pathlib import Path

from app.utils.rag_functions import get_list_openai_models, compile_rag, load_documents_to_chroma, get_collections_info
from app.utils.models import QAItem

with gr.Blocks(title="Document Management & Pipeline Optimization", theme=gr.themes.Soft()) as compile_pipeline:
    gr.Markdown("""
    # 📚 Document Management & DSPy Pipeline Optimization
    
    ## 🎯 Two Main Functions:
    
    ### 📄 **Document Loading**
    Upload PDF, TXT, or DOCX files to ChromaDB for RAG queries
    
    ### 🚀 **Pipeline Optimization** 
    Train DSPy pipeline using CSV with question-answer pairs
    """)
    
    with gr.Tabs():
        # ===== TAB 1: DOCUMENT MANAGEMENT =====
        with gr.TabItem("📄 Document Management"):
            gr.Markdown("### Upload documents to make them available for RAG queries")
            
            # File upload section
            with gr.Row():
                doc_files = gr.File(
                    label="📁 Select Documents", 
                    file_types=[".pdf", ".txt", ".docx"],
                    file_count="multiple",
                    height=120
                )
            
            # Configuration section
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("⚙️ Document Processing Settings", open=True):
                        chunk_size = gr.Slider(
                            minimum=200, 
                            maximum=2000, 
                            value=1000, 
                            step=100,
                            label="📏 Chunk Size (tokens)",
                            info="Size of each text chunk for better retrieval"
                        )
                        
                        chunk_overlap = gr.Slider(
                            minimum=0, 
                            maximum=500, 
                            value=200, 
                            step=50,
                            label="🔄 Chunk Overlap (tokens)",
                            info="Overlap between chunks to maintain context"
                        )
                        
                        chunking_strategy = gr.Dropdown(
                            choices=["recursive", "sentence", "fixed_size"],
                            value="recursive",
                            label="📋 Chunking Strategy",
                            info="How to split documents into chunks"
                        )
                
                with gr.Column(scale=1):
                    collection_info = gr.HTML(
                        label="📊 Database Status",
                        value="<div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>No collections found</div>"
                    )
            
            # Action buttons
            with gr.Row():
                load_docs_button = gr.Button("📥 Load Documents to Database", variant="primary", size="lg")
                refresh_info_button = gr.Button("🔄 Refresh Database Info", variant="secondary")
            
            # Status display
            doc_status = gr.Textbox(
                label="📋 Processing Status", 
                interactive=False,
                placeholder="Select documents and click 'Load Documents' to start...",
                lines=3
            )
            
        # ===== TAB 2: PIPELINE OPTIMIZATION =====
        with gr.TabItem("🚀 Pipeline Optimization"):
            gr.Markdown("""
            ### Train DSPy pipeline with question-answer pairs
            
            Upload a CSV file with `question` and `answer` columns:
            ```csv
            question,answer
            "What is machine learning?","Machine learning is a subset of AI..."
            "How does RAG work?","RAG combines retrieval and generation..."
            ```
            """)
            
            # CSV upload
            csv_file = gr.File(
                label="📊 CSV Training Data", 
                file_types=[".csv"],
                height=80
            )
            qa_preview = gr.Dataframe(
                label="📋 Data Preview",
                
                interactive=False
            )

            # Model parameters
            with gr.Accordion("🤖 Model Parameters", open=True):
                with gr.Row():
                    model_name = gr.Dropdown(
                        get_list_openai_models(),
                        value=get_list_openai_models()[0] if get_list_openai_models() else "gpt-3.5-turbo",
                        label="🧠 OpenAI Model",
                        info="Model for pipeline optimization"
                    )

                    max_tokens = gr.Slider(
                        minimum=128, maximum=2048, value=512, label="📝 Max Tokens"
                    )

                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0, maximum=2, step=0.1, value=0.1, label="🌡️ Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0, maximum=1, step=0.1, value=1, label="🎯 Top-p"
                    )
                    k_retrieve = gr.Slider(
                        minimum=1, maximum=20, value=5, label="🔍 K (docs to retrieve)"
                    )

            # Compile button
            compile_button = gr.Button("🚀 Compile Pipeline", variant="primary", size="lg")
            compile_status = gr.Textbox(
                label="📊 Compilation Status", 
                interactive=False,
                placeholder="Upload CSV and click 'Compile Pipeline' to start training...",
                lines=3
            )

    # Clear button
    with gr.Row():
        clear_all = gr.ClearButton(
            [doc_files, csv_file, qa_preview, doc_status, compile_status], 
            value="🗑️ Clear All"
        )

    # ===== EVENT HANDLERS =====
    
    def update_collection_info():
        """Update database collection information"""
        try:
            info = get_collections_info()
            if info["collections"]:
                html_content = "<div style='padding: 10px; background: #e8f5e8; border-radius: 5px; border-left: 4px solid #4CAF50;'>"
                html_content += f"<strong>📊 Database Status:</strong><br>"
                html_content += f"<strong>Total Collections:</strong> {info['total_collections']}<br>"
                html_content += f"<strong>Total Documents:</strong> {info['total_documents']}<br><br>"
                
                for coll in info["collections"][:3]:  # Show first 3 collections
                    html_content += f"• <strong>{coll['name']}</strong>: {coll['count']} docs<br>"
                
                if len(info["collections"]) > 3:
                    html_content += f"... and {len(info['collections']) - 3} more collections<br>"
                
                html_content += "</div>"
            else:
                html_content = "<div style='padding: 10px; background: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;'>No collections found. Upload documents to create your first collection.</div>"
            
            return html_content
        except Exception as e:
            return f"<div style='padding: 10px; background: #f8d7da; border-radius: 5px; border-left: 4px solid #dc3545;'>Error: {str(e)}</div>"
    
    def load_documents(files, chunk_sz, overlap, strategy):
        """Load documents to ChromaDB"""
        if not files:
            return "❌ No files selected. Please select at least one document."
        
        try:
            result = load_documents_to_chroma(
                files=files,
                chunk_size=chunk_sz,
                chunk_overlap=overlap,
                chunking_strategy=strategy
            )
            return f"✅ Success!\n\n{result}"
        except Exception as e:
            return f"❌ Error loading documents:\n{str(e)}"
    
    def preview_csv(csv_file):
        """Preview CSV data"""
        if not csv_file:
            return None
        try:
            df = pd.read_csv(csv_file.name)
            if 'question' not in df.columns or 'answer' not in df.columns:
                gr.Warning("CSV must have 'question' and 'answer' columns")
                return None
            return df.head(10)  # Show first 10 rows
        except Exception as e:
            gr.Warning(f"Error reading CSV: {e}")
            return None

    def run_compile_pipeline(csv_file, model_name_str, temperature, top_p, max_tokens, k):
        """Compile DSPy pipeline"""
        if not csv_file:
            return "❌ No CSV file selected. Please upload a training file."
            
        try:
            df = pd.read_csv(csv_file.name)
            
            # Validate CSV structure
            if 'question' not in df.columns or 'answer' not in df.columns:
                return "❌ CSV must have 'question' and 'answer' columns"
            
            # Convert to QAItem objects
            items_list = []
            for _, row in df.iterrows():
                try:
                    qa_item = QAItem(question=str(row['question']), answer=str(row['answer']))
                    items_list.append(qa_item)
                except Exception as e:
                    return f"❌ Error validating row: {e}"

            if not items_list:
                return "❌ No valid question-answer pairs found in CSV"

            # Compile pipeline
            compile_response = compile_rag(
                items=items_list,
                model_name=model_name_str,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                k=k,
            )
            
            return f"✅ Pipeline compiled successfully!\n\nProcessed {len(items_list)} training examples.\nModel: {model_name_str}\nThe pipeline is now optimized for better RAG performance."
            
        except Exception as e:
            return f"❌ Compilation error:\n{str(e)}"

    # Connect event handlers
    load_docs_button.click(
        load_documents,
        inputs=[doc_files, chunk_size, chunk_overlap, chunking_strategy],
        outputs=[doc_status]
    ).then(
        update_collection_info,
        outputs=[collection_info]
    )
    
    refresh_info_button.click(
        update_collection_info,
        outputs=[collection_info]
    )
    
    csv_file.upload(
        preview_csv, 
        inputs=[csv_file], 
        outputs=[qa_preview]
    )
    
    compile_button.click(
        run_compile_pipeline,
        inputs=[csv_file, model_name, temperature, top_p, max_tokens, k_retrieve],
        outputs=[compile_status]
    )
    
    # Initialize collection info on load
    compile_pipeline.load(
        update_collection_info,
        outputs=[collection_info]
    )