import gradio as gr

from app.utils.rag_functions import get_list_openai_models, retrieve_only

with gr.Blocks(title="Retrieve only query") as retrieve_only_compilation:
    chatbot = gr.Chatbot(label="Retrieve only query", show_copy_button=True)

    with gr.Row():
        msg = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )
        submit_button = gr.Button("Submit")
        clear = gr.ClearButton([msg, chatbot], value="Clear chat")

    with gr.Accordion("Model parameters", open=False):
        with gr.Row():
            model_name = gr.Dropdown(
                get_list_openai_models(),
                value=get_list_openai_models()[0],
                label="OpenAI model",
                info="List of models available on your machine.",
            )

            max_tokens = gr.Slider(
                minimum=128, maximum=2048, value=150, label="max-tokens"
            )

        with gr.Row():
            temperature = gr.Slider(
                minimum=0, maximum=2, step=0.1, value=0.1, label="temperature"
            )
            top_p = gr.Slider(minimum=0, maximum=1, step=0.1, value=1, label="top-p")
            
            # AGREGAR: Slider para k_relevant
            k_relevant = gr.Slider(
                minimum=1, maximum=20, value=5, step=1, label="Number of documents (k)"
            )

    def respond(
        message: str,
        chat_history: list,
        k_value: int,  # Renombrado para claridad
    ):
        # retrieve_only devuelve List[str], no un objeto con .answer
        retrieved_docs = retrieve_only(
            query=message,
            k=k_value
        )
        
        # Formatear los documentos recuperados
        if retrieved_docs:
            bot_message = f"📄 **Documentos encontrados ({len(retrieved_docs)}):**\n\n"
            for i, doc in enumerate(retrieved_docs, 1):
                bot_message += f"**Documento {i}:**\n{doc}\n\n---\n\n"
        else:
            bot_message = "❌ No se encontraron documentos relevantes."

        chat_history.append((message, bot_message))
        return "", chat_history

    # CORREGIR: Agregar k_relevant a los inputs
    msg.submit(
        respond,
        [msg, chatbot, k_relevant],  # ← Agregar k_relevant
        [msg, chatbot],
    )
    submit_button.click(
        respond,
        [msg, chatbot, k_relevant],  # ← Agregar k_relevant
        [msg, chatbot],
    )
